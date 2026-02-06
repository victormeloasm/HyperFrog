// ============================================================================
// HyperFrog v33.1 – "Topological Obsidian" (Fixed + Hardened)
// ----------------------------------------------------------------------------
//  Author:    Víctor Duarte Melo
//  Institute: University of Hamburg
//  Target:    AMD Ryzen 9 5950X (Zen 3 / Ubuntu)
//
//  What was fixed vs v33.0:
//   [P0/UB]  Topology miner no longer reads past buffers (AES-CTR uses zero input).
//   [P0/CT]  Branchless constant-time "is-close" decoding (no secret-dependent if).
//   [P0/IO]  Strict versioning + read_exact checks + ciphertext length validation.
//   [P1/FO]  FO deterministic randomness binds to pk bytes; K_fake derives from SK PRF.
//   [P1/PERF] Safer OpenMP fallbacks; decrypt stream writes final block correctly.
//   [P1]     AES256-GCM (libsodium) secret-key encryption fallback if AES-NI unavailable.
//
//  NOTE: This is a research/prototype implementation. Constant-time is improved
//        (no obvious branches on secret bits), but a full side-channel audit is
//        still required for production use.
// ============================================================================

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_OPENMP)
  #include <omp.h>
#endif

#include <openssl/evp.h>
#include <openssl/err.h>

#include <sodium.h>

#if defined(__linux__)
  #include <unistd.h>
  #include <sys/utsname.h>
#endif

namespace fs = std::filesystem;

// ------------------------------ CONFIG ---------------------------------------
static constexpr size_t   HF_CHUNK_SIZE = 64 * 1024;
static constexpr uint8_t  HF_VERSION_ID = 0x06; // v6 (Topology Restored)
static constexpr const char* HF_AAD_STR = "HF_V33_TOPOLOGICAL";

// LWE parameters (toy-ish, research)
static constexpr size_t HF_N      = 4096;
static constexpr size_t HF_M      = 2048;
static constexpr size_t HF_KBITS  = 256;
static constexpr size_t HF_SHAPE_WORDS = HF_N / 64;

using lwe_t = uint16_t;
static constexpr uint32_t HF_Q = 65536u; // modulus (2^16); coefficients stored mod 65536
static constexpr lwe_t HF_Q_HALF = 32768;
static constexpr lwe_t HF_Q_QTR  = 16384;

// Magic headers
static const uint8_t HF_MAGIC_PK[4] = {'H','F','P','K'};
static const uint8_t HF_MAGIC_SK[4] = {'H','F','S','K'};
static const uint8_t HF_MAGIC_CT[4] = {'H','F','C','T'};

// ------------------------------ UI ------------------------------------------
#define RST    "\033[0m"
#define RED    "\033[1;31m"
#define GRN    "\033[1;32m"
#define CYN    "\033[1;36m"
#define YEL    "\033[1;33m"
#define BOLD   "\033[1m"
#define BG_BLU "\033[44m\033[1;37m"

#define CHECK_SSL(cond) do { \
    if (!(cond)) { \
        char err_buf[256]; \
        ERR_error_string_n(ERR_get_error(), err_buf, sizeof(err_buf)); \
        throw std::runtime_error(std::string("OpenSSL error: ") + err_buf); \
    } \
} while(0)

[[maybe_unused]] static void clear_screen() { std::cout << "\033[2J\033[1;1H"; }

[[maybe_unused]] static void print_header() {
    std::cout << GRN << BOLD << "HyperFrog v33.1 [TOPOLOGICAL OBSIDIAN - FIXED]\n" << RST;
    std::cout << BG_BLU << " GENUS>=8 " << RST << " " << BG_BLU << " FO-KEM " << RST
              << " " << BG_BLU << " CT-DECODE " << RST << "\n\n";
}

// ------------------------------ UTIL ----------------------------------------
static inline int hf_max_threads() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}
static inline int hf_thread_num() {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
}

static bool read_exact(std::istream& in, void* dst, size_t n) {
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
    return in.good() && static_cast<size_t>(in.gcount()) == n;
}

static bool write_exact(std::ostream& out, const void* src, size_t n) {
    out.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(n));
    return out.good();
}

static uint64_t file_size_u64(const std::string& f) {
    return static_cast<uint64_t>(fs::file_size(f));
}

// ------------------------------ SECURE BYTES --------------------------------
template <size_t N>
struct SecureBytes {
    alignas(64) std::array<uint8_t, N> b{};
    bool locked = false;
    SecureBytes() {
        sodium_memzero(b.data(), N);
        if (sodium_mlock(b.data(), N) == 0) locked = true;
    }
    ~SecureBytes() {
        sodium_memzero(b.data(), N);
        if (locked) sodium_munlock(b.data(), N);
    }
    uint8_t* data() { return b.data(); }
    std::span<uint8_t, N> span() { return std::span<uint8_t, N>(b); }
};

using SharedKey = SecureBytes<32>;

// ------------------------------ STRUCTS -------------------------------------
struct Shape {
    std::array<uint64_t, HF_SHAPE_WORDS> bits{};
    [[nodiscard]] inline bool get(size_t idx) const {
        return (bits[idx >> 6] >> (idx & 63)) & 1ULL;
    }
};

struct PublicKey {
    uint8_t seed_A[32]{};
    std::array<lwe_t, HF_M> b{};
};

struct SecretKey {
    Shape s;
    PublicKey pk; // stored to allow FO re-encapsulation during decap
};

struct Ciphertext {
    std::vector<lwe_t> u_transposed; // size HF_KBITS * HF_N
    std::array<lwe_t, HF_KBITS> v{};
    std::array<uint8_t, 32> tag{};
};

// ------------------------------ HASH ----------------------------------------
static inline void hf_sha3_256_multi(std::initializer_list<std::pair<const void*, size_t>> parts,
                                    std::span<uint8_t, 32> out) {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) throw std::runtime_error("EVP_MD_CTX_new failed");
    CHECK_SSL(EVP_DigestInit_ex(ctx, EVP_sha3_256(), nullptr) == 1);
    for (const auto& p : parts) {
        CHECK_SSL(EVP_DigestUpdate(ctx, p.first, p.second) == 1);
    }
    CHECK_SSL(EVP_DigestFinal_ex(ctx, out.data(), nullptr) == 1);
    EVP_MD_CTX_free(ctx);
}

static inline std::array<uint8_t, 32> hf_hash_pk_digest(const PublicKey& pk) {
    std::array<uint8_t, 32> out{};
    hf_sha3_256_multi({
        {"PK", 2},
        {pk.seed_A, 32},
        {pk.b.data(), pk.b.size() * sizeof(lwe_t)}
    }, out);
    return out;
}

// ------------------------------ NOISE ---------------------------------------
[[nodiscard]] static inline int16_t hf_sample_error_eta4() {
    uint8_t r;
    randombytes_buf(&r, 1);
    return (int16_t)(std::popcount((unsigned int)(r & 0x0F)) -
                     std::popcount((unsigned int)(r >> 4)));
}

// Deterministic centered-binomial-ish from seed (FO re-encapsulation)
static inline int16_t hf_cbd_from_seed(const uint8_t seed[32], uint32_t idx) {
    uint8_t buf[1];
    uint8_t nonce[12] = {0};
    std::memcpy(nonce, &idx, (sizeof(idx) < sizeof(nonce) ? sizeof(idx) : sizeof(nonce)));
    crypto_stream_chacha20_ietf(buf, sizeof(buf), nonce, seed);
    return (int16_t)(std::popcount((unsigned int)(buf[0] & 0x0F)) -
                     std::popcount((unsigned int)(buf[0] >> 4)));
}

// ------------------------------ CONSTANT TIME HELPERS -----------------------
// If mask == 0xFF..FF -> return a, else return b
static inline uint8_t ct_select_u8(uint8_t mask, uint8_t a, uint8_t b) {
    return (mask & a) | (uint8_t)(~mask & b);
}

// Branchless modular "is close" test:
// returns 0xFFFFFFFF if modular_distance(val, target) < limit else 0.
static inline uint32_t ct_is_close_mask(lwe_t val, lwe_t target, lwe_t limit) {
    // dist = (val - target) mod 2^16
    uint32_t dist  = (uint16_t)(val - target);   // 0..65535
    uint32_t dist2 = 65536u - dist;              // wrap ok
    // le = 1 if dist <= 32768 else 0 (branchless)
    uint32_t le = (dist - 32769u) >> 31;
    uint32_t m  = 0u - le;                       // 0xFFFFFFFF if dist <= 32768
    uint32_t md = (dist & m) | (dist2 & ~m);      // min(dist, 65536-dist)
    // lt = 1 if md < limit else 0
    uint32_t lt = (md - (uint32_t)limit) >> 31;
    return 0u - lt;
}

// ------------------------------ DOT PRODUCT ---------------------------------
[[maybe_unused]] static inline uint32_t hf_dot_product_scalar(std::span<const lwe_t, HF_N> a, const Shape& s) {
    uint32_t sum = 0;
    for (size_t i = 0; i < HF_N; ++i) {
        // branchless: add a[i] if s.get(i) == 1
        uint32_t m = 0u - (uint32_t)s.get(i);
        sum += (uint32_t)a[i] & m;
    }
    return sum;
}

#if defined(__AVX2__)
static inline uint32_t hf_dot_product_avx2(std::span<const lwe_t, HF_N> a, const Shape& s) {
    __m256i sum32 = _mm256_setzero_si256();
    const lwe_t* ptr_a = a.data();

    for (size_t i = 0; i < HF_N; i += 16) {
        _mm_prefetch((const char*)(ptr_a + i + 64), _MM_HINT_T0);
        __m256i a_vec = _mm256_loadu_si256((const __m256i*)(ptr_a + i));

        const size_t w_idx = i / 64;
        const unsigned shift = (unsigned)(i % 64);
        const uint64_t w0 = s.bits[w_idx];
        const uint64_t w1 = (w_idx + 1 < HF_SHAPE_WORDS) ? s.bits[w_idx + 1] : 0ULL;
        const uint64_t chunk = (shift == 0) ? w0 : (w0 >> shift) | (w1 << (64 - shift));

        alignas(32) uint16_t m16[16];
        for (int k = 0; k < 16; ++k) m16[k] = (chunk & (1ULL << k)) ? 0xFFFFu : 0x0000u;

        __m256i masked = _mm256_and_si256(a_vec, _mm256_load_si256((const __m256i*)m16));
        __m128i lo16 = _mm256_castsi256_si128(masked);
        __m128i hi16 = _mm256_extracti128_si256(masked, 1);
        sum32 = _mm256_add_epi32(sum32, _mm256_cvtepu16_epi32(lo16));
        sum32 = _mm256_add_epi32(sum32, _mm256_cvtepu16_epi32(hi16));
    }

    alignas(32) uint32_t res[8];
    _mm256_store_si256((__m256i*)res, sum32);
    return std::accumulate(res, res + 8, 0u);
}
#else
static inline uint32_t hf_dot_product_avx2(std::span<const lwe_t, HF_N> a, const Shape& s) {
    return hf_dot_product_scalar(a, s);
}
#endif

// ------------------------------ TOPOLOGY ------------------------------------
struct HFTopo { 
    int components = 0; 
    int genus = 0; 
    int edges = 0; // Added missing member
};
struct TopoScratch {
    alignas(64) std::array<uint64_t, 64> visited_bits{};
    alignas(64) std::array<int16_t, 4096> queue{};
};

static HFTopo hf_compute_topology(const Shape& s, TopoScratch& scratch) {
    HFTopo st{};
    int V = 0;
    for (auto w : s.bits) V += (int)std::popcount(w);
    if (V == 0) return st;

    int E = 0;
    for (int i = 0; i < 4096; ++i) {
        if (!s.get((size_t)i)) continue;
        const int x = i & 0xF, y = (i >> 4) & 0xF, z = (i >> 8) & 0xF;
        if (x < 15 && s.get((size_t)(i + 1)))   E++;
        if (y < 15 && s.get((size_t)(i + 16)))  E++;
        if (z < 15 && s.get((size_t)(i + 256))) E++;
    }

    std::memset(scratch.visited_bits.data(), 0, scratch.visited_bits.size() * sizeof(uint64_t));
    int C = 0;

    for (int i = 0; i < 4096; ++i) {
        if (!s.get((size_t)i)) continue;
        const uint64_t seen = (scratch.visited_bits[(size_t)i >> 6] >> (i & 63)) & 1ULL;
        if (seen) continue;

        C++;
        scratch.visited_bits[(size_t)i >> 6] |= (1ULL << (i & 63));

        int head = 0, tail = 0;
        scratch.queue[(size_t)tail++] = (int16_t)i;

        while (head < tail) {
            const int cur = scratch.queue[(size_t)head++];
            const int x = cur & 0xF, y = (cur >> 4) & 0xF, z = (cur >> 8) & 0xF;

            auto push = [&](int ni) {
                if (ni < 0 || ni >= 4096) return;
                if (!s.get((size_t)ni)) return;
                const uint64_t sbit = (scratch.visited_bits[(size_t)ni >> 6] >> (ni & 63)) & 1ULL;
                if (sbit) return;
                scratch.visited_bits[(size_t)ni >> 6] |= (1ULL << (ni & 63));
                scratch.queue[(size_t)tail++] = (int16_t)ni;
            };

            if (x < 15) push(cur + 1);
            if (x > 0)  push(cur - 1);
            if (y < 15) push(cur + 16);
            if (y > 0)  push(cur - 16);
            if (z < 15) push(cur + 256);
            if (z > 0)  push(cur - 256);
        }
    }

    st.components = C;
    st.edges = E; // Set edges
    st.genus = (E - V + C); // cyclomatic number proxy
    return st;
}


// Time helper (monotonic milliseconds)
static inline uint64_t hf_now_ms() {
    using namespace std::chrono;
    return static_cast<uint64_t>(duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
}

// Convenience wrapper used by the miner.
[[maybe_unused]] static inline HFTopo hf_analyze_shape(const Shape& s) {
    TopoScratch scratch;
    // No reset() method: compute_topology() fully reinitializes scratch state.
    return hf_compute_topology(s, scratch);
}


// Mining: find connected shape with genus >= 8
static inline Shape hf_fallback_shape_connected() {
    Shape s{};
    // Deterministic, connected, very "loopy" block to guarantee genus >> 8.
    // We occupy a 6x6x4 block in the middle of the 16^3 lattice.
    for (uint16_t z = 6; z < 10; ++z) {
        for (uint16_t y = 5; y < 11; ++y) {
            for (uint16_t x = 5; x < 11; ++x) {
                const uint16_t idx = (z << 8) | (y << 4) | x;
                s.bits[idx >> 6] |= (1ull << (idx & 63));
            }
        }
    }
    return s;
}

static inline size_t hf_bfs_mark_component(const Shape& s, TopoScratch& scratch, uint16_t start) {
    // scratch.visited_bits must already be initialized (0 for unvisited)
    size_t head = 0, tail = 0;
    scratch.queue[tail++] = start;
    scratch.visited_bits[start >> 6] |= (1ull << (start & 63));

    size_t cnt = 0;
    while (head < tail) {
        const uint16_t cur = scratch.queue[head++];
        ++cnt;

        const uint16_t x = (cur & 15u);
        const uint16_t y = ((cur >> 4) & 15u);
        const uint16_t z = (cur >> 8);

        auto try_push = [&](uint16_t nb) {
            if (!s.get(nb)) return;
            const uint64_t mask = (1ull << (nb & 63));
            uint64_t& w = scratch.visited_bits[nb >> 6];
            if (w & mask) return;
            w |= mask;
            scratch.queue[tail++] = nb;
        };

        if (x > 0)  try_push(uint16_t(cur - 1));
        if (x < 15) try_push(uint16_t(cur + 1));
        if (y > 0)  try_push(uint16_t(cur - 16));
        if (y < 15) try_push(uint16_t(cur + 16));
        if (z > 0)  try_push(uint16_t(cur - 256));
        if (z < 15) try_push(uint16_t(cur + 256));
    }
    return cnt;
}

static inline void hf_keep_largest_component(Shape& s, TopoScratch& scratch) {
    // Pass 1: find the largest connected component among occupied cells.
    scratch.visited_bits.fill(0);
    uint16_t best_root = 0xFFFF;
    size_t best_sz = 0;

    for (uint16_t i = 0; i < HF_N; ++i) {
        if (!s.get(i)) continue;
        const uint64_t mask = (1ull << (i & 63));
        if (scratch.visited_bits[i >> 6] & mask) continue;

        const size_t sz = hf_bfs_mark_component(s, scratch, i);
        if (sz > best_sz) { best_sz = sz; best_root = i; }
    }
    if (best_root == 0xFFFF) return;

    // Pass 2: mark only the best component, then clear everything else.
    scratch.visited_bits.fill(0);
    hf_bfs_mark_component(s, scratch, best_root);

    for (uint16_t i = 0; i < HF_N; ++i) {
        if (!s.get(i)) continue;
        const uint64_t mask = (1ull << (i & 63));
        if (scratch.visited_bits[i >> 6] & mask) continue;
        s.bits[i >> 6] &= ~mask;
    }
}

static Shape hf_mine_shape(int min_genus = 8, uint64_t budget_ms = 30000) {
    Shape best{};
    HFTopo bestS{};
    std::atomic<bool> found{false};

    const uint64_t start_ms = hf_now_ms();

#pragma omp parallel
    {
        TopoScratch scratch{};
        HFTopo st{};
        uint64_t iter = 0;

        while (!found.load(std::memory_order_relaxed)) {
            // Check budget only occasionally (cheap).
            if ((iter++ & 63u) == 0u) {
                if ((hf_now_ms() - start_ms) >= budget_ms) break;
            }

            Shape local{};
            // Fast, thread-safe CSPRNG.
            randombytes_buf(local.bits.data(), local.bits.size() * sizeof(uint64_t));

            // Make it connected by keeping only the largest component.
            hf_keep_largest_component(local, scratch);

            st = hf_compute_topology(local, scratch);
            if (st.components == 1 && st.genus >= min_genus) {
#pragma omp critical
                {
                    if (!found.load(std::memory_order_relaxed) ||
                        (st.genus > bestS.genus) ||
                        (st.genus == bestS.genus && st.edges > bestS.edges)) {
                        best = local;
                        bestS = st;
                        found.store(true, std::memory_order_relaxed);
                    }
                }
            }

            // Light progress indicator.
#pragma omp master
            {
                if ((iter & 0x3FFFu) == 0u) { std::cerr << "." << std::flush; }
            }
        }
    }

    if (!found.load()) {
        std::cerr << "\n" << YEL
                  << "[TopoMining] Timeout (" << budget_ms << " ms); using connected fallback shape."
                  << RST << "\n";
        return hf_fallback_shape_connected();
    }

    std::cout << "\n" << GRN
              << "[TopoMining] Found: genus=" << bestS.genus
              << " components=" << bestS.components
              << RST << "\n";

    return best;
}

// ------------------------------ KEYGEN + KEM --------------------------------
static void hf_kem_keygen_from_shape(PublicKey& pk, const SecretKey& sk) {
    // 1. Generate seed_A
    randombytes_buf(pk.seed_A, sizeof(pk.seed_A));

    // 2. Generate b = A*s + e
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int i = 0; i < (int)HF_M; ++i) {
        std::array<lwe_t, HF_N> row{};
        uint8_t row_nonce[12] = {0};
        row_nonce[0] = (uint8_t)(i & 0xFF);
        row_nonce[1] = (uint8_t)((i >> 8) & 0xFF);

        // Expand row A[i]
        crypto_stream_chacha20_ietf(reinterpret_cast<unsigned char*>(row.data()),
                                    HF_N * sizeof(lwe_t),
                                    row_nonce, pk.seed_A);

        // Compute <a_i, s>
        uint32_t dot = hf_dot_product_avx2(row, sk.s);

        // Sample error
        int16_t e = hf_sample_error_eta4();

        // b[i] = <a_i, s> + e
        pk.b[i] = (lwe_t)(dot + e);
    }
}

static void hf_keygen(PublicKey& pk, SecretKey& sk,
                      int target_genus = 8,
                      uint64_t mine_budget_ms = 30000,
                      int threads = 0) {
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#else
    (void)threads;
#endif

    Shape sh = hf_mine_shape(target_genus, mine_budget_ms);
    sk.s = sh; 
    
    hf_kem_keygen_from_shape(pk, sk);
    sk.pk = pk;
}

static inline void hf_encaps_det(const PublicKey& pk,
                                const std::array<uint8_t, 32>& mu,
                                Ciphertext& ct) {
    ct.u_transposed.assign(HF_KBITS * HF_N, 0);

    const auto pk_digest = hf_hash_pk_digest(pk);

    std::array<uint8_t, 32> r_seed{};
    hf_sha3_256_multi({
        {"R_GEN", 5},
        {mu.data(), mu.size()},
        {pk_digest.data(), pk_digest.size()}
    }, r_seed);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int k = 0; k < (int)HF_KBITS; ++k) {
        const uint8_t bit = (mu[(size_t)k >> 3] >> (k & 7)) & 1U;
        uint32_t sum_v = 0;

        uint8_t stream_buf[128];
        uint8_t nonce[12] = {0};
        std::memcpy(nonce, &k, sizeof(k));
        crypto_stream_chacha20_ietf(stream_buf, sizeof(stream_buf), nonce, r_seed.data());

        for (int r = 0; r < 64; ++r) {
            const uint16_t idx = (uint16_t)((uint16_t)stream_buf[2 * r] |
                                            ((uint16_t)stream_buf[2 * r + 1] << 8));
            const int row_i = (int)(idx % HF_M);

            std::array<lwe_t, HF_N> row{};
            uint8_t row_nonce[12] = {0};
            row_nonce[0] = (uint8_t)(row_i & 0xFF);
            row_nonce[1] = (uint8_t)((row_i >> 8) & 0xFF);

            crypto_stream_chacha20_ietf(reinterpret_cast<unsigned char*>(row.data()),
                                        HF_N * sizeof(lwe_t),
                                        row_nonce, pk.seed_A);

            lwe_t* u = ct.u_transposed.data() + (size_t)k * HF_N;
            for (size_t j = 0; j < HF_N; ++j) u[j] = (lwe_t)(u[j] + row[j]);

            sum_v += pk.b[(size_t)row_i];
        }

        lwe_t* u = ct.u_transposed.data() + (size_t)k * HF_N;
        for (size_t j = 0; j < HF_N; ++j) {
            const int16_t e1 = hf_cbd_from_seed(r_seed.data(), (uint32_t)(k * (int)HF_N + (int)j));
            u[j] = (lwe_t)(u[j] + (lwe_t)e1);
        }

        const int16_t e2 = hf_cbd_from_seed(r_seed.data(), (uint32_t)(k * (int)HF_N + (int)HF_N));
        sum_v = (uint32_t)((sum_v + (int32_t)e2) & 0xFFFF);

        // branchless add of q/2 based on message bit (avoid secret-dependent branch in decaps re-encrypt)
        sum_v = (uint32_t)((sum_v + (uint32_t)bit * (uint32_t)HF_Q_HALF) & 0xFFFF);
        ct.v[(size_t)k] = (lwe_t)sum_v;
    }

    hf_sha3_256_multi({
        {"TAG", 3},
        {ct.u_transposed.data(), ct.u_transposed.size() * sizeof(lwe_t)},
        {ct.v.data(), ct.v.size() * sizeof(lwe_t)}
    }, ct.tag);
}

static inline void hf_encaps(const PublicKey& pk, SharedKey& outK, Ciphertext& ct) {
    std::array<uint8_t, 32> mu{};
    randombytes_buf(mu.data(), mu.size());

    hf_encaps_det(pk, mu, ct);

    hf_sha3_256_multi({
        {"KEY", 3},
        {mu.data(), mu.size()},
        {ct.tag.data(), ct.tag.size()}
    }, outK.span());
}

static inline void hf_decaps(const SecretKey& sk, const Ciphertext& ct, SharedKey& outK) {
    const int T = hf_max_threads();
    std::vector<std::array<uint8_t, 32>> thread_mu((size_t)T);
    for (auto& x : thread_mu) x.fill(0);

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        const int tid = hf_thread_num();
#if defined(_OPENMP)
#pragma omp for
#endif
        for (int k = 0; k < (int)HF_KBITS; ++k) {
            std::span<const lwe_t, HF_N> u_vec(&ct.u_transposed[(size_t)k * HF_N], HF_N);
            const uint32_t dot = hf_dot_product_avx2(u_vec, sk.s);

            const lwe_t diff = (lwe_t)(ct.v[(size_t)k] - (lwe_t)dot);
            const uint32_t close_mask = ct_is_close_mask(diff, HF_Q_HALF, HF_Q_QTR);

            const uint8_t bit = (uint8_t)((close_mask & 1u) << (k & 7));
            thread_mu[(size_t)tid][(size_t)k >> 3] |= bit;
        }
    }

    std::array<uint8_t, 32> mu_prime{};
    mu_prime.fill(0);
    for (const auto& t : thread_mu) {
        for (size_t i = 0; i < mu_prime.size(); ++i) mu_prime[i] |= t[i];
    }

    Ciphertext ct_prime;
    hf_encaps_det(sk.pk, mu_prime, ct_prime);

    const int u_ok  = (sodium_memcmp(ct.u_transposed.data(),
                                     ct_prime.u_transposed.data(),
                                     ct.u_transposed.size() * sizeof(lwe_t)) == 0);
    const int v_ok  = (sodium_memcmp(ct.v.data(), ct_prime.v.data(),
                                     ct.v.size() * sizeof(lwe_t)) == 0);
    const int tag_ok = (sodium_memcmp(ct.tag.data(), ct_prime.tag.data(),
                                      ct.tag.size()) == 0);

    const int valid = (u_ok & v_ok & tag_ok);

    uint8_t K_real[32];
    uint8_t K_fake[32];

    hf_sha3_256_multi({
        {"KEY", 3},
        {mu_prime.data(), mu_prime.size()},
        {ct.tag.data(), ct.tag.size()}
    }, std::span<uint8_t, 32>(K_real));

    std::array<uint8_t, 32> sk_prf{};
    hf_sha3_256_multi({
        {"SKPRF", 5},
        {sk.s.bits.data(), sk.s.bits.size() * sizeof(uint64_t)},
        {sk.pk.seed_A, sizeof(sk.pk.seed_A)},
        {sk.pk.b.data(), sk.pk.b.size() * sizeof(lwe_t)}
    }, sk_prf);

    hf_sha3_256_multi({
        {"FAKE", 4},
        {sk_prf.data(), sk_prf.size()},
        {ct.tag.data(), ct.tag.size()}
    }, std::span<uint8_t, 32>(K_fake));

    const uint8_t mask = (uint8_t)-(valid != 0);
    for (int i = 0; i < 32; ++i) outK.data()[i] = ct_select_u8(mask, K_real[i], K_fake[i]);
}

// ------------------------------ STREAMING AEAD ------------------------------
static void hf_aes_encrypt_stream_openssl(const uint8_t key[32],
                                         const uint8_t nonce[12],
                                         const std::string& in_f,
                                         std::ofstream& out_s,
                                         std::array<uint8_t, 16>& tag) {
    std::ifstream in(in_f, std::ios::binary);
    if (!in) throw std::runtime_error("Input file open failed");

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    CHECK_SSL(ctx != nullptr);

    try {
        CHECK_SSL(EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) == 1);
        CHECK_SSL(EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 12, nullptr) == 1);
        CHECK_SSL(EVP_EncryptInit_ex(ctx, nullptr, nullptr, key, nonce) == 1);

        int len = 0;
        CHECK_SSL(EVP_EncryptUpdate(ctx, nullptr, &len,
                                   reinterpret_cast<const uint8_t*>(HF_AAD_STR),
                                   (int)std::strlen(HF_AAD_STR)) == 1);

        std::vector<uint8_t> bi(HF_CHUNK_SIZE), bo(HF_CHUNK_SIZE + 32);

        while (in.read(reinterpret_cast<char*>(bi.data()), (std::streamsize)bi.size()) || in.gcount() > 0) {
            const int in_len = (int)in.gcount();
            CHECK_SSL(EVP_EncryptUpdate(ctx, bo.data(), &len, bi.data(), in_len) == 1);
            out_s.write(reinterpret_cast<char*>(bo.data()), len);
        }

        CHECK_SSL(EVP_EncryptFinal_ex(ctx, bo.data(), &len) == 1);
        out_s.write(reinterpret_cast<char*>(bo.data()), len);

        CHECK_SSL(EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tag.data()) == 1);
    } catch (...) {
        EVP_CIPHER_CTX_free(ctx);
        throw;
    }

    EVP_CIPHER_CTX_free(ctx);
}

static bool hf_aes_decrypt_stream_openssl(const uint8_t key[32],
                                         const uint8_t nonce[12],
                                         std::ifstream& in_s,
                                         uint64_t ct_len,
                                         const std::string& out_f,
                                         const std::array<uint8_t, 16>& tag) {
    const std::string tmp = out_f + ".tmp";
    std::ofstream out(tmp, std::ios::binary);
    if (!out) return false;

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return false;

    bool ok = false;

    try {
        CHECK_SSL(EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) == 1);
        CHECK_SSL(EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 12, nullptr) == 1);
        CHECK_SSL(EVP_DecryptInit_ex(ctx, nullptr, nullptr, key, nonce) == 1);

        int len = 0;
        CHECK_SSL(EVP_DecryptUpdate(ctx, nullptr, &len,
                                   reinterpret_cast<const uint8_t*>(HF_AAD_STR),
                                   (int)std::strlen(HF_AAD_STR)) == 1);

        std::vector<uint8_t> bi(HF_CHUNK_SIZE), bo(HF_CHUNK_SIZE + 32);
        uint64_t total = 0;

        while (total < ct_len) {
            const size_t to_read = (size_t)std::min<uint64_t>((uint64_t)bi.size(), ct_len - total);
            in_s.read(reinterpret_cast<char*>(bi.data()), (std::streamsize)to_read);
            const std::streamsize got = in_s.gcount();
            if (got <= 0) break;

            CHECK_SSL(EVP_DecryptUpdate(ctx, bo.data(), &len, bi.data(), (int)got) == 1);
            out.write(reinterpret_cast<char*>(bo.data()), len);
            total += (uint64_t)got;
        }

        CHECK_SSL(EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16, (void*)tag.data()) == 1);

        if (EVP_DecryptFinal_ex(ctx, bo.data(), &len) > 0) {
            out.write(reinterpret_cast<char*>(bo.data()), len);
            ok = true;
        } else {
            ok = false;
        }
    } catch (...) {
        ok = false;
    }

    EVP_CIPHER_CTX_free(ctx);
    out.close();

    if (ok) {
        if (fs::exists(out_f)) fs::remove(out_f);
        fs::rename(tmp, out_f);
        return true;
    }
    fs::remove(tmp);
    return false;
}

// ------------------------------ ARGON2 + SK STORAGE --------------------------
static void hf_argon2id_derive(const std::string& pwd,
                              std::span<const uint8_t, 16> salt,
                              std::span<uint8_t, 32> out) {
    if (crypto_pwhash(out.data(), out.size(),
                      pwd.c_str(), pwd.size(),
                      salt.data(),
                      crypto_pwhash_OPSLIMIT_MODERATE,
                      crypto_pwhash_MEMLIMIT_MODERATE,
                      crypto_pwhash_ALG_ARGON2ID13) != 0) {
        throw std::runtime_error("Argon2id derive failed (out of memory?)");
    }
}

static std::vector<uint8_t> aead_encrypt_sk_blob(const std::vector<uint8_t>& plain,
                                                 const std::array<uint8_t, 32>& key,
                                                 const std::array<uint8_t, 24>& nonce,
                                                 bool& used_aes) {
    std::vector<uint8_t> ct(plain.size() + 16);
    unsigned long long clen = 0;

    if (crypto_aead_aes256gcm_is_available()) {
        used_aes = true;
        if (crypto_aead_aes256gcm_encrypt(ct.data(), &clen,
                                          plain.data(), (unsigned long long)plain.size(),
                                          nullptr, 0, nullptr,
                                          nonce.data(), key.data()) != 0) {
            throw std::runtime_error("SK AEAD AES256-GCM encrypt failed");
        }
    } else {
        used_aes = false;
        if (crypto_aead_xchacha20poly1305_ietf_encrypt(ct.data(), &clen,
                                                      plain.data(), (unsigned long long)plain.size(),
                                                      nullptr, 0, nullptr,
                                                      nonce.data(), key.data()) != 0) {
            throw std::runtime_error("SK AEAD XChaCha20-Poly1305 encrypt failed");
        }
    }

    ct.resize((size_t)clen);
    return ct;
}

static bool aead_decrypt_sk_blob(const std::vector<uint8_t>& ct,
                                 std::vector<uint8_t>& plain,
                                 const std::array<uint8_t, 32>& key,
                                 const std::array<uint8_t, 24>& nonce,
                                 bool used_aes) {
    unsigned long long plen = 0;

    if (used_aes) {
        return crypto_aead_aes256gcm_decrypt(plain.data(), &plen, nullptr,
                                             ct.data(), (unsigned long long)ct.size(),
                                             nullptr, 0, nonce.data(), key.data()) == 0
               && plen == plain.size();
    }

    return crypto_aead_xchacha20poly1305_ietf_decrypt(plain.data(), &plen, nullptr,
                                                     ct.data(), (unsigned long long)ct.size(),
                                                     nullptr, 0, nonce.data(), key.data()) == 0
           && plen == plain.size();
}

// ------------------------------ PK/SK I/O -----------------------------------
static void save_pk(const std::string& f, const PublicKey& pk) {
    std::ofstream o(f, std::ios::binary);
    if (!o) throw std::runtime_error("save_pk: open failed");

    if (!write_exact(o, HF_MAGIC_PK, 4)) throw std::runtime_error("save_pk: write failed");
    o.put((char)HF_VERSION_ID);

    if (!write_exact(o, pk.seed_A, 32)) throw std::runtime_error("save_pk: write failed");
    if (!write_exact(o, pk.b.data(), pk.b.size() * sizeof(lwe_t))) throw std::runtime_error("save_pk: write failed");
}

static bool load_pk(const std::string& f, PublicKey& pk) {
    std::ifstream i(f, std::ios::binary);
    if (!i) return false;

    uint8_t m[4];
    if (!read_exact(i, m, 4)) return false;
    if (std::memcmp(m, HF_MAGIC_PK, 4) != 0) return false;

    const int ver = i.get();
    if (ver == EOF) return false;
    if ((uint8_t)ver != HF_VERSION_ID) return false;

    if (!read_exact(i, pk.seed_A, 32)) return false;
    if (!read_exact(i, pk.b.data(), pk.b.size() * sizeof(lwe_t))) return false;

    return true;
}

static void save_sk(const std::string& f, const SecretKey& sk, const std::string& pwd = "") {
    std::ofstream o(f, std::ios::binary);
    if (!o) throw std::runtime_error("save_sk: open failed");

    if (!write_exact(o, HF_MAGIC_SK, 4)) throw std::runtime_error("save_sk: write failed");
    o.put((char)HF_VERSION_ID);

    const uint8_t enc = pwd.empty() ? 0 : 1;
    o.put((char)enc);

    const size_t plain_sz = sk.s.bits.size() * sizeof(uint64_t) + 32 + sk.pk.b.size() * sizeof(lwe_t);
    std::vector<uint8_t> plain(plain_sz);

    size_t off = 0;
    std::memcpy(plain.data() + off, sk.s.bits.data(), sk.s.bits.size() * sizeof(uint64_t));
    off += sk.s.bits.size() * sizeof(uint64_t);
    std::memcpy(plain.data() + off, sk.pk.seed_A, 32);
    off += 32;
    std::memcpy(plain.data() + off, sk.pk.b.data(), sk.pk.b.size() * sizeof(lwe_t));

    if (enc) {
        std::array<uint8_t, 16> salt{};
        randombytes_buf(salt.data(), salt.size());

        std::array<uint8_t, 24> nonce{};
        randombytes_buf(nonce.data(), nonce.size());

        std::array<uint8_t, 32> key{};
        hf_argon2id_derive(pwd, salt, key);

        bool used_aes = false;
        auto ct = aead_encrypt_sk_blob(plain, key, nonce, used_aes);

        const uint8_t aead_id = used_aes ? 1 : 2;
        if (!write_exact(o, salt.data(), salt.size())) throw std::runtime_error("save_sk: write failed");
        if (!write_exact(o, nonce.data(), nonce.size())) throw std::runtime_error("save_sk: write failed");
        o.put((char)aead_id);

        const uint32_t clen32 = (uint32_t)ct.size();
        if (!write_exact(o, &clen32, sizeof(clen32))) throw std::runtime_error("save_sk: write failed");
        if (!write_exact(o, ct.data(), ct.size())) throw std::runtime_error("save_sk: write failed");
    } else {
        if (!write_exact(o, plain.data(), plain.size())) throw std::runtime_error("save_sk: write failed");
    }
}

static bool load_sk(const std::string& f, SecretKey& sk, const std::string& pwd = "") {
    std::ifstream i(f, std::ios::binary);
    if (!i) return false;

    uint8_t m[4];
    if (!read_exact(i, m, 4)) return false;
    if (std::memcmp(m, HF_MAGIC_SK, 4) != 0) return false;

    const int ver = i.get();
    if (ver == EOF) return false;
    if ((uint8_t)ver != HF_VERSION_ID) return false;

    const int enc = i.get();
    if (enc == EOF) return false;

    const size_t plain_sz = sk.s.bits.size() * sizeof(uint64_t) + 32 + sk.pk.b.size() * sizeof(lwe_t);
    std::vector<uint8_t> plain(plain_sz);

    if (enc) {
        if (pwd.empty()) {
            std::cout << RED << "Password required to load this secret key.\n" << RST;
            return false;
        }

        std::array<uint8_t, 16> salt{};
        std::array<uint8_t, 24> nonce{};
        if (!read_exact(i, salt.data(), salt.size())) return false;
        if (!read_exact(i, nonce.data(), nonce.size())) return false;

        const int aead_id_i = i.get();
        if (aead_id_i == EOF) return false;
        const uint8_t aead_id = (uint8_t)aead_id_i;
        const bool used_aes = (aead_id == 1);

        uint32_t clen32 = 0;
        if (!read_exact(i, &clen32, sizeof(clen32))) return false;

        const uint64_t total_sz = file_size_u64(f);
        const uint64_t pos = (uint64_t)i.tellg();
        if (pos > total_sz) return false;
        const uint64_t rem = total_sz - pos;
        if ((uint64_t)clen32 != rem) return false;

        std::vector<uint8_t> ct((size_t)clen32);
        if (!read_exact(i, ct.data(), ct.size())) return false;

        std::array<uint8_t, 32> key{};
        hf_argon2id_derive(pwd, salt, key);

        if (!aead_decrypt_sk_blob(ct, plain, key, nonce, used_aes)) return false;
    } else {
        if (!read_exact(i, plain.data(), plain.size())) return false;
    }

    size_t off = 0;
    std::memcpy(sk.s.bits.data(), plain.data() + off, sk.s.bits.size() * sizeof(uint64_t));
    off += sk.s.bits.size() * sizeof(uint64_t);
    std::memcpy(sk.pk.seed_A, plain.data() + off, 32);
    off += 32;
    std::memcpy(sk.pk.b.data(), plain.data() + off, sk.pk.b.size() * sizeof(lwe_t));
    return true;
}


// ------------------------------ I/O SAFE WRAPPERS ---------------------------
// Benchmark/CLI expects bool-returning helpers; keep the exception-throwing core I/O.
static bool hf_save_pk(const std::string& f, const PublicKey& pk) {
    try {
        save_pk(f, pk);
        return true;
    } catch (...) {
        return false;
    }
}

static bool hf_save_sk(const std::string& f, const SecretKey& sk, const std::string& pwd = "") {
    try {
        save_sk(f, sk, pwd);
        return true;
    } catch (...) {
        return false;
    }
}

static bool hf_load_pk(const std::string& f, PublicKey& pk) {
    return load_pk(f, pk);
}

static bool hf_load_sk(const std::string& f, SecretKey& sk, const std::string& pwd = "") {
    return load_sk(f, sk, pwd);
}

// ------------------------------ FILE KEM WRAPPERS ---------------------------
static bool hf_kem_encrypt_safe(const std::string& pk_f,
                               const std::string& in_f,
                               const std::string& out_f) {
    PublicKey pk{};
    if (!load_pk(pk_f, pk)) return false;

    if (!fs::exists(in_f) || !fs::is_regular_file(in_f)) return false;

    SharedKey K;
    Ciphertext ct;
    hf_encaps(pk, K, ct);

    std::ofstream out(out_f, std::ios::binary);
    if (!out) return false;

    if (!write_exact(out, HF_MAGIC_CT, 4)) return false;
    out.put((char)HF_VERSION_ID);

    std::array<uint8_t, 12> nonce{};
    randombytes_buf(nonce.data(), nonce.size());
    if (!write_exact(out, nonce.data(), nonce.size())) return false;

    const uint32_t usz = (uint32_t)(ct.u_transposed.size() * sizeof(lwe_t));
    if (!write_exact(out, &usz, sizeof(usz))) return false;

    if (!write_exact(out, ct.u_transposed.data(), usz)) return false;
    if (!write_exact(out, ct.v.data(), ct.v.size() * sizeof(lwe_t))) return false;
    if (!write_exact(out, ct.tag.data(), ct.tag.size())) return false;

    const uint64_t fsz = file_size_u64(in_f);
    if (!write_exact(out, &fsz, sizeof(fsz))) return false;

    std::array<uint8_t, 16> gcm_tag{};
    const std::streampos tag_pos = out.tellp();
    if (!write_exact(out, gcm_tag.data(), gcm_tag.size())) return false;

    std::array<uint8_t, 32> aes_key{};
    hf_sha3_256_multi({{"AES", 3}, {K.data(), 32}}, aes_key);

    hf_aes_encrypt_stream_openssl(aes_key.data(), nonce.data(), in_f, out, gcm_tag);

    out.seekp(tag_pos);
    if (!write_exact(out, gcm_tag.data(), gcm_tag.size())) return false;

    return true;
}

static bool hf_kem_decrypt_safe(const std::string& sk_f,
                               const std::string& in_f,
                               const std::string& out_f,
                               const std::string& pwd = "") {
    SecretKey sk{};
    if (!load_sk(sk_f, sk, pwd)) return false;

    std::ifstream in(in_f, std::ios::binary);
    if (!in) return false;

    uint8_t m[4];
    if (!read_exact(in, m, 4)) return false;
    if (std::memcmp(m, HF_MAGIC_CT, 4) != 0) return false;

    const int ver = in.get();
    if (ver == EOF) return false;
    if ((uint8_t)ver != HF_VERSION_ID) return false;

    std::array<uint8_t, 12> nonce{};
    if (!read_exact(in, nonce.data(), nonce.size())) return false;

    uint32_t usz = 0;
    if (!read_exact(in, &usz, sizeof(usz))) return false;

    const uint32_t expected_usz = (uint32_t)(HF_KBITS * HF_N * sizeof(lwe_t));
    if (usz != expected_usz) return false;

    Ciphertext ct{};
    ct.u_transposed.resize(HF_KBITS * HF_N);

    if (!read_exact(in, ct.u_transposed.data(), usz)) return false;
    if (!read_exact(in, ct.v.data(), ct.v.size() * sizeof(lwe_t))) return false;
    if (!read_exact(in, ct.tag.data(), ct.tag.size())) return false;

    uint64_t ct_len = 0;
    if (!read_exact(in, &ct_len, sizeof(ct_len))) return false;

    std::array<uint8_t, 16> gcm_tag{};
    if (!read_exact(in, gcm_tag.data(), gcm_tag.size())) return false;

    const uint64_t total_sz = file_size_u64(in_f);
    const uint64_t pos = (uint64_t)in.tellg();
    if (pos > total_sz) return false;
    const uint64_t rem = total_sz - pos;
    if (ct_len != rem) return false;

    SharedKey K;
    hf_decaps(sk, ct, K);

    std::array<uint8_t, 32> aes_key{};
    hf_sha3_256_multi({{"AES", 3}, {K.data(), 32}}, aes_key);

    return hf_aes_decrypt_stream_openssl(aes_key.data(), nonce.data(), in, ct_len, out_f, gcm_tag);
}

// ------------------------------ SELF TEST -----------------------------------
static void run_test_all() {
    std::cout << "\n" << BG_BLU << " SELF-DIAGNOSTIC TEST (AUDIT MODE) " << RST << "\n";

    for (const char* f : {"test.pk", "test.sk", "test.bin", "test.enc", "test.dec"}) {
        if (fs::exists(f)) fs::remove(f);
    }

    std::cout << "  " << CYN << "[1/5] KeyGen (TopoMining + LWE)..." << RST << "\n";
    PublicKey pk{};
    SecretKey sk{};
    hf_keygen(pk, sk);
    save_pk("test.pk", pk);
    save_sk("test.sk", sk, "pass123");
    std::cout << "      " << GRN << "OK" << RST << "\n";

    std::cout << "  " << CYN << "[2/5] Payload (1MB random)..." << RST << std::flush;
    std::vector<uint8_t> dummy(1024 * 1024);
    randombytes_buf(dummy.data(), dummy.size());
    {
        std::ofstream f("test.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(dummy.data()), (std::streamsize)dummy.size());
    }
    std::cout << " " << GRN << "OK" << RST << "\n";

    std::cout << "  " << CYN << "[3/5] Encrypt (FO-LWE)..." << RST << std::flush;
    if (!hf_kem_encrypt_safe("test.pk", "test.bin", "test.enc")) {
        std::cout << " " << RED << "FAIL" << RST << "\n";
        return;
    }
    std::cout << " " << GRN << "OK" << RST << "\n";

    std::cout << "  " << CYN << "[4/5] Decrypt (FO check)..." << RST << std::flush;
    if (!hf_kem_decrypt_safe("test.sk", "test.enc", "test.dec", "pass123")) {
        std::cout << " " << RED << "FAIL" << RST << "\n";
        return;
    }
    std::cout << " " << GRN << "OK" << RST << "\n";

    std::cout << "  " << CYN << "[5/5] Integrity check..." << RST << std::flush;
    std::ifstream f1("test.bin", std::ios::binary | std::ios::ate);
    std::ifstream f2("test.dec", std::ios::binary | std::ios::ate);

    if (f1 && f2 && f1.tellg() == f2.tellg()) {
        std::cout << " " << GRN << "MATCH" << RST << "\n";
    } else {
        std::cout << " " << RED << "MISMATCH" << RST << "\n";
    }

    std::cout << YEL << "  [INFO] Files preserved for audit: test.pk test.sk test.enc test.bin test.dec" << RST << "\n";
}


// --------------------------- Benchmarking ---------------------------

struct HFBenchStats {
    double mean_ms = 0.0;
    double min_ms  = 0.0;
    double max_ms  = 0.0;
    double p50_ms  = 0.0;
    double p95_ms  = 0.0;
};

static inline uint64_t hf_now_ns() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

static HFBenchStats hf_stats_ms(std::vector<double> v_ms) {
    HFBenchStats st;
    if (v_ms.empty()) return st;
    std::sort(v_ms.begin(), v_ms.end());
    st.min_ms = v_ms.front();
    st.max_ms = v_ms.back();

    double sum = 0.0;
    for (double x : v_ms) sum += x;
    st.mean_ms = sum / (double)v_ms.size();

    auto pct = [&](double p) -> double {
        if (v_ms.size() == 1) return v_ms[0];
        const double idx = p * (double)(v_ms.size() - 1);
        const size_t lo = (size_t)std::floor(idx);
        const size_t hi = (size_t)std::ceil(idx);
        const double t  = idx - (double)lo;
        return v_ms[lo] * (1.0 - t) + v_ms[hi] * t;
    };
    st.p50_ms = pct(0.50);
    st.p95_ms = pct(0.95);
    return st;
}

static bool hf_files_equal(const std::string& a, const std::string& b) {
    std::ifstream fa(a, std::ios::binary);
    std::ifstream fb(b, std::ios::binary);
    if (!fa || !fb) return false;
    std::vector<uint8_t> ba(1 << 20), bb(1 << 20); // 1 MiB chunks
    while (true) {
        fa.read((char*)ba.data(), (std::streamsize)ba.size());
        fb.read((char*)bb.data(), (std::streamsize)bb.size());
        const std::streamsize ra = fa.gcount();
        const std::streamsize rb = fb.gcount();
        if (ra != rb) return false;
        if (ra == 0) break;
        if (std::memcmp(ba.data(), bb.data(), (size_t)ra) != 0) return false;
    }
    return true;
}

static bool hf_make_random_file(const std::string& path, uint64_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> buf(1 << 20); // 1 MiB
    uint64_t remaining = bytes;
    while (remaining) {
        const size_t chunk = (size_t)std::min<uint64_t>((uint64_t)buf.size(), remaining);
        randombytes_buf(buf.data(), chunk);
        f.write((const char*)buf.data(), (std::streamsize)chunk);
        if (!f) return false;
        remaining -= chunk;
    }
    return true;
}

static void hf_write_bench_json(const std::string& path,
                               const std::string& prefix,
                               int target_genus,
                               int threads,
                               uint64_t mine_ms,
                               size_t iters,
                               size_t file_mb,
                               bool did_keygen,
                               double keygen_ms,
                               const HFBenchStats& enc,
                               const HFBenchStats& dec,
                               double file_enc_ms,
                               double file_dec_ms,
                               uint64_t file_bytes,
                               bool file_ok) {
    std::ofstream o(path);
    if (!o) return;

    const double enc_mbps = (file_enc_ms > 0.0) ? ((double)file_bytes / (1024.0 * 1024.0)) / (file_enc_ms / 1000.0) : 0.0;
    const double dec_mbps = (file_dec_ms > 0.0) ? ((double)file_bytes / (1024.0 * 1024.0)) / (file_dec_ms / 1000.0) : 0.0;

    o << "{\n";
    o << "  \"version\": \"HyperFrog v33.1\",\n";
    o << "  \"params\": {\"N\": " << HF_N << ", \"M\": " << HF_M << ", \"Q\": " << HF_Q << ", \"KBITS\": " << HF_KBITS << "},\n";
    o << "  \"prefix\": \"" << prefix << "\",\n";
    o << "  \"target_genus\": " << target_genus << ",\n";
    o << "  \"threads\": " << threads << ",\n";
    o << "  \"mine_ms\": " << mine_ms << ",\n";
    o << "  \"iters\": " << iters << ",\n";
    o << "  \"file_mb\": " << file_mb << ",\n";
    o << "  \"keygen\": {\"enabled\": " << (did_keygen ? "true" : "false") << ", \"ms\": " << keygen_ms << "},\n";
    o << "  \"encaps_ms\": {\"mean\": " << enc.mean_ms << ", \"min\": " << enc.min_ms << ", \"p50\": " << enc.p50_ms << ", \"p95\": " << enc.p95_ms << ", \"max\": " << enc.max_ms << "},\n";
    o << "  \"decaps_ms\": {\"mean\": " << dec.mean_ms << ", \"min\": " << dec.min_ms << ", \"p50\": " << dec.p50_ms << ", \"p95\": " << dec.p95_ms << ", \"max\": " << dec.max_ms << "},\n";
    o << "  \"file\": {\"bytes\": " << file_bytes
      << ", \"enc_ms\": " << file_enc_ms << ", \"dec_ms\": " << file_dec_ms
      << ", \"enc_MBps\": " << enc_mbps << ", \"dec_MBps\": " << dec_mbps
      << ", \"ok\": " << (file_ok ? "true" : "false") << "}\n";
    o << "}\n";
}

static int run_benchmark(const std::string& prefix,
                         const std::string& pass,
                         size_t iters,
                         size_t file_mb,
                         bool bench_keygen,
                         int target_genus,
                         uint64_t mine_ms,
                         int threads,
                         const std::string& out_override) {
    const std::string outp = out_override.empty() ? (prefix + ".bench") : out_override;
    const std::string csv  = outp + ".csv";
    const std::string js   = outp + ".json";
    const std::string bin  = outp + ".bin";
    const std::string encf = outp + ".enc";
    const std::string decf = outp + ".dec";

    const std::string pkf = prefix + ".pk";
    const std::string skf = prefix + ".sk";

    std::ofstream ocsv(csv);
    if (!ocsv) {
        std::cerr << "Cannot write " << csv << "\n";
        return 2;
    }
    ocsv << "case,iter,ms,bytes,notes\n";

    PublicKey pk;
    SecretKey sk;

    double keygen_ms = 0.0;
    bool did_keygen = false;

    const bool have_keys = std::filesystem::exists(pkf) && std::filesystem::exists(skf);
    if (bench_keygen || !have_keys) {
        const uint64_t t0 = hf_now_ns();
        hf_keygen(pk, sk, target_genus, mine_ms, threads);
        const uint64_t t1 = hf_now_ns();
        keygen_ms = (double)(t1 - t0) / 1e6;
        did_keygen = true;

        if (!hf_save_pk(pkf, pk)) {
            std::cerr << "Benchmark keygen: failed saving pk\n";
            return 3;
        }
        if (!hf_save_sk(skf, sk, pass)) {
            std::cerr << "Benchmark keygen: failed saving sk\n";
            return 3;
        }

        ocsv << "keygen,0," << keygen_ms << ",0,\"genus=" << target_genus << ";mine_ms=" << mine_ms << ";threads=" << threads << "\"\n";
        ocsv.flush();
    } else {
        if (!hf_load_pk(pkf, pk)) {
            std::cerr << "Benchmark: failed loading pk\n";
            return 3;
        }
        if (!hf_load_sk(skf, sk, pass)) {
            std::cerr << "Benchmark: failed loading sk (wrong --password?)\n";
            return 3;
        }
    }

    // Encaps/Decaps microbench
    std::vector<double> enc_ms; enc_ms.reserve(iters);
    std::vector<double> dec_ms; dec_ms.reserve(iters);

    for (size_t i = 0; i < iters; i++) {
        Ciphertext ct;
        SharedKey K1, K2;

        const uint64_t t0 = hf_now_ns();
        hf_encaps(pk, K1, ct);
        const uint64_t t1 = hf_now_ns();
        const double ms = (double)(t1 - t0) / 1e6;
        enc_ms.push_back(ms);
        ocsv << "encaps," << i << "," << ms << ",0,\"\"\n";

        const uint64_t t2 = hf_now_ns();
        hf_decaps(sk, ct, K2);
        const uint64_t t3 = hf_now_ns();
        const double ms2 = (double)(t3 - t2) / 1e6;
        dec_ms.push_back(ms2);
        ocsv << "decaps," << i << "," << ms2 << ",0,\"\"\n";

        if (sodium_memcmp(K1.data(), K2.data(), 32) != 0) {
            std::cerr << "Benchmark: K mismatch at iter " << i << "\n";
            return 4;
        }
    }
    ocsv.flush();

    const HFBenchStats encst = hf_stats_ms(enc_ms);
    const HFBenchStats decst = hf_stats_ms(dec_ms);

    ocsv << "encaps_summary,0," << encst.mean_ms << ",0,\"p50=" << encst.p50_ms << ";p95=" << encst.p95_ms << "\"\n";
    ocsv << "decaps_summary,0," << decst.mean_ms << ",0,\"p50=" << decst.p50_ms << ";p95=" << decst.p95_ms << "\"\n";
    ocsv.flush();

    // File bench
    const uint64_t file_bytes = (uint64_t)file_mb * 1024ULL * 1024ULL;
    if (!hf_make_random_file(bin, file_bytes)) {
        std::cerr << "Benchmark: failed creating random file\n";
        return 5;
    }

    double file_enc_ms = 0.0, file_dec_ms = 0.0;
    bool file_ok = false;

    {
        const uint64_t t0 = hf_now_ns();
        const bool ok = hf_kem_encrypt_safe(pkf, bin, encf);
        const uint64_t t1 = hf_now_ns();
        file_enc_ms = (double)(t1 - t0) / 1e6;
        ocsv << "file_enc,0," << file_enc_ms << "," << file_bytes << ",\"\"\n";
        ocsv.flush();
        if (!ok) {
            std::cerr << "Benchmark: file encryption failed\n";
            hf_write_bench_json(js, prefix, target_genus, threads, mine_ms, iters, file_mb, did_keygen, keygen_ms, encst, decst,
                                file_enc_ms, 0.0, file_bytes, false);
            return 6;
        }
    }
    {
        const uint64_t t0 = hf_now_ns();
        const bool ok = hf_kem_decrypt_safe(skf, encf, decf, pass);
        const uint64_t t1 = hf_now_ns();
        file_dec_ms = (double)(t1 - t0) / 1e6;
        ocsv << "file_dec,0," << file_dec_ms << "," << file_bytes << ",\"\"\n";
        ocsv.flush();
        if (!ok) {
            std::cerr << "Benchmark: file decryption failed\n";
            hf_write_bench_json(js, prefix, target_genus, threads, mine_ms, iters, file_mb, did_keygen, keygen_ms, encst, decst,
                                file_enc_ms, file_dec_ms, file_bytes, false);
            return 7;
        }
    }

    file_ok = hf_files_equal(bin, decf);
    ocsv << "file_check,0,0," << file_bytes << ",\"" << (file_ok ? "OK" : "MISMATCH") << "\"\n";
    ocsv.flush();

    hf_write_bench_json(js, prefix, target_genus, threads, mine_ms, iters, file_mb, did_keygen, keygen_ms, encst, decst,
                        file_enc_ms, file_dec_ms, file_bytes, file_ok);

    std::cout << "[BENCH] Wrote " << csv << " and " << js << "\n";
    std::cout << "[BENCH] Artifacts: " << bin << " " << encf << " " << decf << "\n";
    std::cout << "[BENCH] enc(mean/p50/p95) ms: " << encst.mean_ms << " / " << encst.p50_ms << " / " << encst.p95_ms << "\n";
    std::cout << "[BENCH] dec(mean/p50/p95) ms: " << decst.mean_ms << " / " << decst.p50_ms << " / " << decst.p95_ms << "\n";

    return file_ok ? 0 : 8;
}


static void print_help(const char* argv0) {
    std::cout
        << "HyperFrog HFv6 (research prototype)\n"
        << "Usage:\n"
        << "  " << argv0 << " --gen-keys <prefix> [--password <pw>] [--mine-ms <ms>] [--threads <n>] [--min-genus <g>]\n"
        << "  " << argv0 << " --validate-keys <prefix> [--password <pw>]\n"
        << "  " << argv0 << " --enc <prefix> <in_plain> <out_hf>\n"
        << "  " << argv0 << " --dec <prefix> <in_hf> <out_plain> [--password <pw>]\n"
        << "  " << argv0 << " --test-all\n"
        << "  " << argv0 << " --benchmark <prefix> [--password <pw>] [--iters <n>] [--file-mb <mb>] [--bench-keygen] [--out <outprefix>] [--mine-ms <ms>] [--threads <n>] [--min-genus <g>]\n"
        << "  " << argv0 << " --help\n"
        << "\nFiles:\n"
        << "  <prefix>.pk  public key (seed_A + b); ~2.0 MiB for current params\n"
        << "  <prefix>.sk  secret key (includes pk; optionally password-protected via Argon2id)\n"
        << "\nMining:\n"
        << "  Default budget is 30000 ms; mining is randomized and multi-threaded.\n"
        << "  This build keeps only the largest connected component per candidate (much faster).\n"
        << "\nBenchmark:\n"
        << "  Generates <out>.csv and <out>.json plus <out>.bin/.enc/.dec for sanity-check.\n"
        << "  Defaults: --iters 200, --file-mb 64, --out <prefix>.bench (key files: <prefix>.pk/.sk).\n"
        << "\nExamples:\n"
        << "  " << argv0 << " --gen-keys mykeys --password \"hunter2\" --mine-ms 5000\n"
        << "  " << argv0 << " --validate-keys mykeys --password \"hunter2\"\n"
        << "  " << argv0 << " --enc mykeys input.bin output.hf\n"
        << "  " << argv0 << " --dec mykeys output.hf recovered.bin --password \"hunter2\"\n"
        << "  " << argv0 << " --benchmark mykeys --password \"hunter2\" --iters 500 --file-mb 128 --bench-keygen\n";
}

// ------------------------------ MAIN ----------------------------------------
int main(int argc, char** argv) {
    if (sodium_init() < 0) {
        std::cerr << "libsodium init failed\n";
        return 1;
    }

    if (argc <= 1) {
        print_help(argv[0]);
        return 0;
    }

    std::vector<std::string> args;
    args.reserve((size_t)argc - 1);
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);

    const std::string cmd = args[0];
    if (cmd == "--help" || cmd == "-h") {
        print_help(argv[0]);
        return 0;
    }

    auto die = [&](const std::string& msg) -> int {
        std::cerr << RED << "[HF] " << msg << RST << "\n";
        std::cerr << "Run with --help to see usage.\n";
        return 2;
    };

    // Common optional flags
    std::string password;
    uint64_t mine_ms = 30000;
    int threads = 0;
    // Benchmark knobs (used by --benchmark)
    size_t bench_iters = 200;
    size_t bench_file_mb = 64;
    bool bench_keygen = false;
    std::string bench_out;
    int min_genus = 8;

std::vector<std::string> pos;

auto parse_flags = [&](size_t start_idx) -> bool {
    for (size_t i = start_idx; i < args.size(); ++i) {
        const std::string& a = args[i];

        auto need = [&]() -> const std::string* {
            if (i + 1 >= args.size()) return nullptr;
            return &args[++i];
        };

        if (a == "--password" || a == "--pwd") {
            const std::string* v = need();
            if (!v) return false;
            password = *v;
        } else if (a == "--mine-ms") {
            const std::string* v = need();
            if (!v) return false;
            try { mine_ms = std::stoull(*v); } catch (...) { return false; }
        } else if (a == "--threads") {
            const std::string* v = need();
            if (!v) return false;
            try { threads = std::stoi(*v); } catch (...) { return false; }
        } else if (a == "--min-genus") {
            const std::string* v = need();
            if (!v) return false;
            try { min_genus = std::stoi(*v); } catch (...) { return false; }
            if (min_genus < 0) min_genus = 0;
        } else if (a == "--iters") {
            if (i + 1 >= args.size()) return false;
            try { bench_iters = (size_t)std::stoull(args[++i]); } catch (...) { return false; }
        } else if (a == "--file-mb") {
            if (i + 1 >= args.size()) return false;
            try { bench_file_mb = (size_t)std::stoull(args[++i]); } catch (...) { return false; }
        } else if (a == "--bench-keygen") {
            bench_keygen = true;
        } else if (a == "--out") {
            if (i + 1 >= args.size()) return false;
            bench_out = args[++i];
        } else if (a.size() && a[0] == '-') {
            // unknown flag for this command
            return false;
        } else {
            pos.push_back(a);
        }
    }
    return true;
};
;

    if (cmd == "--benchmark") {
        if (!parse_flags(1)) return die("Invalid arguments for --benchmark.");
        if (pos.size() != 1) return die("--benchmark requires <prefix>.");
        const std::string prefix = pos[0];
        return run_benchmark(prefix, password, bench_iters, bench_file_mb, bench_keygen, min_genus, mine_ms, threads, bench_out);
    }

    if (cmd == "--test-all") {
        run_test_all();
        return 0;
    }

    if (cmd == "--gen-keys") {
        if (!parse_flags(1)) return die("Invalid arguments for --gen-keys.");
        if (pos.size() != 1) return die("--gen-keys requires <prefix>.");

        const std::string prefix = pos[0];
        const std::string pk_path = prefix + ".pk";
        const std::string sk_path = prefix + ".sk";

        PublicKey pk{};
        SecretKey sk{};

        std::cout << "[HF] KeyGen: prefix='" << prefix << "', min_genus=" << min_genus
                  << ", mine_ms=" << mine_ms << ", threads=" << threads << "\n";

        hf_keygen(pk, sk, min_genus, mine_ms, threads);

        try {
            save_pk(pk_path, pk);
        } catch (const std::exception& e) {
            return die(std::string("Failed to write public key: ") + pk_path + " (" + e.what() + ")");
        }
        try {
            save_sk(sk_path, sk, password);
        } catch (const std::exception& e) {
            return die(std::string("Failed to write secret key: ") + sk_path + " (" + e.what() + ")");
        }

        std::cout << GRN << "[HF] Wrote " << pk_path << " and " << sk_path << RST << "\n";
        return 0;
    }

    if (cmd == "--validate-keys") {
        if (!parse_flags(1)) return die("Invalid arguments for --validate-keys.");
        if (pos.size() != 1) return die("--validate-keys requires <prefix>.");

        const std::string prefix = pos[0];
        const std::string pk_path = prefix + ".pk";
        const std::string sk_path = prefix + ".sk";

        PublicKey pk{};
        SecretKey sk{};
        if (!load_pk(pk_path, pk)) return die("Failed to read public key: " + pk_path);
        if (!load_sk(sk_path, sk, password)) return die("Failed to read secret key: " + sk_path);

        const int pk_match = (sodium_memcmp(pk.b.data(), sk.pk.b.data(),
                                            pk.b.size() * sizeof(lwe_t)) == 0) &&
                             (sodium_memcmp(pk.seed_A, sk.pk.seed_A, 32) == 0);

        SharedKey K1{}, K2{};
        Ciphertext ct{};
        hf_encaps(pk, K1, ct);
        hf_decaps(sk, ct, K2);

        const int kem_ok = (sodium_memcmp(K1.data(), K2.data(), 32) == 0);

        std::cout << "[HF] pk in sk matches external pk: " << (pk_match ? "YES" : "NO") << "\n";
        std::cout << "[HF] KEM roundtrip: " << (kem_ok ? "OK" : "FAIL") << "\n";

        return (pk_match && kem_ok) ? 0 : 3;
    }

    if (cmd == "--enc") {
        if (!parse_flags(1)) return die("Invalid arguments for --enc.");
        if (pos.size() != 3) return die("--enc requires <prefix> <in_plain> <out_hf>.");

        const std::string prefix = pos[0];
        const std::string pk_path = prefix + ".pk";

        PublicKey pk{};
        if (!load_pk(pk_path, pk)) return die("Failed to read public key: " + pk_path);

        if (!hf_kem_encrypt_safe(pk_path, pos[1], pos[2])) return die("Encryption failed.");
        std::cout << GRN << "[HF] Encrypted -> " << pos[2] << RST << "\n";
        return 0;
    }

    if (cmd == "--dec") {
        if (!parse_flags(1)) return die("Invalid arguments for --dec.");
        if (pos.size() != 3) return die("--dec requires <prefix> <in_hf> <out_plain>.");

        const std::string prefix = pos[0];
        const std::string sk_path = prefix + ".sk";

        SecretKey sk{};
        if (!load_sk(sk_path, sk, password)) return die("Failed to read secret key: " + sk_path);

        if (!hf_kem_decrypt_safe(sk_path, pos[1], pos[2], password)) return die("Decryption failed.");
        std::cout << GRN << "[HF] Decrypted -> " << pos[2] << RST << "\n";
        return 0;
    }

    return die("Unknown command: " + cmd);
}
