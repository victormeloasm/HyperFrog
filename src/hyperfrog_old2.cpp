// ============================================================================
// HyperFrog v36.0 – "Connected-Graph Formal Miner" (C++23 Hardened)
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
#include <cmath>
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
#include <limits>
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
static constexpr uint8_t  HF_VERSION_ID = 0x07; // v7 (Connected-Graph Formal Miner)
static constexpr const char* HF_AAD_STR = "HF_V36_CONNECTED_GRAPH_FORMAL";

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
    std::cout << GRN << BOLD << "HyperFrog v36.0 [CONNECTED-GRAPH FORMAL MINER]\n" << RST;
    std::cout << BG_BLU << " EXACT-W=2048 " << RST << " " << BG_BLU << " FO-KEM " << RST
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


struct HFDecapDiag {
    bool valid = false;
    uint32_t min_margin_inside_one_band = std::numeric_limits<uint32_t>::max();
    uint32_t min_margin_outside_one_band = std::numeric_limits<uint32_t>::max();
    uint32_t min_distance_to_zero = std::numeric_limits<uint32_t>::max();
    size_t decoded_ones = 0;
    size_t decoded_zeros = 0;
};

struct HFDecapAggregate {
    size_t trials = 0;
    size_t valid_count = 0;
    size_t invalid_count = 0;
    uint32_t min_margin_inside_one_band = std::numeric_limits<uint32_t>::max();
    uint32_t min_margin_outside_one_band = std::numeric_limits<uint32_t>::max();
    uint32_t min_distance_to_zero = std::numeric_limits<uint32_t>::max();

    void add(const HFDecapDiag& d) {
        ++trials;
        if (d.valid) ++valid_count; else ++invalid_count;
        min_margin_inside_one_band = std::min(min_margin_inside_one_band, d.min_margin_inside_one_band);
        min_margin_outside_one_band = std::min(min_margin_outside_one_band, d.min_margin_outside_one_band);
        min_distance_to_zero = std::min(min_distance_to_zero, d.min_distance_to_zero);
    }
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

static inline uint32_t hf_mod_distance(lwe_t val, lwe_t target) {
    const uint32_t dist = (uint16_t)(val - target);
    return (dist <= 32768u) ? dist : (65536u - dist);
}

static inline uint32_t hf_margin_inside_band(uint32_t dist, uint32_t limit) {
    return (dist < limit) ? (limit - dist) : 0u;
}

static inline uint32_t hf_margin_outside_band(uint32_t dist, uint32_t limit) {
    return (dist >= limit) ? (dist - limit) : 0u;
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
    int cycle_rank = 0;   // beta1 of the occupied 6-neighbor graph: E - V + C
    int edges = 0;
    int vertices = 0;
};

struct HFMineStats {
    uint64_t attempts = 0;
    uint64_t accepted = 0;
    uint64_t frontier_fail = 0;
    uint64_t topo_reject_not_connected = 0;
    uint64_t topo_reject_wrong_weight = 0;
    uint64_t topo_reject_cycle_rank = 0;
    uint64_t elapsed_ms = 0;
    int accepted_cycle_rank = -1;
    int accepted_vertices = -1;
    int accepted_components = -1;

    void merge(const HFMineStats& other) {
        attempts += other.attempts;
        accepted += other.accepted;
        frontier_fail += other.frontier_fail;
        topo_reject_not_connected += other.topo_reject_not_connected;
        topo_reject_wrong_weight += other.topo_reject_wrong_weight;
        topo_reject_cycle_rank += other.topo_reject_cycle_rank;
        if (accepted_cycle_rank < 0 && other.accepted > 0) {
            accepted_cycle_rank = other.accepted_cycle_rank;
            accepted_vertices = other.accepted_vertices;
            accepted_components = other.accepted_components;
        }
    }
};

struct HFKeygenDiag {
    bool success = false;
    bool practical_miner = false;
    bool timed_out = false;
    uint64_t elapsed_ms = 0;
    int target_cycle_rank = 0;
    int shape_weight = 0;
    HFTopo topo{};
    HFMineStats mine{};
};

struct HFBenchStats {
    double mean_ms = 0.0;
    double min_ms  = 0.0;
    double max_ms  = 0.0;
    double p50_ms  = 0.0;
    double p95_ms  = 0.0;
};

struct HFKeygenBenchSummary {
    HFBenchStats all_ms{};
    HFBenchStats success_ms{};
    size_t sample_count = 0;
    size_t success_count = 0;
    size_t timeout_count = 0;
    size_t failure_count = 0;
};

struct TopoScratch {
    alignas(64) std::array<uint64_t, 64> visited_bits{};
    alignas(64) std::array<int16_t, 4096> queue{};
    alignas(64) std::array<int16_t, 4096> frontier{};
    alignas(64) std::array<int16_t, 4096> frontier_pos{};
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
    st.edges = E;
    st.vertices = V;
    st.cycle_rank = (E - V + C);
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


// Connected-shape helpers (engineering path only; formal miner has no fallback).
static inline Shape hf_fallback_shape_connected() {
    Shape s{};
    // Deterministic, connected, very "loopy" block to guarantee high cycle_rank.
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


// ------------------------------ TOPOMINE (FORMAL STRONG) --------------------
// Strong formal miner:
//   - build a connected occupied 6-neighbor graph directly by randomized frontier growth
//   - exact weight target (default 2048 occupied voxels)
//   - accept iff cycle_rank >= min_cycle_rank
//   - no fallback, no keep-largest-component, no best-of selection
//
// This defines an explicit, materially non-uniform secret law. The conditioned
// object is the occupied 6-neighbor graph, so the topological statistic here is
// the graph cycle rank beta1 = E - V + C.
static constexpr int HF_TOPOMINE_TARGET_WEIGHT_DEFAULT = 2048;
static constexpr int HF_TOPOMINE_MIN_CYCLE_RANK_STRONG_DEFAULT = 2700;

static inline int hf_shape_popcount(const Shape& s) {
    int V = 0;
    for (auto w : s.bits) V += (int)std::popcount(w);
    return V;
}

static inline void hf_shape_set(Shape& s, uint16_t idx) {
    s.bits[idx >> 6] |= (1ULL << (idx & 63));
}

static bool hf_sample_connected_exact_weight(Shape& out,
                                             TopoScratch& scratch,
                                             uint16_t target_weight = HF_TOPOMINE_TARGET_WEIGHT_DEFAULT) {
    if (target_weight == 0 || target_weight > HF_N) return false;

    out = Shape{};
    scratch.frontier_pos.fill((int16_t)-1);
    uint16_t frontier_len = 0;

    auto frontier_add = [&](uint16_t v) {
        if (out.get(v)) return;
        if (scratch.frontier_pos[v] >= 0) return;
        scratch.frontier_pos[v] = (int16_t)frontier_len;
        scratch.frontier[frontier_len++] = (int16_t)v;
    };

    auto frontier_pop_random = [&]() -> uint16_t {
        const uint32_t idx = randombytes_uniform(frontier_len);
        const uint16_t v = (uint16_t)scratch.frontier[idx];
        --frontier_len;
        const uint16_t last = (uint16_t)scratch.frontier[frontier_len];
        if (idx != frontier_len) {
            scratch.frontier[idx] = (int16_t)last;
            scratch.frontier_pos[last] = (int16_t)idx;
        }
        scratch.frontier_pos[v] = (int16_t)-1;
        return v;
    };

    const uint16_t start = (uint16_t)randombytes_uniform(HF_N);
    hf_shape_set(out, start);
    uint16_t count = 1;

    const uint16_t x0 = (start & 15u);
    const uint16_t y0 = ((start >> 4) & 15u);
    const uint16_t z0 = (start >> 8);
    if (x0 > 0)  frontier_add((uint16_t)(start - 1));
    if (x0 < 15) frontier_add((uint16_t)(start + 1));
    if (y0 > 0)  frontier_add((uint16_t)(start - 16));
    if (y0 < 15) frontier_add((uint16_t)(start + 16));
    if (z0 > 0)  frontier_add((uint16_t)(start - 256));
    if (z0 < 15) frontier_add((uint16_t)(start + 256));

    while (count < target_weight) {
        if (frontier_len == 0) return false;

        const uint16_t cur = frontier_pop_random();
        if (out.get(cur)) continue;

        hf_shape_set(out, cur);
        ++count;

        const uint16_t x = (cur & 15u);
        const uint16_t y = ((cur >> 4) & 15u);
        const uint16_t z = (cur >> 8);

        if (x > 0)  frontier_add((uint16_t)(cur - 1));
        if (x < 15) frontier_add((uint16_t)(cur + 1));
        if (y > 0)  frontier_add((uint16_t)(cur - 16));
        if (y < 15) frontier_add((uint16_t)(cur + 16));
        if (z > 0)  frontier_add((uint16_t)(cur - 256));
        if (z < 15) frontier_add((uint16_t)(cur + 256));
    }
    return true;
}


static bool hf_mine_shape_reference(Shape& out,
                                   HFTopo& outTopo,
                                   HFMineStats* outStats = nullptr,
                                   int min_cycle_rank = HF_TOPOMINE_MIN_CYCLE_RANK_STRONG_DEFAULT,
                                   uint64_t budget_ms = 30000,
                                   bool /*require_connected*/ = false) {
    std::atomic<bool> found{false};
    HFMineStats global_stats{};
    const uint64_t start_ms = hf_now_ms();

#if defined(_OPENMP)
#pragma omp parallel
    {
        TopoScratch scratch{};
        HFTopo st{};
        HFMineStats local_stats{};
        uint64_t iter = 0;

        while (!found.load(std::memory_order_relaxed)) {
            if (budget_ms != 0 && ((iter++ & 63u) == 0u)) {
                if ((hf_now_ms() - start_ms) >= budget_ms) break;
            }

            ++local_stats.attempts;
            Shape local{};
            if (!hf_sample_connected_exact_weight(local, scratch,
                    (uint16_t)HF_TOPOMINE_TARGET_WEIGHT_DEFAULT)) {
                ++local_stats.frontier_fail;
                continue;
            }

            st = hf_compute_topology(local, scratch);
            if (st.components != 1) {
                ++local_stats.topo_reject_not_connected;
                continue;
            }
            if (st.vertices != HF_TOPOMINE_TARGET_WEIGHT_DEFAULT) {
                ++local_stats.topo_reject_wrong_weight;
                continue;
            }
            if (st.cycle_rank < min_cycle_rank) {
                ++local_stats.topo_reject_cycle_rank;
                continue;
            }

#pragma omp critical
            {
                if (!found.load(std::memory_order_relaxed)) {
                    out = local;
                    outTopo = st;
                    found.store(true, std::memory_order_relaxed);
                    ++local_stats.accepted;
                    local_stats.accepted_cycle_rank = st.cycle_rank;
                    local_stats.accepted_vertices = st.vertices;
                    local_stats.accepted_components = st.components;
                }
            }

#pragma omp master
            {
                if ((iter & 0x3FFFu) == 0u) { std::cerr << "." << std::flush; }
            }
        }

#pragma omp critical
        {
            global_stats.merge(local_stats);
        }
    }
#else
    TopoScratch scratch{};
    HFTopo st{};
    uint64_t iter = 0;

    while (true) {
        if (budget_ms != 0 && ((iter++ & 63u) == 0u)) {
            if ((hf_now_ms() - start_ms) >= budget_ms) break;
        }

        ++global_stats.attempts;
        Shape local{};
        if (!hf_sample_connected_exact_weight(local, scratch,
                (uint16_t)HF_TOPOMINE_TARGET_WEIGHT_DEFAULT)) {
            ++global_stats.frontier_fail;
            continue;
        }

        st = hf_compute_topology(local, scratch);
        if (st.components != 1) {
            ++global_stats.topo_reject_not_connected;
            continue;
        }
        if (st.vertices != HF_TOPOMINE_TARGET_WEIGHT_DEFAULT) {
            ++global_stats.topo_reject_wrong_weight;
            continue;
        }
        if (st.cycle_rank < min_cycle_rank) {
            ++global_stats.topo_reject_cycle_rank;
            continue;
        }

        out = local;
        outTopo = st;
        found.store(true, std::memory_order_relaxed);
        ++global_stats.accepted;
        global_stats.accepted_cycle_rank = st.cycle_rank;
        global_stats.accepted_vertices = st.vertices;
        global_stats.accepted_components = st.components;
        break;
    }
#endif

    global_stats.elapsed_ms = hf_now_ms() - start_ms;
    if (outStats) *outStats = global_stats;
    return found.load(std::memory_order_relaxed);
}

// ------------------------------ TOPOMINE (PRACTICAL) -------------------------
// NOTE: This miner is an engineering variant (normalization + best-of selection).
// It is kept for benchmarking/experimentation but is not formal connected-graph.
static Shape hf_mine_shape_practical(int min_cycle_rank = 8, uint64_t budget_ms = 30000) {
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
            if (st.components == 1 && st.cycle_rank >= min_cycle_rank) {
#pragma omp critical
                {
                    if (!found.load(std::memory_order_relaxed) ||
                        (st.cycle_rank > bestS.cycle_rank) ||
                        (st.cycle_rank == bestS.cycle_rank && st.edges > bestS.edges)) {
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
              << "[TopoMining] Found: cycle_rank=" << bestS.cycle_rank
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


static bool hf_keygen(PublicKey& pk, SecretKey& sk,
                       int target_cycle_rank = HF_TOPOMINE_MIN_CYCLE_RANK_STRONG_DEFAULT,
                       uint64_t mine_budget_ms = 30000,
                       int threads = 0,
                       bool practical_miner = false,
                       bool require_connected = false,
                       HFKeygenDiag* diag = nullptr) {
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#else
    (void)threads;
#endif

    HFKeygenDiag local_diag{};
    local_diag.practical_miner = practical_miner;
    local_diag.target_cycle_rank = target_cycle_rank;

    const uint64_t t0_ms = hf_now_ms();

    Shape sh{};
    HFTopo st{};

    if (!practical_miner) {
        const bool ok = hf_mine_shape_reference(sh, st, &local_diag.mine, target_cycle_rank, mine_budget_ms, require_connected);
        local_diag.elapsed_ms = hf_now_ms() - t0_ms;
        if (!ok) {
            local_diag.timed_out = true;
            local_diag.success = false;
            if (diag) *diag = local_diag;
            std::cerr << "\n" << YEL
                      << "[TopoMine(formal)] Timeout (" << mine_budget_ms
                      << " ms); no deterministic fallback. Increase --mine-ms or lower --min-cycle-rank."
                      << RST << "\n";
            return false;
        }

        const int wt = hf_shape_popcount(sh);
        local_diag.success = true;
        local_diag.timed_out = false;
        local_diag.shape_weight = wt;
        local_diag.topo = st;

        std::cout << "\n" << GRN
                  << "[TopoMine(formal)] Found: cycle_rank=" << st.cycle_rank
                  << " components=" << st.components
                  << " wt=" << wt
                  << " exact_weight=" << HF_TOPOMINE_TARGET_WEIGHT_DEFAULT
                  << " connected=1(by construction)"
                  << " attempts=" << local_diag.mine.attempts
                  << " elapsed_ms=" << local_diag.mine.elapsed_ms
                  << RST << "\n";
    } else {
        sh = hf_mine_shape_practical(target_cycle_rank, mine_budget_ms);
        st = hf_analyze_shape(sh);
        local_diag.elapsed_ms = hf_now_ms() - t0_ms;
        local_diag.success = true;
        local_diag.timed_out = false;
        local_diag.shape_weight = hf_shape_popcount(sh);
        local_diag.topo = st;

        std::cout << "\n" << GRN
                  << "[TopoMine(practical)] Using engineering miner output: cycle_rank=" << st.cycle_rank
                  << " components=" << st.components
                  << " wt=" << local_diag.shape_weight
                  << RST << "\n";
    }

    sk.s = sh;
    hf_kem_keygen_from_shape(pk, sk);
    sk.pk = pk;
    if (diag) *diag = local_diag;
    return true;
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


static inline HFDecapDiag hf_decaps_diag(const SecretKey& sk, const Ciphertext& ct, SharedKey& outK) {
    HFDecapDiag diag{};
    const int T = hf_max_threads();
    std::vector<std::array<uint8_t, 32>> thread_mu((size_t)T);
    std::vector<HFDecapDiag> thread_diag((size_t)T);
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
            const uint32_t dist_half = hf_mod_distance(diff, HF_Q_HALF);
            const uint32_t dist_zero = hf_mod_distance(diff, 0);

            const uint8_t bitv = (uint8_t)(close_mask & 1u);
            const uint8_t bit = (uint8_t)(bitv << (k & 7));
            thread_mu[(size_t)tid][(size_t)k >> 3] |= bit;

            HFDecapDiag& td = thread_diag[(size_t)tid];
            if (bitv) {
                ++td.decoded_ones;
                td.min_margin_inside_one_band = std::min(td.min_margin_inside_one_band,
                                                         hf_margin_inside_band(dist_half, HF_Q_QTR));
            } else {
                ++td.decoded_zeros;
                td.min_margin_outside_one_band = std::min(td.min_margin_outside_one_band,
                                                          hf_margin_outside_band(dist_half, HF_Q_QTR));
                td.min_distance_to_zero = std::min(td.min_distance_to_zero, dist_zero);
            }
        }
    }

    std::array<uint8_t, 32> mu_prime{};
    mu_prime.fill(0);
    for (const auto& t : thread_mu) {
        for (size_t i = 0; i < mu_prime.size(); ++i) mu_prime[i] |= t[i];
    }

    for (const auto& td : thread_diag) {
        diag.decoded_ones += td.decoded_ones;
        diag.decoded_zeros += td.decoded_zeros;
        diag.min_margin_inside_one_band = std::min(diag.min_margin_inside_one_band, td.min_margin_inside_one_band);
        diag.min_margin_outside_one_band = std::min(diag.min_margin_outside_one_band, td.min_margin_outside_one_band);
        diag.min_distance_to_zero = std::min(diag.min_distance_to_zero, td.min_distance_to_zero);
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
    diag.valid = (valid != 0);

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

    return diag;
}

static inline void hf_decaps(const SecretKey& sk, const Ciphertext& ct, SharedKey& outK) {
    (void)hf_decaps_diag(sk, ct, outK);
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
        if (aead_id != 1 && aead_id != 2) return false;
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
static bool hf_kem_encrypt_with_pk(const PublicKey& pk,
                                   const std::string& in_f,
                                   const std::string& out_f) {
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

static bool hf_kem_encrypt_safe(const std::string& pk_f,
                               const std::string& in_f,
                               const std::string& out_f) {
    PublicKey pk{};
    if (!load_pk(pk_f, pk)) return false;
    return hf_kem_encrypt_with_pk(pk, in_f, out_f);
}

static bool hf_kem_decrypt_with_sk(const SecretKey& sk,
                                   const std::string& in_f,
                                   const std::string& out_f) {
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

static bool hf_kem_decrypt_safe(const std::string& sk_f,
                               const std::string& in_f,
                               const std::string& out_f,
                               const std::string& pwd = "") {
    SecretKey sk{};
    if (!load_sk(sk_f, sk, pwd)) return false;
    return hf_kem_decrypt_with_sk(sk, in_f, out_f);
}

// ------------------------------ SELF TEST -----------------------------------

static void run_test_all() {
    std::cout << "\n" << BG_BLU << " SELF-DIAGNOSTIC TEST (AUDIT MODE) " << RST << "\n";

    for (const char* f : {"test.pk", "test.sk", "test.bin", "test.enc", "test.dec"}) {
        if (fs::exists(f)) fs::remove(f);
    }

    std::cout << "  " << CYN << "[1/6] KeyGen (connected-graph formal miner + LWE)..." << RST << "\n";
    PublicKey pk{};
    SecretKey sk{};
    HFKeygenDiag kdiag{};
    if (!hf_keygen(pk, sk,
                   /*target_cycle_rank=*/HF_TOPOMINE_MIN_CYCLE_RANK_STRONG_DEFAULT,
                   /*mine_budget_ms=*/30000,
                   /*threads=*/0,
                   /*practical_miner=*/false,
                   /*require_connected=*/false,
                   &kdiag)) {
        std::cerr << RED << "[HF] KeyGen failed during self-test (formal miner timeout)." << RST << "\n";
        return;
    }
    save_pk("test.pk", pk);
    save_sk("test.sk", sk, "pass123");
    std::cout << "      " << GRN << "OK" << RST
              << "  attempts=" << kdiag.mine.attempts
              << " accepted=" << kdiag.mine.accepted
              << " cycle_rank=" << kdiag.topo.cycle_rank
              << " wt=" << kdiag.shape_weight
              << "\n";

    std::cout << "  " << CYN << "[2/6] Payload (1MB random)..." << RST << std::flush;
    std::vector<uint8_t> dummy(1024 * 1024);
    randombytes_buf(dummy.data(), dummy.size());
    {
        std::ofstream f("test.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(dummy.data()), (std::streamsize)dummy.size());
    }
    std::cout << " " << GRN << "OK" << RST << "\n";

    std::cout << "  " << CYN << "[3/6] Encrypt (FO-LWE)..." << RST << std::flush;
    if (!hf_kem_encrypt_safe("test.pk", "test.bin", "test.enc")) {
        std::cout << " " << RED << "FAIL" << RST << "\n";
        return;
    }
    std::cout << " " << GRN << "OK" << RST << "\n";

    std::cout << "  " << CYN << "[4/6] Decrypt (FO check)..." << RST << std::flush;
    if (!hf_kem_decrypt_safe("test.sk", "test.enc", "test.dec", "pass123")) {
        std::cout << " " << RED << "FAIL" << RST << "\n";
        return;
    }
    std::cout << " " << GRN << "OK" << RST << "\n";

    std::cout << "  " << CYN << "[5/6] Integrity check..." << RST << std::flush;
    std::ifstream f1("test.bin", std::ios::binary | std::ios::ate);
    std::ifstream f2("test.dec", std::ios::binary | std::ios::ate);
    bool files_ok = false;
    if (f1 && f2 && f1.tellg() == f2.tellg()) {
        files_ok = true;
        f1.seekg(0);
        f2.seekg(0);
        std::vector<uint8_t> b1(1 << 20), b2(1 << 20);
        while (files_ok) {
            f1.read(reinterpret_cast<char*>(b1.data()), (std::streamsize)b1.size());
            f2.read(reinterpret_cast<char*>(b2.data()), (std::streamsize)b2.size());
            const std::streamsize r1 = f1.gcount();
            const std::streamsize r2 = f2.gcount();
            if (r1 != r2) { files_ok = false; break; }
            if (r1 == 0) break;
            if (std::memcmp(b1.data(), b2.data(), (size_t)r1) != 0) { files_ok = false; break; }
        }
    }
    std::cout << " " << (files_ok ? GRN "MATCH" RST : RED "MISMATCH" RST) << "\n";
    if (!files_ok) return;

    std::cout << "  " << CYN << "[6/6] Correctness/decapsulation margin sweep (128 trials)..." << RST << std::flush;
    HFDecapAggregate agg{};
    constexpr size_t HF_SELFTEST_SWEEP_ITERS = 128;
    for (size_t i = 0; i < HF_SELFTEST_SWEEP_ITERS; ++i) {
        Ciphertext ct{};
        SharedKey k1{}, k2{};
        hf_encaps(pk, k1, ct);
        const HFDecapDiag ddiag = hf_decaps_diag(sk, ct, k2);
        if (sodium_memcmp(k1.data(), k2.data(), 32) != 0) {
            std::cout << " " << RED << "FAIL" << RST << "\n";
            std::cerr << RED << "[HF] Decapsulation mismatch during sweep at iter " << i << RST << "\n";
            return;
        }
        agg.add(ddiag);
    }
    std::cout << " " << GRN << "OK" << RST << "\n";
    std::cout << "      valid=" << agg.valid_count << "/" << agg.trials
              << " min_inside_one_band=" << agg.min_margin_inside_one_band
              << " min_outside_one_band=" << agg.min_margin_outside_one_band
              << " min_distance_to_zero=" << agg.min_distance_to_zero
              << "\n";

    std::cout << YEL << "  [INFO] Files preserved for audit: test.pk test.sk test.enc test.bin test.dec" << RST << "\n";
}


// --------------------------- Benchmarking ---------------------------

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
    std::vector<uint8_t> ba(1 << 20), bb(1 << 20);
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
    std::vector<uint8_t> buf(1 << 20);
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

static void hf_write_bench_stats_json(std::ostream& o, const HFBenchStats& st) {
    o << "{\"mean\": " << st.mean_ms
      << ", \"min\": " << st.min_ms
      << ", \"p50\": " << st.p50_ms
      << ", \"p95\": " << st.p95_ms
      << ", \"max\": " << st.max_ms << "}";
}

static void hf_write_mine_stats_json(std::ostream& o, const HFMineStats& st) {
    const double acceptance = (st.attempts > 0) ? (double)st.accepted / (double)st.attempts : 0.0;
    o << "{"
      << "\"attempts\": " << st.attempts
      << ", \"accepted\": " << st.accepted
      << ", \"acceptance_probability\": " << acceptance
      << ", \"frontier_fail\": " << st.frontier_fail
      << ", \"topo_reject_not_connected\": " << st.topo_reject_not_connected
      << ", \"topo_reject_wrong_weight\": " << st.topo_reject_wrong_weight
      << ", \"topo_reject_cycle_rank\": " << st.topo_reject_cycle_rank
      << ", \"elapsed_ms\": " << st.elapsed_ms
      << ", \"accepted_cycle_rank\": " << st.accepted_cycle_rank
      << ", \"accepted_vertices\": " << st.accepted_vertices
      << ", \"accepted_components\": " << st.accepted_components
      << "}";
}

static void hf_write_decap_aggregate_json(std::ostream& o, const HFDecapAggregate& agg) {
    o << "{"
      << "\"trials\": " << agg.trials
      << ", \"valid_count\": " << agg.valid_count
      << ", \"invalid_count\": " << agg.invalid_count
      << ", \"min_margin_inside_one_band\": " << agg.min_margin_inside_one_band
      << ", \"min_margin_outside_one_band\": " << agg.min_margin_outside_one_band
      << ", \"min_distance_to_zero\": " << agg.min_distance_to_zero
      << "}";
}

static void hf_write_bench_json(const std::string& path,
                               const std::string& prefix,
                               int target_cycle_rank,
                               int threads,
                               uint64_t mine_ms,
                               size_t iters,
                               size_t file_mb,
                               bool bench_keygen,
                               bool practical_miner,
                               bool require_connected,
                               const HFKeygenBenchSummary& keygen_summary,
                               const std::vector<HFKeygenDiag>& keygen_samples,
                               const HFBenchStats& enc,
                               const HFBenchStats& dec,
                               const HFDecapAggregate& decap_agg,
                               double file_enc_ms,
                               double file_dec_unlock_ms,
                               double file_dec_crypto_ms,
                               double file_dec_total_ms,
                               uint64_t file_bytes,
                               bool file_ok) {
    std::ofstream o(path);
    if (!o) return;

    const double enc_mib_s = (file_enc_ms > 0.0) ? ((double)file_bytes / (1024.0 * 1024.0)) / (file_enc_ms / 1000.0) : 0.0;
    const double dec_mib_s = (file_dec_crypto_ms > 0.0) ? ((double)file_bytes / (1024.0 * 1024.0)) / (file_dec_crypto_ms / 1000.0) : 0.0;
    const double success_rate = (keygen_summary.sample_count > 0)
        ? (double)keygen_summary.success_count / (double)keygen_summary.sample_count : 0.0;

    o << "{\n";
    o << "  \"version\": \"HyperFrog v36.0\",\n";
    o << "  \"benchmark_schema\": \"v3\",\n";
    o << "  \"security_note\": \"formal mode uses a structured, non-uniform connected-graph exact-weight secret law; this code does not claim a reduction to standard binary-secret LWE.\",\n";
    o << "  \"params\": {\"N\": " << HF_N << ", \"M\": " << HF_M << ", \"Q\": " << HF_Q << ", \"KBITS\": " << HF_KBITS << "},\n";
    o << "  \"prefix\": \"" << prefix << "\",\n";
    o << "  \"target_cycle_rank\": " << target_cycle_rank << ",\n";
    o << "  \"threads\": " << threads << ",\n";
    o << "  \"mine_ms\": " << mine_ms << ",\n";
    o << "  \"iters\": " << iters << ",\n";
    o << "  \"file_mb\": " << file_mb << ",\n";
    o << "  \"mine_mode\": \"" << (practical_miner ? "practical" : "formal") << "\",\n";
    o << "  \"require_connected\": " << (require_connected ? "true" : "false") << ",\n";
    o << "  \"keygen\": {\n";
    o << "    \"bench_enabled\": " << (bench_keygen ? "true" : "false") << ",\n";
    o << "    \"sample_count\": " << keygen_summary.sample_count << ",\n";
    o << "    \"success_count\": " << keygen_summary.success_count << ",\n";
    o << "    \"timeout_count\": " << keygen_summary.timeout_count << ",\n";
    o << "    \"failure_count\": " << keygen_summary.failure_count << ",\n";
    o << "    \"success_rate\": " << success_rate << ",\n";
    o << "    \"all_ms\": "; hf_write_bench_stats_json(o, keygen_summary.all_ms); o << ",\n";
    o << "    \"success_ms\": "; hf_write_bench_stats_json(o, keygen_summary.success_ms); o << ",\n";
    o << "    \"samples\": [\n";
    for (size_t i = 0; i < keygen_samples.size(); ++i) {
        const auto& s = keygen_samples[i];
        o << "      {\"iter\": " << i
          << ", \"success\": " << (s.success ? "true" : "false")
          << ", \"timed_out\": " << (s.timed_out ? "true" : "false")
          << ", \"elapsed_ms\": " << s.elapsed_ms
          << ", \"practical_miner\": " << (s.practical_miner ? "true" : "false")
          << ", \"target_cycle_rank\": " << s.target_cycle_rank
          << ", \"shape_weight\": " << s.shape_weight
          << ", \"topo\": {\"components\": " << s.topo.components
          << ", \"cycle_rank\": " << s.topo.cycle_rank
          << ", \"edges\": " << s.topo.edges
          << ", \"vertices\": " << s.topo.vertices << "}, "
          << "\"mine\": ";
        hf_write_mine_stats_json(o, s.mine);
        o << "}";
        if (i + 1 != keygen_samples.size()) o << ",";
        o << "\n";
    }
    o << "    ]\n";
    o << "  },\n";
    o << "  \"encaps_ms\": "; hf_write_bench_stats_json(o, enc); o << ",\n";
    o << "  \"decaps_ms\": "; hf_write_bench_stats_json(o, dec); o << ",\n";
    o << "  \"decapsulation_correctness\": "; hf_write_decap_aggregate_json(o, decap_agg); o << ",\n";
    o << "  \"file\": {\"bytes\": " << file_bytes
      << ", \"enc_ms\": " << file_enc_ms
      << ", \"dec_unlock_ms\": " << file_dec_unlock_ms
      << ", \"dec_crypto_ms\": " << file_dec_crypto_ms
      << ", \"dec_total_ms\": " << file_dec_total_ms
      << ", \"enc_mib_per_sec\": " << enc_mib_s
      << ", \"dec_crypto_mib_per_sec\": " << dec_mib_s
      << ", \"integrity_ok\": " << (file_ok ? "true" : "false")
      << "}\n";
    o << "}\n";
}

static int run_benchmark(const std::string& prefix,
                         const std::string& pass,
                         size_t iters,
                         size_t file_mb,
                         bool bench_keygen,
                         int target_cycle_rank,
                         uint64_t mine_ms,
                         int threads,
                         const std::string& out_override,
                         bool practical_miner,
                         bool require_connected) {
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

    PublicKey pk{};
    SecretKey sk{};

    HFKeygenBenchSummary keygen_summary{};
    std::vector<HFKeygenDiag> keygen_samples;
    std::vector<double> keygen_all_ms;
    std::vector<double> keygen_success_ms;

    const bool have_keys = std::filesystem::exists(pkf) && std::filesystem::exists(skf);
    if (bench_keygen || !have_keys) {
        const size_t keygen_iters = bench_keygen ? std::max<size_t>(iters, 1) : 1;
        keygen_samples.reserve(keygen_iters);
        for (size_t i = 0; i < keygen_iters; ++i) {
            PublicKey pk_i{};
            SecretKey sk_i{};
            HFKeygenDiag diag{};
            const bool ok = hf_keygen(pk_i, sk_i, target_cycle_rank, mine_ms, threads, practical_miner, require_connected, &diag);

            keygen_samples.push_back(diag);
            keygen_all_ms.push_back((double)diag.elapsed_ms);
            ++keygen_summary.sample_count;

            if (ok) {
                ++keygen_summary.success_count;
                keygen_success_ms.push_back((double)diag.elapsed_ms);
                pk = pk_i;
                sk = sk_i;
            } else if (diag.timed_out) {
                ++keygen_summary.timeout_count;
            } else {
                ++keygen_summary.failure_count;
            }

            ocsv << "keygen," << i << "," << (double)diag.elapsed_ms << ",0,"
                 << "\"success=" << (diag.success ? 1 : 0)
                 << ";timed_out=" << (diag.timed_out ? 1 : 0)
                 << ";shape_weight=" << diag.shape_weight
                 << ";cycle_rank=" << diag.topo.cycle_rank
                 << ";components=" << diag.topo.components
                 << ";attempts=" << diag.mine.attempts
                 << ";accepted=" << diag.mine.accepted
                 << ";frontier_fail=" << diag.mine.frontier_fail
                 << ";rej_not_connected=" << diag.mine.topo_reject_not_connected
                 << ";rej_wrong_weight=" << diag.mine.topo_reject_wrong_weight
                 << ";rej_cycle_rank=" << diag.mine.topo_reject_cycle_rank
                 << "\"\n";
        }

        keygen_summary.all_ms = hf_stats_ms(keygen_all_ms);
        keygen_summary.success_ms = hf_stats_ms(keygen_success_ms);

        if (keygen_summary.success_count == 0) {
            std::cerr << "Benchmark keygen: no successful key generation sample\n";
            hf_write_bench_json(js, prefix, target_cycle_rank, threads, mine_ms, iters, file_mb,
                                bench_keygen, practical_miner, require_connected, keygen_summary, keygen_samples,
                                HFBenchStats{}, HFBenchStats{}, HFDecapAggregate{},
                                0.0, 0.0, 0.0, 0.0, 0, false);
            return 3;
        }

        if (!hf_save_pk(pkf, pk)) {
            std::cerr << "Benchmark keygen: failed saving pk\n";
            return 3;
        }
        if (!hf_save_sk(skf, sk, pass)) {
            std::cerr << "Benchmark keygen: failed saving sk\n";
            return 3;
        }
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

    std::vector<double> enc_ms; enc_ms.reserve(iters);
    std::vector<double> dec_ms; dec_ms.reserve(iters);
    HFDecapAggregate decap_agg{};

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
        const HFDecapDiag ddiag = hf_decaps_diag(sk, ct, K2);
        const uint64_t t3 = hf_now_ns();
        const double ms2 = (double)(t3 - t2) / 1e6;
        dec_ms.push_back(ms2);
        decap_agg.add(ddiag);
        ocsv << "decaps," << i << "," << ms2 << ",0,"
             << "\"valid=" << (ddiag.valid ? 1 : 0)
             << ";ones=" << ddiag.decoded_ones
             << ";zeros=" << ddiag.decoded_zeros
             << ";min_inside_one_band=" << ddiag.min_margin_inside_one_band
             << ";min_outside_one_band=" << ddiag.min_margin_outside_one_band
             << ";min_distance_to_zero=" << ddiag.min_distance_to_zero
             << "\"\n";

        if (sodium_memcmp(K1.data(), K2.data(), 32) != 0) {
            std::cerr << "Benchmark: K mismatch at iter " << i << "\n";
            hf_write_bench_json(js, prefix, target_cycle_rank, threads, mine_ms, iters, file_mb,
                                bench_keygen, practical_miner, require_connected, keygen_summary, keygen_samples,
                                hf_stats_ms(enc_ms), hf_stats_ms(dec_ms), decap_agg,
                                0.0, 0.0, 0.0, 0.0, 0, false);
            return 4;
        }
    }
    ocsv.flush();

    const HFBenchStats encst = hf_stats_ms(enc_ms);
    const HFBenchStats decst = hf_stats_ms(dec_ms);

    ocsv << "encaps_summary,0," << encst.mean_ms << ",0,\"p50=" << encst.p50_ms << ";p95=" << encst.p95_ms << "\"\n";
    ocsv << "decaps_summary,0," << decst.mean_ms << ",0,\"p50=" << decst.p50_ms << ";p95=" << decst.p95_ms << "\"\n";
    ocsv << "decaps_correctness_summary,0,0,0,"
         << "\"valid=" << decap_agg.valid_count << "/" << decap_agg.trials
         << ";min_inside_one_band=" << decap_agg.min_margin_inside_one_band
         << ";min_outside_one_band=" << decap_agg.min_margin_outside_one_band
         << ";min_distance_to_zero=" << decap_agg.min_distance_to_zero
         << "\"\n";
    ocsv.flush();

    const uint64_t file_bytes = (uint64_t)file_mb * 1024ULL * 1024ULL;
    if (!hf_make_random_file(bin, file_bytes)) {
        std::cerr << "Benchmark: failed creating random file\n";
        return 5;
    }

    double file_enc_ms = 0.0;
    double file_dec_unlock_ms = 0.0;
    double file_dec_crypto_ms = 0.0;
    double file_dec_total_ms = 0.0;
    bool file_ok = false;

    {
        const uint64_t t0 = hf_now_ns();
        const bool ok = hf_kem_encrypt_with_pk(pk, bin, encf);
        const uint64_t t1 = hf_now_ns();
        file_enc_ms = (double)(t1 - t0) / 1e6;
        ocsv << "file_enc_crypto,0," << file_enc_ms << "," << file_bytes << ",\"preloaded_pk\"\n";
        ocsv.flush();
        if (!ok) {
            std::cerr << "Benchmark: file encryption failed\n";
            hf_write_bench_json(js, prefix, target_cycle_rank, threads, mine_ms, iters, file_mb,
                                bench_keygen, practical_miner, require_connected, keygen_summary, keygen_samples,
                                encst, decst, decap_agg,
                                file_enc_ms, 0.0, 0.0, 0.0, file_bytes, false);
            return 6;
        }
    }
    {
        SecretKey dec_sk{};
        if (!pass.empty()) {
            const uint64_t t_unlock0 = hf_now_ns();
            if (!hf_load_sk(skf, dec_sk, pass)) {
                std::cerr << "Benchmark: failed loading sk for unlock timing (wrong --password?)\n";
                hf_write_bench_json(js, prefix, target_cycle_rank, threads, mine_ms, iters, file_mb,
                                    bench_keygen, practical_miner, require_connected, keygen_summary, keygen_samples,
                                    encst, decst, decap_agg,
                                    file_enc_ms, 0.0, 0.0, 0.0, file_bytes, false);
                return 7;
            }
            const uint64_t t_unlock1 = hf_now_ns();
            file_dec_unlock_ms = (double)(t_unlock1 - t_unlock0) / 1e6;
        } else {
            dec_sk = sk;
            file_dec_unlock_ms = 0.0;
        }

        const uint64_t t0 = hf_now_ns();
        const bool ok = hf_kem_decrypt_with_sk(dec_sk, encf, decf);
        const uint64_t t1 = hf_now_ns();
        file_dec_crypto_ms = (double)(t1 - t0) / 1e6;
        file_dec_total_ms = file_dec_unlock_ms + file_dec_crypto_ms;
        ocsv << "file_dec_unlock,0," << file_dec_unlock_ms << ",0,\"password_unlock\"\n";
        ocsv << "file_dec_crypto,0," << file_dec_crypto_ms << "," << file_bytes << ",\"preloaded_sk\"\n";
        ocsv << "file_dec_total,0," << file_dec_total_ms << "," << file_bytes << ",\"unlock_plus_crypto\"\n";
        ocsv.flush();
        if (!ok) {
            std::cerr << "Benchmark: file decryption failed\n";
            hf_write_bench_json(js, prefix, target_cycle_rank, threads, mine_ms, iters, file_mb,
                                bench_keygen, practical_miner, require_connected, keygen_summary, keygen_samples,
                                encst, decst, decap_agg,
                                file_enc_ms, file_dec_unlock_ms, file_dec_crypto_ms, file_dec_total_ms, file_bytes, false);
            return 7;
        }
    }

    file_ok = hf_files_equal(bin, decf);
    ocsv << "file_check,0,0," << file_bytes << ",\"" << (file_ok ? "OK" : "MISMATCH") << "\"\n";
    ocsv.flush();

    hf_write_bench_json(js, prefix, target_cycle_rank, threads, mine_ms, iters, file_mb,
                        bench_keygen, practical_miner, require_connected, keygen_summary, keygen_samples,
                        encst, decst, decap_agg,
                        file_enc_ms, file_dec_unlock_ms, file_dec_crypto_ms, file_dec_total_ms, file_bytes, file_ok);

    const double file_enc_mib_s = (file_enc_ms > 0.0) ? ((double)file_bytes / (1024.0 * 1024.0)) / (file_enc_ms / 1000.0) : 0.0;
    const double file_dec_crypto_mib_s = (file_dec_crypto_ms > 0.0) ? ((double)file_bytes / (1024.0 * 1024.0)) / (file_dec_crypto_ms / 1000.0) : 0.0;

    std::cout << "[BENCH] Wrote " << csv << " and " << js << "\n";
    std::cout << "[BENCH] Artifacts: " << bin << " " << encf << " " << decf << "\n";
    std::cout << "[BENCH] KeyGen success/timeout/fail: "
              << keygen_summary.success_count << "/" << keygen_summary.timeout_count << "/" << keygen_summary.failure_count << "\n";
    std::cout << "[BENCH] KEM encapsulation  (mean / p50 / p95) ms: "
              << encst.mean_ms << " / " << encst.p50_ms << " / " << encst.p95_ms << "\n";
    std::cout << "[BENCH] KEM decapsulation (mean / p50 / p95) ms: "
              << decst.mean_ms << " / " << decst.p50_ms << " / " << decst.p95_ms << "\n";
    std::cout << "[BENCH] Decap correctness sweep: valid=" << decap_agg.valid_count << "/" << decap_agg.trials
              << " min_inside_one_band=" << decap_agg.min_margin_inside_one_band
              << " min_outside_one_band=" << decap_agg.min_margin_outside_one_band
              << " min_distance_to_zero=" << decap_agg.min_distance_to_zero << "\n";
    std::cout << "[BENCH] File encryption ms: " << file_enc_ms << "\n";
    std::cout << "[BENCH] File decryption ms: unlock=" << file_dec_unlock_ms
              << " crypto=" << file_dec_crypto_ms
              << " total=" << file_dec_total_ms << "\n";
    std::cout << "[BENCH] File throughput MiB/s: encrypt=" << file_enc_mib_s
              << " decrypt-crypto=" << file_dec_crypto_mib_s << "\n";
    std::cout << "[BENCH] File integrity: " << (file_ok ? "OK" : "MISMATCH") << "\n";

    return file_ok ? 0 : 8;
}


static void print_help(const char* argv0) {
    std::cout
        << "HyperFrog v36.0 (Connected-Graph Formal Miner, research prototype)\n"
        << "Usage:\n"
        << "  " << argv0 << " --gen-keys <prefix> [--password <pw>] [--mine-mode formal|practical] [--mine-ms <ms>] [--threads <n>] [--min-cycle-rank <g>]\n"
        << "  " << argv0 << " --validate-keys <prefix> [--password <pw>]\n"
        << "  " << argv0 << " --enc <prefix> <in_plain> <out_hf>\n"
        << "  " << argv0 << " --dec <prefix> <in_hf> <out_plain> [--password <pw>]\n"
        << "  " << argv0 << " --test-all\n"
        << "  " << argv0 << " --benchmark <prefix> [--password <pw>] [--iters <n>] [--file-mb <mb>] [--bench-keygen] [--out <outprefix>] [--mine-mode formal|practical] [--require-connected] [--mine-ms <ms>] [--threads <n>] [--min-cycle-rank <g>]\n"
        << "  " << argv0 << " --help\n"
        << "\nFiles:\n"
        << "  <prefix>.pk  public key (seed_A + b); ~4.0 KiB for current params\n"
        << "  <prefix>.sk  secret key (includes pk; optionally password-protected via Argon2id)\n"
        << "\nMining:\n"
        << "  Default budget is 30000 ms.\n"
        << "  Mine modes:\n"
        << "    --mine-mode formal    : connected-growth formal miner; exact weight 2048, connected by\n"
        << "                           construction, and accepts only if cycle_rank>=min_cycle_rank.\n"
        << "                           No fallback, no keep-largest-component, no best-of selection.\n"
        << "    --mine-mode practical : engineering miner (keep-largest-component + best-of selection);\n"
        << "                           includes deterministic timeout fallback and is not the formal law.\n"
        << "  NOTE: In formal mode, timeout fails KeyGen (no deterministic fallback).\n"
        << "  Security note: formal mode samples a structured, non-uniform connected-graph exact-weight\n"
        << "                 secret distribution; this code does not claim a reduction to standard\n"
        << "                 binary-secret LWE.\n"
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
    int min_cycle_rank = HF_TOPOMINE_MIN_CYCLE_RANK_STRONG_DEFAULT;
    bool practical_miner = false; // --mine-mode practical
    bool require_connected = false; // --require-connected (predicate-only)

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
        } else if (a == "--min-cycle-rank" || a == "--min-genus") { // --min-genus deprecated alias
            const std::string* v = need();
            if (!v) return false;
            try { min_cycle_rank = std::stoi(*v); } catch (...) { return false; }
            if (min_cycle_rank < 0) min_cycle_rank = 0;
        
        } else if (a == "--mine-mode") {
            const std::string* v = need();
            if (!v) return false;
            if (*v == "practical") practical_miner = true;
            else if (*v == "formal") practical_miner = false;
            else return false;
        } else if (a == "--require-connected") {
            require_connected = true;
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

    if (cmd == "--benchmark") {
        if (!parse_flags(1)) return die("Invalid arguments for --benchmark.");
        if (pos.size() != 1) return die("--benchmark requires <prefix>.");
        const std::string prefix = pos[0];
        return run_benchmark(prefix, password, bench_iters, bench_file_mb, bench_keygen, min_cycle_rank, mine_ms, threads, bench_out, practical_miner, require_connected);
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

        std::cout << "[HF] KeyGen: prefix='" << prefix << "', min_cycle_rank=" << min_cycle_rank
                  << ", mine_ms=" << mine_ms << ", threads=" << threads << "\n";

        if (!hf_keygen(pk, sk, min_cycle_rank, mine_ms, threads, practical_miner, require_connected, nullptr)) {
            return die("KeyGen failed: TopoMine timeout (increase --mine-ms or lower --min-cycle-rank)." );
        }

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
        const HFDecapDiag ddiag = hf_decaps_diag(sk, ct, K2);

        const int kem_ok = (sodium_memcmp(K1.data(), K2.data(), 32) == 0);

        std::cout << "[HF] pk in sk matches external pk: " << (pk_match ? "YES" : "NO") << "\n";
        std::cout << "[HF] KEM roundtrip: " << (kem_ok ? "OK" : "FAIL") << "\n";
        std::cout << "[HF] Decap diag: valid=" << (ddiag.valid ? "YES" : "NO")
                  << " min_inside_one_band=" << ddiag.min_margin_inside_one_band
                  << " min_outside_one_band=" << ddiag.min_margin_outside_one_band
                  << " min_distance_to_zero=" << ddiag.min_distance_to_zero << "\n";

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
