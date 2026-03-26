#!/usr/bin/env bash
set -Eeuo pipefail

SRC="${1:-hyperfrog_v4_ready.cpp}"
BIN="${2:-./hyperfrog}"
PASS="${HYPERFROG_PASS:-hunter2}"

TS="$(date +%Y%m%d_%H%M%S)"
ROOT="hyperfrog_run_${TS}"
mkdir -p "$ROOT"
LOG="$ROOT/run.log"

exec > >(tee -a "$LOG") 2>&1

echo "[*] ROOT=$ROOT"
echo "[*] SRC=$SRC"
echo "[*] BIN=$BIN"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "[!] missing command: $1"; exit 1; }
}

need clang++
need cmp
need awk
need date

echo "[*] Compiling with clang++..."
clang++ -std=c++23 -O3 -DNDEBUG -march=native -mtune=native -flto=thin -fuse-ld=lld -fopenmp \
  "$SRC" -lsodium -lssl -lcrypto -o "$BIN"

if command -v strip >/dev/null 2>&1; then
  strip -s "$BIN" || true
fi

echo
echo "===================="
echo "[1] HELP / SMOKE"
echo "===================="
"$BIN" --help | tee "$ROOT/help.txt"
"$BIN" --test-all | tee "$ROOT/test_all.txt"

echo
echo "===================="
echo "[2] FORMAL KEYGEN / VALIDATE"
echo "===================="
"$BIN" --gen-keys "$ROOT/hf_formal_5s"  --password "$PASS" --mine-mode formal --mine-ms 5000  --threads 8  --min-cycle-rank 8
"$BIN" --validate-keys "$ROOT/hf_formal_5s"  --password "$PASS" | tee "$ROOT/validate_formal_5s.txt"

"$BIN" --gen-keys "$ROOT/hf_formal_30s" --password "$PASS" --mine-mode formal --mine-ms 30000 --threads 16 --min-cycle-rank 8
"$BIN" --validate-keys "$ROOT/hf_formal_30s" --password "$PASS" | tee "$ROOT/validate_formal_30s.txt"

echo
echo "===================="
echo "[3] PRACTICAL KEYGEN / VALIDATE"
echo "===================="
"$BIN" --gen-keys "$ROOT/hf_practical" --password "$PASS" --mine-mode practical --mine-ms 5000 --threads 8 --min-cycle-rank 8
"$BIN" --validate-keys "$ROOT/hf_practical" --password "$PASS" | tee "$ROOT/validate_practical.txt"

echo
echo "===================="
echo "[4] NOPASS KEYGEN / VALIDATE"
echo "===================="
"$BIN" --gen-keys "$ROOT/hf_nopass" --mine-mode formal --mine-ms 10000 --threads 16 --min-cycle-rank 8
"$BIN" --validate-keys "$ROOT/hf_nopass" | tee "$ROOT/validate_nopass.txt"

echo
echo "===================="
echo "[5] ENC / DEC SANITY"
echo "===================="
printf 'hyperfrog sanity check\n' > "$ROOT/input.bin"

"$BIN" --enc "$ROOT/hf_formal_30s" "$ROOT/input.bin" "$ROOT/output_formal.hf"
"$BIN" --dec "$ROOT/hf_formal_30s" "$ROOT/output_formal.hf" "$ROOT/recovered_formal.bin" --password "$PASS"
cmp -s "$ROOT/input.bin" "$ROOT/recovered_formal.bin" && echo "[OK] formal enc/dec" || { echo "[FAIL] formal enc/dec"; exit 1; }

"$BIN" --enc "$ROOT/hf_practical" "$ROOT/input.bin" "$ROOT/output_practical.hf"
"$BIN" --dec "$ROOT/hf_practical" "$ROOT/output_practical.hf" "$ROOT/recovered_practical.bin" --password "$PASS"
cmp -s "$ROOT/input.bin" "$ROOT/recovered_practical.bin" && echo "[OK] practical enc/dec" || { echo "[FAIL] practical enc/dec"; exit 1; }

echo
echo "===================="
echo "[6] REPEATED TEST-ALL"
echo "===================="
for i in $(seq 1 10); do
  echo "[RUN $i]"
  "$BIN" --test-all | tee "$ROOT/test_all_repeat_${i}.txt"
done

echo
echo "===================="
echo "[7] LIGHT / MID / HEAVY BENCH"
echo "===================="
"$BIN" --benchmark "$ROOT/bench_light_keys" --password "$PASS" --iters 20  --file-mb 4   --bench-keygen --out "$ROOT/bench_light" --mine-mode formal --mine-ms 5000  --threads 8  --min-cycle-rank 8
"$BIN" --benchmark "$ROOT/bench_mid_keys"   --password "$PASS" --iters 100 --file-mb 16  --bench-keygen --out "$ROOT/bench_mid"   --mine-mode formal --mine-ms 10000 --threads 16 --min-cycle-rank 8
"$BIN" --benchmark "$ROOT/bench_heavy_keys" --password "$PASS" --iters 500 --file-mb 128 --bench-keygen --out "$ROOT/bench_heavy" --mine-mode formal --mine-ms 30000 --threads 32 --min-cycle-rank 8

echo
echo "===================="
echo "[8] FORMAL vs PRACTICAL"
echo "===================="
"$BIN" --benchmark "$ROOT/cmp_formal_keys"    --password "$PASS" --iters 200 --file-mb 64 --bench-keygen --out "$ROOT/cmp_formal"    --mine-mode formal    --mine-ms 30000 --threads 16 --min-cycle-rank 8
"$BIN" --benchmark "$ROOT/cmp_practical_keys" --password "$PASS" --iters 200 --file-mb 64 --bench-keygen --out "$ROOT/cmp_practical" --mine-mode practical --mine-ms 30000 --threads 16 --min-cycle-rank 8

echo
echo "===================="
echo "[9] BUDGET SWEEP"
echo "===================="
for ms in 1000 5000 10000 30000; do
  "$BIN" --benchmark "$ROOT/mine_${ms}_keys" --password "$PASS" --iters 50 --file-mb 8 --bench-keygen --out "$ROOT/mine_${ms}" --mine-mode formal --mine-ms "$ms" --threads 16 --min-cycle-rank 8
done

echo
echo "===================="
echo "[10] CYCLE-RANK SWEEP"
echo "===================="
for g in 6 8 10 12; do
  "$BIN" --benchmark "$ROOT/rank${g}_keys" --password "$PASS" --iters 50 --file-mb 8 --bench-keygen --out "$ROOT/rank${g}" --mine-mode formal --mine-ms 10000 --threads 16 --min-cycle-rank "$g"
done

echo
echo "===================="
echo "[11] THREAD SWEEP"
echo "===================="
for th in 1 8 16 32; do
  "$BIN" --benchmark "$ROOT/th${th}_keys" --password "$PASS" --iters 100 --file-mb 16 --bench-keygen --out "$ROOT/th${th}" --mine-mode formal --mine-ms 10000 --threads "$th" --min-cycle-rank 8
done

echo
echo "===================="
echo "[12] LARGE FILE"
echo "===================="
"$BIN" --benchmark "$ROOT/file256_keys" --password "$PASS" --iters 50 --file-mb 256 --bench-keygen --out "$ROOT/file256" --mine-mode formal --mine-ms 30000 --threads 16 --min-cycle-rank 8

echo
echo "===================="
echo "[13] SUMMARY"
echo "===================="
find "$ROOT" -maxdepth 1 \( -name '*.json' -o -name '*.csv' -o -name '*.pk' -o -name '*.sk' -o -name '*.hf' -o -name '*.bin' -o -name '*.dec' \) | sort | tee "$ROOT/artifacts.txt"

echo
echo "[*] Done. Main log: $LOG"
echo "[*] Root dir:  $ROOT"