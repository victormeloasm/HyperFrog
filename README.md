# HyperFrog
The HyperFrog Cryptosystem: High-Genus Voxel Topology as a Trapdoor for Post-Quantum KEMs

Compiling: 
clang++ -std=c++23 -O3 -march=native -mtune=native -flto -fopenmp -fuse-ld=lld \
  hyperfrog36.cpp -o hyperfrog36 \
  -lomp -lsodium -lssl -lcrypto

Testing: 
./hyperfrog36 --benchmark formal_pw --password "123" \
  --iters 20 --file-mb 32 --out bench_formal_pw_v36 \
  --mine-mode formal --mine-ms 30000 --threads 32 --min-cycle-rank 2700
