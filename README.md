# HyperFrog
The HyperFrog Cryptosystem: High-Genus Voxel Topology as a Trapdoor for Post-Quantum KEMs

Compiling: 
clang++ -std=c++23 -O3 -march=native -mtune=native -flto -fopenmp -fuse-ld=lld \
  hyperfrog.cpp -o hyperfrog \
  -lomp -lsodium -lssl -lcrypto

