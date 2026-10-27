# README

Accompanying repository to the manuscript titled **Weak Instances of the Two Matrix Code Equivalence Problem**.

## How to use the code

```bash
sage -python algorithm1.py --help
usage: algorithm1.py [-h] [--family {2-sMCE,sIMCE}] [--k K] [--m M] [--n N] [--q Q] [--eps-a {1,-1}] [--eps-b {1,-1}] [--seed SEED]

Algorithm 1 on a random 2-sMCE / sIMCE instance

options:
  -h, --help            show this help message and exit
  --family {2-sMCE,sIMCE}
  --k K
  --m M
  --n N
  --q Q
  --eps-a {1,-1}
  --eps-b {1,-1}
  --seed SEED
```

## Testing alrgotihm 1

```sh
# sIMCE
sage -python algorithm1.py --family sIMCE --k 20 --m 20 --n 20 --q 4093
sage -python algorithm1.py --family sIMCE --k 20 --m 20 --n 20 --q 4093 --eps-a -1
sage -python algorithm1.py --family sIMCE --k 20 --m 20 --n 20 --q 4093 --eps-b -1
sage -python algorithm1.py --family sIMCE --k 20 --m 20 --n 20 --q 4093 --eps-a -1 --eps-b -1

sage -python algorithm1.py --family sIMCE --k 31 --m 31 --n 31 --q 4093

sage -python algorithm1.py --family sIMCE --k 42 --m 42 --n 42 --q 4093
sage -python algorithm1.py --family sIMCE --k 42 --m 42 --n 42 --q 4093 --eps-a -1
sage -python algorithm1.py --family sIMCE --k 42 --m 42 --n 42 --q 4093 --eps-b -1
sage -python algorithm1.py --family sIMCE --k 42 --m 42 --n 42 --q 4093 --eps-a -1 --eps-b -1

sage -python algorithm1.py --family sIMCE --k 46 --m 46 --n 46 --q 4093
sage -python algorithm1.py --family sIMCE --k 46 --m 46 --n 46 --q 4093 --eps-a -1
sage -python algorithm1.py --family sIMCE --k 46 --m 46 --n 46 --q 4093 --eps-b -1
sage -python algorithm1.py --family sIMCE --k 46 --m 46 --n 46 --q 4093 --eps-a -1 --eps-b -1

# 2-sMCE
sage -python algorithm1.py --family 2-sMCE --k 20 --m 20 --n 20 --q 4093
sage -python algorithm1.py --family 2-sMCE --k 20 --m 20 --n 20 --q 4093 --eps-a -1
sage -python algorithm1.py --family 2-sMCE --k 20 --m 20 --n 20 --q 4093 --eps-b -1
sage -python algorithm1.py --family 2-sMCE --k 20 --m 20 --n 20 --q 4093 --eps-a -1 --eps-b -1

sage -python algorithm1.py --family 2-sMCE --k 31 --m 31 --n 31 --q 4093

sage -python algorithm1.py --family 2-sMCE --k 42 --m 42 --n 42 --q 4093
sage -python algorithm1.py --family 2-sMCE --k 42 --m 42 --n 42 --q 4093 --eps-a -1
sage -python algorithm1.py --family 2-sMCE --k 42 --m 42 --n 42 --q 4093 --eps-b -1
sage -python algorithm1.py --family 2-sMCE --k 42 --m 42 --n 42 --q 4093 --eps-a -1 --eps-b -1

sage -python algorithm1.py --family 2-sMCE --k 46 --m 46 --n 46 --q 4093
sage -python algorithm1.py --family 2-sMCE --k 46 --m 46 --n 46 --q 4093 --eps-a -1
sage -python algorithm1.py --family 2-sMCE --k 46 --m 46 --n 46 --q 4093 --eps-b -1
sage -python algorithm1.py --family 2-sMCE --k 46 --m 46 --n 46 --q 4093 --eps-a -1 --eps-b -1
```

## Experiments on the heuristics

We provide two scripts for the experiments on the heuristics.

```bash
sage -python experiment_small_parameters.py --help
usage: experiment_small_parameters.py [-h] [--trials TRIALS] [--q Q] [--cores CORES] [--shapes SHAPES]

options:
  -h, --help       show this help message and exit
  --trials TRIALS
  --q Q
  --cores CORES
  --shapes SHAPES  comma-separated k:m:n list (default: all shapes of the paper)
```

```bash
sage -python experiment_field_size.py --help
usage: experiment_field_size.py [-h] [--k K] [--trials TRIALS] [--batch BATCH] [--cores CORES]

options:
  -h, --help       show this help message and exit
  --k K
  --trials TRIALS
  --batch BATCH
  --cores CORES
```

```bash
# Heuristics
sage -python experiment_small_parameters.py --trials 25 --cores 10
sage -python experiment_field_size.py --k 20 --trials 25 --cores 10
```