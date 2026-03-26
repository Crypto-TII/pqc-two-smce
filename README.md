# README

Accompanying repository to the manuscript titled **Weak Instances of the Two Matrix Code Equivalence Problem**

## Requirements

```python
import sys
import argparse
from random import choice
import numpy as np
from time import time
    
# SageMath imports
from sage.all import (
    matrix,
    identity_matrix,
    zero_matrix,
    FiniteField,
    vector,
)
```

## Instructions

```bash
% sage -python two_smce.py                       
usage: two_smce.py [-h] [-k CODE_DIMENSION] [-m CODE_LENGTH_M] -n CODE_LENGTH_N -q PRIME [-i]

Parses command.

options:
  -h, --help            show this help message and exit
  -k, --code_dimension CODE_DIMENSION
                        Code dimension parameter k
  -m, --code_length_m CODE_LENGTH_M
                        Code length parameter m
  -n, --code_length_n CODE_LENGTH_N
                        Code length parameter n
  -q, --prime PRIME     Field characteristic
  -i, --simce           For solving sIMCE instances
```

Solver can be tested as follows

```bash
# 2-sMCE
sage -python two_smce.py -q 4093 -k 20 -m 20 -n 20
sage -python two_smce.py -q 4093 -k 31 -m 31 -n 31
sage -python two_smce.py -q 2039 -k 42 -m 42 -n 42

# sIMCE
sage -python two_smce.py -q 4093 -k 20 -m 20 -n 20 --simce
sage -python two_smce.py -q 4093 -k 31 -m 31 -n 31 --simce
sage -python two_smce.py -q 2039 -k 42 -m 42 -n 42 --simce
```

## Remarks

The code correctly runs on the **SageMath version**: `SageMath version 10.8, Release Date: 2025-12-18`.

## License

Apache License Version 2.0, January 2004
