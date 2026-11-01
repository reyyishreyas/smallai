# Results

Accuracy (%). Test A = clean, Test B = sensor noise, Test C = distribution shift.

| Model | Test A acc | Test A F1 | Test B acc | Test B F1 | Test C acc | Test C F1 |
| --- | --- | --- | --- | --- | --- | --- |
| **M** | 87.16 | 86.19 | 86.58 | 85.59 | 88.60 | 86.70 |
| B2 | 70.18 | 66.37 | 70.68 | 67.01 | 73.06 | 67.06 |
| B1 | 71.88 | 69.74 | 72.10 | 70.13 | 74.96 | 69.68 |
| B0 | 46.24 | 23.52 | 45.90 | 22.37 | 54.66 | 24.17 |

**Journey-learning delta on Test C (M − B2): +15.54 points**

## Step-level metrics

| Model | Test | Step 1 acc | Step 2 acc | Step 3 acc | Step-consistency |
| --- | --- | --- | --- | --- | --- |
| M | testA | 92.76 | 87.28 | 92.08 | 98.32 |
| M | testB | 92.66 | 86.60 | 92.34 | 98.60 |
| M | testC | 92.60 | 86.26 | 92.56 | 97.80 |
| B2 | testA | 77.78 | 72.48 | 80.62 | 85.84 |
| B2 | testB | 79.96 | 73.06 | 82.08 | 85.82 |
| B2 | testC | 76.96 | 70.36 | 80.98 | 86.84 |

Low-confidence detection for M (threshold 0.5 of the verifier): testA=8.00%, testB=9.00%, testC=15.50%

## Verifier quality (corrupted-step detection)

- AUC: 0.9381
- Precision @0.5: 0.9475
- Recall @0.5: 0.9336
- AUC per corruption strategy: random_flip=0.9504, neighbour=0.9286, shortcut=n/a (steps always valid)

## Model size

- Full model (M): 5,374 parameters
- Shortcut MLP (B1): 2,004 parameters

## LLM cost proxy (B3, analytical)

- Model: gpt-4o-mini (published pricing)
- Tokens per decision: ~500
- Cost per 1,000 decisions: $0.1110
- Cost per 1M decisions: $111.00
- Latency p50/p95: 900/2200 ms

### Confusion matrix — M / testA

| true\pred | GO | GO_REDUCED_PAYLOAD | HOLD | ABORT |
| --- | --- | --- | --- | --- |
| GO | 706 | 83 | 0 | 0 |
| GO_REDUCED_PAYLOAD | 117 | 1063 | 46 | 153 |
| HOLD | 3 | 29 | 548 | 75 |
| ABORT | 0 | 103 | 33 | 2041 |

### Confusion matrix — M / testB

| true\pred | GO | GO_REDUCED_PAYLOAD | HOLD | ABORT |
| --- | --- | --- | --- | --- |
| GO | 693 | 72 | 0 | 0 |
| GO_REDUCED_PAYLOAD | 129 | 1061 | 46 | 163 |
| HOLD | 5 | 30 | 547 | 70 |
| ABORT | 0 | 107 | 49 | 2028 |

### Confusion matrix — M / testC

| true\pred | GO | GO_REDUCED_PAYLOAD | HOLD | ABORT |
| --- | --- | --- | --- | --- |
| GO | 413 | 60 | 0 | 0 |
| GO_REDUCED_PAYLOAD | 56 | 819 | 31 | 123 |
| HOLD | 2 | 45 | 755 | 76 |
| ABORT | 0 | 92 | 85 | 2443 |

### Confusion matrix — B2 / testA

| true\pred | GO | GO_REDUCED_PAYLOAD | HOLD | ABORT |
| --- | --- | --- | --- | --- |
| GO | 631 | 139 | 4 | 15 |
| GO_REDUCED_PAYLOAD | 274 | 679 | 93 | 333 |
| HOLD | 25 | 98 | 336 | 196 |
| ABORT | 14 | 215 | 85 | 1863 |

### Confusion matrix — B2 / testB

| true\pred | GO | GO_REDUCED_PAYLOAD | HOLD | ABORT |
| --- | --- | --- | --- | --- |
| GO | 617 | 134 | 11 | 3 |
| GO_REDUCED_PAYLOAD | 284 | 685 | 83 | 347 |
| HOLD | 18 | 78 | 348 | 208 |
| ABORT | 15 | 201 | 84 | 1884 |

### Confusion matrix — B2 / testC

| true\pred | GO | GO_REDUCED_PAYLOAD | HOLD | ABORT |
| --- | --- | --- | --- | --- |
| GO | 364 | 94 | 5 | 10 |
| GO_REDUCED_PAYLOAD | 182 | 488 | 65 | 294 |
| HOLD | 9 | 82 | 486 | 301 |
| ABORT | 12 | 158 | 135 | 2315 |

### Confusion matrix — B1 / testA

| true\pred | GO | GO_REDUCED_PAYLOAD | HOLD | ABORT |
| --- | --- | --- | --- | --- |
| GO | 589 | 195 | 1 | 4 |
| GO_REDUCED_PAYLOAD | 238 | 716 | 92 | 333 |
| HOLD | 14 | 76 | 450 | 115 |
| ABORT | 9 | 237 | 92 | 1839 |

### Confusion matrix — B1 / testB

| true\pred | GO | GO_REDUCED_PAYLOAD | HOLD | ABORT |
| --- | --- | --- | --- | --- |
| GO | 568 | 191 | 3 | 3 |
| GO_REDUCED_PAYLOAD | 210 | 744 | 97 | 348 |
| HOLD | 7 | 67 | 455 | 123 |
| ABORT | 14 | 231 | 101 | 1838 |

### Confusion matrix — B1 / testC

| true\pred | GO | GO_REDUCED_PAYLOAD | HOLD | ABORT |
| --- | --- | --- | --- | --- |
| GO | 296 | 168 | 4 | 5 |
| GO_REDUCED_PAYLOAD | 121 | 589 | 67 | 252 |
| HOLD | 5 | 80 | 620 | 173 |
| ABORT | 1 | 186 | 190 | 2243 |

### Confusion matrix — B0 / testA

| true\pred | GO | GO_REDUCED_PAYLOAD | HOLD | ABORT |
| --- | --- | --- | --- | --- |
| GO | 37 | 215 | 178 | 359 |
| GO_REDUCED_PAYLOAD | 0 | 56 | 153 | 1170 |
| HOLD | 0 | 0 | 61 | 594 |
| ABORT | 0 | 8 | 11 | 2158 |

### Confusion matrix — B0 / testB

| true\pred | GO | GO_REDUCED_PAYLOAD | HOLD | ABORT |
| --- | --- | --- | --- | --- |
| GO | 29 | 201 | 184 | 351 |
| GO_REDUCED_PAYLOAD | 1 | 54 | 182 | 1162 |
| HOLD | 0 | 0 | 47 | 605 |
| ABORT | 0 | 10 | 9 | 2165 |

### Confusion matrix — B0 / testC

| true\pred | GO | GO_REDUCED_PAYLOAD | HOLD | ABORT |
| --- | --- | --- | --- | --- |
| GO | 8 | 137 | 120 | 208 |
| GO_REDUCED_PAYLOAD | 0 | 59 | 116 | 854 |
| HOLD | 0 | 1 | 62 | 815 |
| ABORT | 0 | 4 | 12 | 2604 |
