# Baseline vs. Protocol Defense

Both runs used a sequence length of 1,000,000 states.

| Metric | Baseline | Protocol defense | Difference |
|---|---:|---:|---:|
| Alice packets sent | 929,998 | 550,039 | -379,959 (-40.9%) |
| Bob public `Y` accuracy | 95.25% | 97.62% | +2.37 percentage points |
| Bob private `S` accuracy | 96.58% | 98.55% | +1.97 percentage points |
| Bob joint `(Y, S)` accuracy | 93.53% | 96.89% | +3.36 percentage points |
| Eve `S` accuracy | 88.10% | 68.13% | -19.97 percentage points |
| Average CRA | 9.84% | 31.22% | +21.38 percentage points (about 3.17x) |
| Simulation time | 11.98 s | 10.24 s | -1.74 s |

## Interpretation

The defense sent about 40.9% fewer packets. Bob's public-task accuracy and
joint accuracy increased, while Eve's accuracy on the sensitive attribute `S`
decreased. CRA increased from 9.84% to 31.22%.

In this simulator, CRA is the fraction of time steps where Bob reconstructs
both `Y` and `S` correctly while Eve gets `S` wrong.
