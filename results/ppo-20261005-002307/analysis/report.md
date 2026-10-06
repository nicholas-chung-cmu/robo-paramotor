# Run report: ppo-20261005-002307

## Training (mean of the last 5% of updates)

- updates: 192, env steps: 25.17 M, throughput: 357 steps/s
- curriculum difficulty: 0.2 (1.0 = full)
- reward/step: 0.060, finished-episode return: 92.3
- cross-track: 6.21 m, altitude error: 0.54 m, outside alpha envelope: 0.3%
- approx KL: 0.0083, entropy: -1.323

## Last curriculum gate

| metric | value | threshold | pass |
| --- | --- | --- | --- |
| Cross-track error (m) | 6.62 | below 3 | no |
| Altitude error (m) | 0.631 | below 2 | yes |
| Time outside alpha envelope | 0.00178 | below 0.05 | yes |
| Failure rate | 0 | below 0.1 | yes |
| Progress along route (m/s) | 5.35 | above 2 | yes |
| Flights finishing the route | 1 | above 0.75 | yes |

Gates passed: 2 of 7.
