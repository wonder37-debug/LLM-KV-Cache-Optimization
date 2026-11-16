### Capacity sweep (legacy_round_robin, seed 20260101)

| Cell | Decisions | p50 full | p50 delta | p90 full | p90 delta | p95 full | p95 delta | max full | max delta | Total full | Total delta | Speed-up | Saving / delta-cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy_round_robin-cap128-s20260101 | 40 | 6.48 | 2.05 | 22.7 | 4.5 | 23.0 | 4.5 | 25.1 | 4.6 | 459 | 102 | 4.5x | 0.0x |
| legacy_round_robin-cap160-s20260101 | 40 | 8.61 | 2.27 | 31.6 | 5.7 | 32.4 | 5.9 | 35.2 | 6.3 | 597 | 121 | 4.9x | 9.2x |
| legacy_round_robin-cap224-s20260101 | 40 | 8.59 | 2.35 | 55.3 | 7.9 | 55.9 | 8.2 | 58.9 | 12.4 | 727 | 141 | 5.2x | 14.7x |
| legacy_round_robin-cap240-s20260101 | 40 | 8.37 | 2.18 | 60.6 | 8.4 | 61.2 | 8.5 | 69.2 | 8.5 | 766 | 138 | 5.6x | 21.2x |
| legacy_round_robin-cap256-s20260101 | 40 | 9.59 | 2.29 | 65.1 | 9.0 | 72.6 | 9.1 | 73.2 | 9.8 | 843 | 145 | 5.8x | 30.4x |
| legacy_round_robin-cap264-s20260101 | 40 | 9.61 | 2.31 | 50.0 | 7.6 | 50.3 | 7.7 | 52.8 | 7.8 | 673 | 133 | 5.1x | 51.8x |
| legacy_round_robin-cap272-s20260101 | 40 | 9.00 | 2.27 | 35.4 | 6.1 | 35.5 | 6.1 | 35.6 | 6.2 | 512 | 114 | 4.5x | 110.1x |
| legacy_round_robin-cap280-s20260101 | 40 | 7.81 | 2.20 | 22.1 | 4.3 | 22.7 | 4.3 | 23.6 | 4.4 | 389 | 95 | 4.1x | 195.6x |
| legacy_round_robin-cap288-s20260101 | 40 | 5.76 | 2.12 | 11.8 | 2.7 | 11.9 | 2.7 | 12.0 | 2.8 | 246 | 79 | 3.1x | 301.0x |
| legacy_round_robin-cap292-s20260101 | 40 | 3.39 | 1.71 | 12.1 | 2.6 | 12.1 | 2.7 | 12.5 | 2.8 | 200 | 69 | 2.9x | 395.3x |
| legacy_round_robin-cap296-s20260101 | 24 | 5.12 | 2.01 | 12.2 | 2.8 | 12.5 | 3.0 | 12.9 | 3.5 | 163 | 48 | 3.4x | 0.0x |
| legacy_round_robin-cap320-s20260101 | 24 | 6.06 | 2.41 | 13.1 | 2.9 | 13.1 | 3.0 | 14.8 | 3.0 | 181 | 53 | 3.4x | 0.0x |

Median per-decision speed-up (p50): **3.61x**; median per-cell total speed-up: **4.49x**.
