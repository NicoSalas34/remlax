| case | n | software (backend) | wall s | solver s | evals | s / eval | logLik | agreement with asreml |
|---|---|---|---|---|---|---|---|---|
| iid | 500 | remlax (gpu, a100_n500_2000) | 11.152 | 0.063 | 24 | 0.0026 | -370.943512 | dlogLik +0.00e+00, comp rel 2.1e-06 |
| iid | 500 | remlax (cpu, cpu4_n500_2000) | 9.24 | 0.596 | 27 | 0.0221 | -370.943512 | dlogLik +0.00e+00, comp rel 2.1e-06 |
| iid | 500 | asreml (cpu, cpu4_n500_2000) | 4.144 |  | 6 | 0.6907 | -370.943512 |  |
| iid | 500 | lme4 (cpu, cpu4_n500_2000) | 0.072 |  | 15 | 0.0048 | -828.574901 | comp rel 2.1e-06 |
| iid | 500 | sommer (cpu, cpu4_n500_2000) | 0.555 |  | 6 | 0.0925 | -118.571827 | comp rel 2.0e-05 |
| grm | 500 | remlax (gpu, a100_n500_2000) | 8.186 | 0.15 | 67 | 0.0022 | -352.830777 | dlogLik +0.00e+00, comp rel 2.7e-05 |
| grm | 500 | remlax (cpu, cpu4_n500_2000) | 28.989 | 23.084 | 56 | 0.4122 | -352.830777 | dlogLik +0.00e+00, comp rel 2.7e-05 |
| grm | 500 | asreml (cpu, cpu4_n500_2000) | 4.03 |  | 6 | 0.6717 | -352.830777 |  |
| grm | 500 | lme4 | skipped | | | | | model not expressible or size above cap |
| grm | 500 | sommer (cpu, cpu4_n500_2000) | 0.377 |  | 5 | 0.0754 | -105.192225 | comp rel 2.3e-05 |
| us3 | 498 | remlax (gpu, a100_n500_2000) | 8.569 | 0.206 | 89 | 0.0023 | -449.715263 | dlogLik +0.00e+00, comp rel 4.0e-01 |
| us3 | 498 | remlax (cpu, cpu4_n500_2000) | 7.67 | 2.241 | 89 | 0.0252 | -449.715263 | dlogLik +0.00e+00, comp rel 4.0e-01 |
| us3 | 498 | asreml (cpu, cpu4_n500_2000) | 3.966 |  | 17 | 0.2333 | -449.715263 |  |
| us3 | 498 | lme4 | skipped | | | | | model not expressible or size above cap |
| us3 | 498 | sommer (cpu, cpu4_n500_2000) | 0.765 |  | 7 | 0.1093 | -199.945132 | comp rel 4.0e-01 |
| us6 | 498 | remlax (gpu, a100_n500_2000) | 8.919 | 0.74 | 330 | 0.0022 | -424.600273 | dlogLik +7.00e-06, comp rel 8.2e-01 |
| us6 | 498 | remlax (cpu, cpu4_n500_2000) | 14.15 | 10.016 | 331 | 0.0303 | -424.600273 | dlogLik +7.00e-06, comp rel 8.2e-01 |
| us6 | 498 | asreml (cpu, cpu4_n500_2000) | 3.948 |  | 47 | 0.084 | -424.600280 |  |
| us6 | 498 | lme4 | skipped | | | | | model not expressible or size above cap |
| us6 | 498 | sommer (cpu, cpu4_n500_2000) | 2.684 |  | 7 | 0.3834 | -156.744337 | comp rel 8.2e-01 |
| ar1ar1 | 500 | remlax (gpu, a100_n500_2000) | 7.779 | 0.098 | 39 | 0.0025 | -320.171305 | dlogLik +2.80e-05, comp rel 3.7e-04 |
| ar1ar1 | 500 | remlax (cpu, cpu4_n500_2000) | 5.429 | 0.834 | 37 | 0.0225 | -320.171305 | dlogLik +2.80e-05, comp rel 3.7e-04 |
| ar1ar1 | 500 | asreml (cpu, cpu4_n500_2000) | 3.793 |  | 8 | 0.4741 | -320.171333 |  |
| ar1ar1 | 500 | lme4 | skipped | | | | | model not expressible or size above cap |
| ar1ar1 | 500 | sommer | skipped | | | | | model not expressible or size above cap |
| iid | 2000 | remlax (gpu, a100_n500_2000) | 7.628 | 0.41 | 24 | 0.0171 | -1464.374797 | dlogLik +0.00e+00, comp rel 2.0e-06 |
| iid | 2000 | remlax (cpu, cpu4_n500_2000) | 105.789 | 90.964 | 24 | 3.7902 | -1464.374797 | dlogLik +0.00e+00, comp rel 2.0e-06 |
| iid | 2000 | asreml (cpu, cpu4_n500_2000) | 3.83 |  | 6 | 0.6383 | -1464.374797 |  |
| iid | 2000 | lme4 (cpu, cpu4_n500_2000) | 0.023 |  | 14 | 0.0016 | -3300.413986 | comp rel 2.0e-06 |
| iid | 2000 | sommer (cpu, cpu4_n500_2000) | 6.583 |  | 6 | 1.0972 | -489.098703 | comp rel 1.4e-05 |
| grm | 2000 | remlax (gpu, a100_n500_2000) | 13.331 | 0.744 | 44 | 0.0169 | -1424.845760 | dlogLik +0.00e+00, comp rel 1.0e-05 |
| grm | 2000 | remlax (cpu, cpu4_n500_2000) | 36.789 | 24.387 | 32 | 0.7621 | -1424.845760 | dlogLik +0.00e+00, comp rel 1.0e-05 |
| grm | 2000 | asreml (cpu, cpu4_n500_2000) | 4.105 |  | 6 | 0.6842 | -1424.845760 |  |
| grm | 2000 | lme4 | skipped | | | | | model not expressible or size above cap |
| grm | 2000 | sommer (cpu, cpu4_n500_2000) | 6.489 |  | 6 | 1.0815 | -477.492444 | comp rel 1.5e-05 |
| us3 | 1998 | remlax (gpu, a100_n500_2000) | 10.13 | 1.36 | 89 | 0.0153 | -1816.965187 | dlogLik +0.00e+00, comp rel 8.0e-01 |
| us3 | 1998 | remlax (cpu, cpu4_n500_2000) | 102.53 | 96.536 | 89 | 1.0847 | -1816.965187 | dlogLik +0.00e+00, comp rel 8.0e-01 |
| us3 | 1998 | asreml (cpu, cpu4_n500_2000) | 3.337 |  | 17 | 0.1963 | -1816.965187 |  |
| us3 | 1998 | lme4 | skipped | | | | | model not expressible or size above cap |
| us3 | 1998 | sommer (cpu, cpu4_n500_2000) | 23.492 |  | 7 | 3.356 | -796.524607 | comp rel 8.0e-01 |
| us6 | 1998 | remlax (gpu, a100_n500_2000) | 12.969 | 4.402 | 329 | 0.0134 | -1800.628675 | dlogLik +0.00e+00, comp rel 1.2e+00 |
| us6 | 1998 | remlax (cpu, cpu4_n500_2000) | 1395.809 | 1383.06 | 328 | 4.2166 | -1800.628675 | dlogLik +0.00e+00, comp rel 1.2e+00 |
| us6 | 1998 | asreml (cpu, cpu4_n500_2000) | 3.626 |  | 47 | 0.0771 | -1800.628675 |  |
| us6 | 1998 | lme4 | skipped | | | | | model not expressible or size above cap |
| us6 | 1998 | sommer (cpu, cpu4_n500_2000) | 83.999 |  | 7 | 11.9999 | -731.730493 | comp rel 1.2e+00 |
| ar1ar1 | 2000 | remlax (gpu, a100_n500_2000) | 8.307 | 0.645 | 40 | 0.0161 | -1314.742565 | dlogLik +3.00e-06, comp rel 5.4e-05 |
| ar1ar1 | 2000 | remlax (cpu, cpu4_n500_2000) | 278.694 | 262.459 | 41 | 6.4014 | -1314.742565 | dlogLik +3.00e-06, comp rel 5.4e-05 |
| ar1ar1 | 2000 | asreml (cpu, cpu4_n500_2000) | 4.354 |  | 8 | 0.5442 | -1314.742568 |  |
| ar1ar1 | 2000 | lme4 | skipped | | | | | model not expressible or size above cap |
| ar1ar1 | 2000 | sommer | skipped | | | | | model not expressible or size above cap |
| iid | 8000 | remlax (gpu, a100_n8000) | 29.698 | 10.718 | 34 | 0.3152 | -6016.871716 |  |
| grm | 8000 | remlax (gpu, a100_n8000) | 332.784 | 12.8 | 41 | 0.3122 | -5717.685293 |  |
| ar1ar1 | 8000 | remlax (gpu, a100_n8000) | 28.062 | 11.88 | 38 | 0.3126 | -5536.860751 |  |
| us3 | 7998 | remlax (gpu, a100_n8000) | 78.109 | 52.29 | 164 | 0.3188 | -5490.831968 |  |
| us6 | 7998 | remlax (gpu, a100_n8000) | 68.748 | 45.546 | 159 | 0.2865 | -6426.591138 |  |
