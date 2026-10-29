# Campaign summary

Updated 2026-10-23 14:05. Job table: status.json; logs: logs/.

## A1_pbmc_fast_independent: done (0.2 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| palm_fast | 0.071 | 0.542 | 0.439 | 0.103 | 0.802 | 0.373 | 0.933 | 0.025 | 0.690 | 237.704 |

## A2_pbmc_fast_shared: done (0.2 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| palm_fast | 0.069 | 0.518 | 0.525 | 0.066 | 0.756 | 0.383 | 0.981 | 0.027 | 0.512 | 238.106 |

## A3_pbmc_fast_shared_only: done (0.4 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| palm_fast | 0.074 | 0.419 | 0.562 | 0.068 | 0.491 | 0.056 | 0.943 | 0.030 | 0.548 | 415.318 |

## B1_pbmc5_competitors: done (8.3 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_two_step | 0.115 |  | 0.773 |  |  | 0.325 | 0.876 | 0.037 |  | 0.136 |
| reallocation | 0.115 |  | 0.787 | 0.300 |  | 0.468 | 0.785 | 0.042 |  | 1.773 |
| reallocation+ulm | 0.115 |  | 0.437 | -0.024 |  | 0.182 | 0.791 |  |  | 1.773 |
| toast | 0.115 |  |  |  |  | 0.202 | 0.785 | 0.115 |  | 32.250 |
| tca | 0.115 |  |  |  |  | -0.080 | 0.772 | 0.137 |  | 33.632 |
| bayesprism | 0.178 |  | 0.658 | 0.308 |  | 0.383 | 0.885 | 0.034 |  | 5915.976 |
| bayesprism+ulm | 0.178 |  | 0.307 | -0.024 |  | 0.235 | 0.889 |  |  | 5915.976 |

## B2_pbmc5_fast_shared: done (0.3 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| palm_fast | 0.072 | 0.499 | 0.509 | 0.105 | 0.761 | 0.536 | 0.952 | 0.031 | 0.571 | 238.237 |

## B3_pbmc5_fast_independent: done (0.5 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| palm_fast | 0.073 | 0.517 | 0.434 | 0.136 | 0.810 | 0.547 | 0.928 | 0.026 | 0.714 | 369.134 |

## B4_pbmc5_palm_independent: done (7.9 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| palm | 0.065 | 0.679 | 0.461 | 0.155 | 0.865 | 0.596 | 0.976 | 0.026 | 0.707 | 5694.226 |

## B5_pbmc5_palm_shared: done (8.2 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| palm | 0.065 | 0.622 | 0.554 | 0.167 | 0.770 | 0.368 | 0.925 | 0.031 | 0.479 | 5870.059 |

## D1_onek1k_broad: done (2.7 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_two_step | 0.068 |  | 0.648 |  |  |  |  |  |  | 0.027 |
| reallocation | 0.068 |  | 0.673 | 0.221 |  |  |  |  |  | 0.034 |
| reallocation+ulm | 0.068 |  | 0.378 | 0.078 |  |  |  |  |  | 0.034 |
| bayesprism | 0.070 |  | 0.734 | 0.310 |  |  |  |  |  | 910.900 |
| bayesprism+ulm | 0.070 |  | 0.403 | 0.074 |  |  |  |  |  | 910.900 |
| palm_fast | 0.056 | 0.552 | 0.397 | 0.091 | 0.694 |  |  |  |  | 127.109 |
| palm | 0.057 | 0.651 | 0.422 | 0.049 | 0.730 |  |  |  |  | 2137.058 |

## D2_onek1k_broad_sex: done (0.2 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_two_step | 0.068 |  | 0.648 |  |  | 0.256 |  | 0.015 |  | 0.031 |
| reallocation | 0.068 |  | 0.673 | 0.221 |  | 0.462 |  | 0.014 |  | 0.049 |
| reallocation+ulm | 0.068 |  | 0.378 | 0.078 |  | 0.126 |  |  |  | 0.049 |
| toast | 0.068 |  |  |  |  | 0.049 |  | 0.076 |  | 17.430 |
| tca | 0.068 |  |  |  |  | 0.174 |  | 0.279 |  | 16.225 |
| palm_fast | 0.055 | 0.562 | 0.378 | 0.087 | 0.669 | 0.154 |  | 0.021 | 0.750 | 232.159 |

## D3_onek1k_5types: done (0.3 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_two_step | 0.066 |  | 0.672 |  |  |  |  |  |  | 0.026 |
| reallocation | 0.066 |  | 0.713 | 0.251 |  |  |  |  |  | 0.053 |
| reallocation+ulm | 0.066 |  | 0.375 | 0.046 |  |  |  |  |  | 0.053 |
| palm_fast | 0.035 | 0.650 | 0.353 | 0.099 | 0.762 |  |  |  |  | 340.652 |

## C1_sim_N24: done (5.0 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_two_step | 0.043 |  | 0.391 |  |  | 0.057 | 0.500 | 0.598 |  | 0.017 |
| reallocation | 0.043 |  | 0.491 | 0.592 |  | 0.498 | 0.787 | 0.502 |  | 0.059 |
| reallocation+ulm | 0.043 |  | 0.413 | -0.044 |  | 0.060 | 0.548 |  |  | 0.059 |
| toast | 0.043 |  |  |  |  | 0.494 | 0.690 | 0.514 |  | 23.204 |
| tca | 0.043 |  |  |  |  | 0.585 | 0.790 | 0.718 |  | 14.519 |
| bayesprism | 0.027 |  | 0.488 | 0.589 |  | 0.548 | 0.804 | 0.492 |  | 837.751 |
| bayesprism+ulm | 0.027 |  | 0.414 | -0.057 |  | 0.058 | 0.525 |  |  | 837.751 |
| palm | 0.013 | 0.908 | 0.693 | 0.754 | 0.890 | 0.984 | 1.000 | 0.102 | 0.890 | 2218.702 |
| palm_fast | 0.014 | 0.785 | 0.708 | 0.764 | 0.794 | 0.983 | 1.000 | 0.088 | 0.810 | 498.804 |

## C2_sim_N48: done (8.2 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_two_step | 0.059 |  | 0.390 |  |  | 0.019 | 0.500 | 0.544 |  | 0.014 |
| reallocation | 0.059 |  | 0.469 | 0.543 |  | 0.574 | 0.840 | 0.448 |  | 0.106 |
| reallocation+ulm | 0.059 |  | 0.401 | 0.092 |  | -0.012 | 0.498 |  |  | 0.106 |
| toast | 0.059 |  |  |  |  | 0.620 | 0.842 | 0.393 |  | 19.031 |
| tca | 0.059 |  |  |  |  | 0.575 | 0.860 | 0.500 |  | 11.937 |
| bayesprism | 0.024 |  | 0.474 | 0.558 |  | 0.611 | 0.848 | 0.443 |  | 1085.782 |
| bayesprism+ulm | 0.024 |  | 0.402 | 0.088 |  | 0.048 | 0.527 |  |  | 1085.782 |
| palm | 0.013 | 0.903 | 0.712 | 0.756 | 0.869 | 0.991 | 1.000 | 0.057 | 0.880 | 3556.048 |
| palm_fast | 0.013 | 0.832 | 0.714 | 0.755 | 0.796 | 0.990 | 1.000 | 0.058 | 0.800 | 1180.954 |

## C3_sim_N96: done (14.8 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_two_step | 0.051 |  | 0.394 |  |  | 0.059 | 0.500 | 0.592 |  | 0.028 |
| reallocation | 0.051 |  | 0.483 | 0.581 |  | 0.583 | 0.821 | 0.491 |  | 0.175 |
| reallocation+ulm | 0.051 |  | 0.401 | 0.015 |  | 0.075 | 0.525 |  |  | 0.175 |
| palm | 0.012 | 0.910 | 0.709 | 0.787 | 0.865 | 0.989 | 1.000 | 0.066 | 0.830 | 8281.798 |
| palm_fast | 0.012 | 0.852 | 0.714 | 0.800 | 0.817 | 0.991 | 1.000 | 0.060 | 0.760 | 2358.895 |

## C4_sim_N48_shared: done (1.7 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| palm_fast | 0.013 | 0.831 | 0.622 | 0.709 | 0.670 | 0.991 | 1.000 | 0.058 | 0.800 | 1239.140 |

## E1_pbmc_fast_genes500: done (0.7 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| palm_fast | 0.068 | 0.662 | 0.415 | 0.029 | 0.790 | 0.374 | 0.453 | 0.030 | 0.786 | 781.024 |

## E2_pbmc_fast_genes2000: done (0.4 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| palm_fast | 0.084 | 0.386 | 0.400 | 0.189 | 0.782 | 0.619 | 1.000 | 0.020 | 0.750 | 465.281 |

## E3_pbmc_fast_all_programs: done (2.2 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_two_step | 0.122 |  | 0.734 |  |  | 0.339 | 0.541 | 0.046 |  | 0.187 |
| palm_fast | 0.075 | 0.535 | 0.402 | 0.142 | 0.779 | 0.076 | 0.391 | 0.045 | 0.476 | 2575.789 |

## F1_onek1k_full_broad: done (20.4 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_two_step | 0.066 |  | 0.676 |  |  |  |  |  |  | 0.125 |
| reallocation | 0.066 |  | 0.710 | 0.307 |  |  |  |  |  | 0.596 |
| reallocation+ulm | 0.066 |  | 0.341 | 0.062 |  |  |  |  |  | 0.596 |
| bayesprism | 0.065 |  | 0.785 | 0.399 |  |  |  |  |  | 19263.190 |
| bayesprism+ulm | 0.065 |  | 0.373 | 0.050 |  |  |  |  |  | 19263.190 |
| palm_fast | 0.076 | 0.377 | 0.435 | 0.017 | 0.632 |  |  |  |  | 5143.310 |

## F2_onek1k_full_sex: done (11.7 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_two_step | 0.066 |  | 0.686 |  |  | 0.241 |  | 0.005 |  | 0.198 |
| reallocation | 0.066 |  | 0.725 | 0.307 |  | 0.609 |  | 0.004 |  | 0.946 |
| reallocation+ulm | 0.066 |  | 0.350 | 0.056 |  | 0.117 |  |  |  | 0.946 |
| toast | 0.066 |  |  |  |  | 0.152 |  | 0.056 |  | 25.028 |
| tca | 0.066 |  |  |  |  | -0.197 |  | 0.024 |  | 22.027 |
| palm_fast | 0.086 | 0.306 | 0.430 | 0.022 | 0.644 | 0.107 |  | 0.005 | 0.690 | 14001.500 |

## F3_onek1k_full_5types: done (11.1 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| naive_two_step | 0.074 |  | 0.697 |  |  |  |  |  |  | 0.135 |
| reallocation | 0.074 |  | 0.710 | 0.222 |  |  |  |  |  | 0.688 |
| reallocation+ulm | 0.074 |  | 0.350 | 0.064 |  |  |  |  |  | 0.688 |
| palm_fast | 0.051 | 0.588 | 0.400 | 0.032 | 0.676 |  |  |  |  | 13283.528 |

## F4_onek1k_full_palm: done (16.2 h)

| method | frac_rmse | frac_cov90 | act_spearman_across_samples | act_spearman_across_types | act_cov90 | eff_pearson | eff_auroc | eff_mae | eff_cov90 | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| palm | 0.078 | 0.420 | 0.472 | 0.036 | 0.626 | 0.236 |  | 0.007 | 0.518 | 29124.333 |
