> RoBERTa rows use the corrected comparator (independent sigmoid outputs; 2026-10-02, `results/roberta-correction/`). Rows marked *superseded softmax* are the values first reported, kept for the record. Axis and claim-check rows are unchanged.

### test split, target y_general (n=1059, positives=265)

| System | AUROC [95% CI, country-clustered] | AP | F1 pos / macro @ train-prevalence cut | F1 pos / macro @ train-F1-optimal cut | F1 pos / macro @ oracle cut |
|---|---|---:|---|---|---|
| E_axis | 0.754 [0.714, 0.799] | 0.467 | 0.458 / 0.642 | 0.534 / 0.670 | 0.542 / 0.657 |
| E_axis_plus_nli | 0.756 [0.714, 0.801] | 0.456 | 0.462 / 0.645 | 0.540 / 0.655 | 0.544 / 0.673 |
| E_nli | 0.670 [0.632, 0.706] | 0.380 | 0.384 / 0.593 | 0.468 / 0.550 | 0.479 / 0.577 |
| PE_axis | 0.711 [0.674, 0.750] | 0.410 | 0.408 / 0.609 | 0.505 / 0.615 | 0.505 / 0.613 |
| PE_axis_plus_nli | 0.687 [0.654, 0.717] | 0.396 | 0.388 / 0.596 | 0.480 / 0.628 | 0.493 / 0.609 |
| P_axis | 0.621 [0.589, 0.652] | 0.315 | 0.326 / 0.555 | 0.434 / 0.509 | 0.444 / 0.504 |
| RoBERTa_anti_elitism | 0.571 [0.530, 0.604] | 0.347 | 0.352 / 0.571 | 0.383 / 0.411 | 0.385 / 0.430 |
| RoBERTa_anti_elitism (superseded softmax) | 0.573 [0.536, 0.603] | 0.352 | 0.346 / 0.568 | 0.386 / 0.432 | 0.388 / 0.417 |
| RoBERTa_populist_total | 0.569 [0.533, 0.599] | 0.334 | 0.318 / 0.550 | 0.394 / 0.426 | 0.400 / 0.427 |
| RoBERTa_populist_total (superseded softmax) | 0.569 [0.536, 0.599] | 0.327 | 0.343 / 0.552 | 0.404 / 0.290 | 0.404 / 0.290 |
| XLMT_released_classifier | 0.808 [0.771, 0.842] | 0.632 | 0.610 / 0.742 | 0.559 / 0.720 | 0.614 / 0.742 |

### test split, target y_any (n=1059, positives=416)

| System | AUROC [95% CI, country-clustered] | AP | F1 pos / macro @ train-prevalence cut | F1 pos / macro @ train-F1-optimal cut | F1 pos / macro @ oracle cut |
|---|---|---:|---|---|---|
| E_axis | 0.797 [0.772, 0.822] | 0.709 | 0.674 / 0.735 | 0.673 / 0.706 | 0.687 / 0.739 |
| E_axis_plus_nli | 0.818 [0.790, 0.844] | 0.756 | 0.691 / 0.749 | 0.686 / 0.730 | 0.696 / 0.745 |
| E_nli | 0.789 [0.756, 0.818] | 0.736 | 0.654 / 0.719 | 0.664 / 0.706 | 0.674 / 0.703 |
| PE_axis | 0.748 [0.721, 0.773] | 0.634 | 0.593 / 0.669 | 0.634 / 0.655 | 0.647 / 0.651 |
| PE_axis_plus_nli | 0.784 [0.750, 0.813] | 0.704 | 0.669 / 0.731 | 0.664 / 0.690 | 0.678 / 0.721 |
| P_axis | 0.644 [0.614, 0.672] | 0.490 | 0.507 / 0.599 | 0.588 / 0.550 | 0.591 / 0.580 |
| RoBERTa_anti_elitism | 0.556 [0.521, 0.583] | 0.476 | 0.434 / 0.539 | 0.524 / 0.468 | 0.523 / 0.473 |
| RoBERTa_anti_elitism (superseded softmax) | 0.558 [0.531, 0.582] | 0.475 | 0.428 / 0.535 | 0.514 / 0.473 | 0.522 / 0.473 |
| RoBERTa_populist_total | 0.556 [0.529, 0.583] | 0.466 | 0.421 / 0.529 | 0.526 / 0.481 | 0.532 / 0.482 |
| RoBERTa_populist_total (superseded softmax) | 0.554 [0.527, 0.580] | 0.458 | 0.432 / 0.524 | 0.562 / 0.372 | 0.562 / 0.372 |
| XLMT_released_classifier | 0.694 [0.653, 0.739] | 0.659 | 0.586 / 0.663 | 0.571 / 0.678 | 0.608 / 0.653 |

### all split, target y_general (n=5401, positives=1289)

| System | AUROC [95% CI, country-clustered] | AP | F1 pos / macro @ train-prevalence cut | F1 pos / macro @ train-F1-optimal cut | F1 pos / macro @ oracle cut |
|---|---|---:|---|---|---|
| E_axis | 0.736 [0.717, 0.758] | 0.458 | 0.469 / 0.652 | 0.506 / 0.655 | 0.506 / 0.654 |
| E_axis_plus_nli | 0.737 [0.717, 0.760] | 0.445 | 0.467 / 0.650 | 0.510 / 0.638 | 0.510 / 0.639 |
| E_nli | 0.646 [0.626, 0.669] | 0.345 | 0.350 / 0.574 | 0.436 / 0.528 | 0.436 / 0.538 |
| PE_axis | 0.698 [0.685, 0.713] | 0.414 | 0.419 / 0.619 | 0.473 / 0.595 | 0.473 / 0.596 |
| PE_axis_plus_nli | 0.676 [0.664, 0.687] | 0.394 | 0.404 / 0.609 | 0.458 / 0.616 | 0.458 / 0.613 |
| P_axis | 0.618 [0.606, 0.630] | 0.322 | 0.332 / 0.562 | 0.416 / 0.499 | 0.417 / 0.487 |
| RoBERTa_anti_elitism | 0.573 [0.553, 0.591] | 0.325 | 0.317 / 0.551 | 0.383 / 0.426 | 0.383 / 0.422 |
| RoBERTa_anti_elitism (superseded softmax) | 0.572 [0.553, 0.589] | 0.326 | 0.329 / 0.560 | 0.382 / 0.433 | 0.382 / 0.433 |
| RoBERTa_populist_total | 0.566 [0.549, 0.582] | 0.309 | 0.311 / 0.547 | 0.382 / 0.415 | 0.382 / 0.415 |
| RoBERTa_populist_total (superseded softmax) | 0.564 [0.549, 0.579] | 0.304 | 0.331 / 0.546 | 0.384 / 0.278 | 0.384 / 0.278 |
| XLMT_released_classifier | 0.971 [0.965, 0.977] | 0.949 | 0.906 / 0.938 | 0.903 / 0.937 | 0.908 / 0.939 |

### all split, target y_any (n=5401, positives=2057)

| System | AUROC [95% CI, country-clustered] | AP | F1 pos / macro @ train-prevalence cut | F1 pos / macro @ train-F1-optimal cut | F1 pos / macro @ oracle cut |
|---|---|---:|---|---|---|
| E_axis | 0.772 [0.755, 0.788] | 0.670 | 0.643 / 0.713 | 0.652 / 0.696 | 0.652 / 0.696 |
| E_axis_plus_nli | 0.792 [0.774, 0.809] | 0.717 | 0.663 / 0.729 | 0.668 / 0.720 | 0.669 / 0.727 |
| E_nli | 0.765 [0.747, 0.780] | 0.695 | 0.628 / 0.700 | 0.640 / 0.691 | 0.640 / 0.691 |
| PE_axis | 0.721 [0.705, 0.736] | 0.605 | 0.576 / 0.659 | 0.612 / 0.643 | 0.614 / 0.626 |
| PE_axis_plus_nli | 0.754 [0.737, 0.769] | 0.657 | 0.626 / 0.699 | 0.638 / 0.671 | 0.639 / 0.682 |
| P_axis | 0.621 [0.608, 0.632] | 0.476 | 0.479 / 0.580 | 0.567 / 0.532 | 0.567 / 0.534 |
| RoBERTa_anti_elitism | 0.565 [0.550, 0.579] | 0.473 | 0.434 / 0.543 | 0.523 / 0.484 | 0.523 / 0.484 |
| RoBERTa_anti_elitism (superseded softmax) | 0.564 [0.550, 0.576] | 0.471 | 0.431 / 0.542 | 0.518 / 0.482 | 0.519 / 0.480 |
| RoBERTa_populist_total | 0.562 [0.547, 0.575] | 0.461 | 0.430 / 0.540 | 0.521 / 0.476 | 0.521 / 0.476 |
| RoBERTa_populist_total (superseded softmax) | 0.563 [0.548, 0.577] | 0.456 | 0.442 / 0.535 | 0.544 / 0.358 | 0.544 / 0.358 |
| XLMT_released_classifier | 0.771 [0.744, 0.799] | 0.801 | 0.682 / 0.744 | 0.724 / 0.798 | 0.725 / 0.797 |

### Paired AUROC differences, test split, target y_general

| Comparison | Δ AUROC | 95% CI | p |
|---|---:|---|---:|
| E_axis_plus_nli - E_axis | +0.002 | [-0.007, +0.010] | 0.699 |
| E_axis_plus_nli - PE_axis_plus_nli | +0.068 | [+0.037, +0.104] | 0.000 |
| E_axis_plus_nli - RoBERTa_anti_elitism | +0.185 | [+0.122, +0.257] | 0.000 |
| E_axis_plus_nli - RoBERTa_anti_elitism (superseded softmax) | +0.183 | [+0.128, +0.247] | 0.000 |
| E_axis - RoBERTa_anti_elitism | +0.183 | [+0.122, +0.253] | 0.000 |
| E_axis - RoBERTa_anti_elitism (superseded softmax) | +0.181 | [+0.127, +0.246] | 0.000 |
| E_axis_plus_nli - XLMT_released_classifier | -0.052 | [-0.102, +0.002] | 0.056 |
| RoBERTa_anti_elitism - XLMT_released_classifier | -0.237 | [-0.289, -0.185] | 0.000 |
| RoBERTa_anti_elitism - XLMT_released_classifier (superseded softmax) | -0.235 | [-0.281, -0.189] | 0.000 |
