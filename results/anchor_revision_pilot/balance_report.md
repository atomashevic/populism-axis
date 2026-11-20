# Anchor balance report

Gates for anchor channels P, E: |Δ words| ≤ 2, |Δ VADER| ≤ 0.15, |Δ pronoun share| ≤ 0.1. M and nuisance sets are reported without gates.

## anchors/candidates/v5_green.json

Gates passed: **False**

| Group | n | Words (L/R) | VADER (L/R) | Neg (L/R) | they (L/R) | we (L/R) | you (L/R) | I (L/R) | Gate |
|---|---:|---|---|---|---|---|---|---|---|
| P (populist / mainstream) | 10 | 12.1 / 9.6 | +0.20 / +0.10 | 0.012 / 0.000 | 20% / 10% | 30% / 40% | 0% / 0% | 0% / 0% | FAIL: words, pronoun_we |
| E (populist / mainstream) | 10 | 11.4 / 9.8 | +0.08 / +0.08 | 0.066 / 0.022 | 20% / 10% | 10% / 10% | 10% / 0% | 0% / 0% | pass |
| M (populist / mainstream) | 10 | 11.3 / 10.7 | -0.23 / +0.31 | 0.196 / 0.092 | 70% / 20% | 10% / 40% | 10% / 0% | 0% / 30% | – |

Pair flags: 0

## anchors/candidates/v5_red.json

Gates passed: **False**

| Group | n | Words (L/R) | VADER (L/R) | Neg (L/R) | they (L/R) | we (L/R) | you (L/R) | I (L/R) | Gate |
|---|---:|---|---|---|---|---|---|---|---|
| P (populist / mainstream) | 10 | 12.7 / 9.5 | +0.12 / +0.08 | 0.012 / 0.000 | 20% / 10% | 20% / 40% | 0% / 0% | 0% / 0% | FAIL: words, pronoun_we |
| E (populist / mainstream) | 10 | 11.2 / 10.7 | +0.06 / +0.11 | 0.066 / 0.028 | 30% / 10% | 10% / 20% | 10% / 0% | 0% / 0% | FAIL: pronoun_they |
| M (populist / mainstream) | 10 | 11.3 / 10.7 | -0.23 / +0.31 | 0.196 / 0.092 | 70% / 20% | 10% / 40% | 10% / 0% | 0% / 30% | – |

Pair flags: 0

## anchors/candidates/v5_green_red.json

Gates passed: **False**

| Group | n | Words (L/R) | VADER (L/R) | Neg (L/R) | they (L/R) | we (L/R) | you (L/R) | I (L/R) | Gate |
|---|---:|---|---|---|---|---|---|---|---|
| P (populist / mainstream) | 10 | 12.1 / 9.5 | +0.20 / +0.08 | 0.012 / 0.000 | 20% / 10% | 30% / 40% | 0% / 0% | 0% / 0% | FAIL: words, pronoun_we |
| E (populist / mainstream) | 10 | 11.4 / 10.8 | +0.08 / +0.11 | 0.066 / 0.028 | 20% / 10% | 10% / 10% | 10% / 0% | 0% / 0% | pass |
| M (populist / mainstream) | 10 | 11.3 / 10.7 | -0.23 / +0.31 | 0.196 / 0.092 | 70% / 20% | 10% / 40% | 10% / 0% | 0% / 30% | – |

Pair flags: 0
