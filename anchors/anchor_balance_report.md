# Anchor balance report

Gates for anchor channels P, E: |Δ words| ≤ 2, |Δ VADER| ≤ 0.15, |Δ pronoun share| ≤ 0.1. M and nuisance sets are reported without gates.

## alternative-pairs/v3/proposition_anchors.json

Gates passed: **False**

| Group | n | Words (L/R) | VADER (L/R) | Neg (L/R) | they (L/R) | we (L/R) | you (L/R) | I (L/R) | Gate |
|---|---:|---|---|---|---|---|---|---|---|
| P (populist / mainstream) | 10 | 12.7 / 9.6 | +0.12 / +0.10 | 0.012 / 0.000 | 20% / 10% | 20% / 40% | 0% / 0% | 0% / 0% | FAIL: words, pronoun_we |
| E (populist / mainstream) | 10 | 11.2 / 9.7 | +0.06 / +0.08 | 0.066 / 0.022 | 30% / 10% | 10% / 20% | 10% / 0% | 0% / 0% | FAIL: pronoun_they |
| M (populist / mainstream) | 10 | 11.3 / 10.7 | -0.23 / +0.31 | 0.196 / 0.092 | 70% / 20% | 10% / 40% | 10% / 0% | 0% / 30% | – |

Pair flags: 0

## alternative-pairs/v4/pair_anchors.json

Gates passed: **True**

| Group | n | Words (L/R) | VADER (L/R) | Neg (L/R) | they (L/R) | we (L/R) | you (L/R) | I (L/R) | Gate |
|---|---:|---|---|---|---|---|---|---|---|
| P (populist / matched_mainstream) | 40 | 12.6 / 11.8 | +0.10 / +0.15 | 0.024 / 0.008 | 22% / 22% | 28% / 28% | 0% / 0% | 5% / 5% | pass |
| E (populist / matched_mainstream) | 40 | 11.9 / 12.1 | +0.10 / -0.01 | 0.092 / 0.069 | 45% / 45% | 15% / 15% | 25% / 25% | 0% / 0% | pass |
| M (populist / matched_mainstream) | 40 | 10.7 / 10.7 | -0.38 / +0.08 | 0.271 / 0.068 | 70% / 68% | 20% / 25% | 5% / 0% | 2% / 2% | – |

Pair flags: 30
- P P03: vader +0.13 vs -0.30
- P P04: vader +0.44 vs +0.00
- P P06: vader -0.25 vs +0.18
- P P11: vader +0.48 vs +0.00
- P P12: vader +0.00 vs +0.44
- P P19: vader +0.00 vs +0.40
- P P25: vader +0.00 vs +0.40
- P P33: vader +0.00 vs +0.59
- P P39: vader +0.30 vs +0.85
- E E02: vader -0.68 vs +0.00
- E E07: vader +0.00 vs +0.40
- E E09: vader +0.40 vs -0.40
- E E13: vader +0.56 vs +0.00
- E E15: vader +0.42 vs +0.00
- E E16: vader +0.00 vs -0.56
- E E17: vader +0.40 vs -0.13
- E E19: vader +0.10 vs -0.32
- E E21: vader +0.78 vs +0.20
- E E24: vader +0.37 vs -0.23
- E E25: vader +0.44 vs +0.00
- E E26: vader +0.48 vs +0.00
- E E27: vader +0.20 vs -0.23
- E E30: vader -0.18 vs +0.56
- E E35: vader +0.72 vs -0.17
- E E39: vader +0.40 vs +0.00
- M M08: pronoun we
- M M19: pronoun we
- M M22: pronoun you
- M M33: pronoun they
- M M36: pronoun you

## alternative-pairs/v4/nuisance_sentences.json

Gates passed: **True**

| Group | n | Words (L/R) | VADER (L/R) | Neg (L/R) | they (L/R) | we (L/R) | you (L/R) | I (L/R) | Gate |
|---|---:|---|---|---|---|---|---|---|---|
| hostility_negative_affect (nuisance / neutral) | 10 | 12.8 / 12.8 | -0.83 / -0.07 | 0.489 / 0.057 | 0% / 0% | 30% / 40% | 0% / 0% | 60% / 50% | – |
| partisan_attack_on_opposition (nuisance / neutral) | 10 | 11.6 / 11.5 | -0.35 / +0.05 | 0.272 / 0.046 | 0% / 10% | 30% / 50% | 0% / 0% | 0% / 0% | – |
| foreign_enemy (nuisance / neutral) | 10 | 11.5 / 11.0 | -0.56 / +0.01 | 0.332 / 0.021 | 40% / 30% | 90% / 90% | 0% / 0% | 0% / 0% | – |
| generic_grievance (nuisance / neutral) | 10 | 11.0 / 10.8 | -0.32 / +0.14 | 0.249 / 0.000 | 30% / 30% | 30% / 30% | 0% / 0% | 0% / 0% | – |
| incivility (nuisance / neutral) | 10 | 11.4 / 11.2 | -0.66 / +0.06 | 0.386 / 0.017 | 10% / 10% | 0% / 0% | 10% / 10% | 20% / 20% | – |

Pair flags: 6
- hostility_negative_affect hos06: pronoun we
- hostility_negative_affect hos09: pronoun i
- partisan_attack_on_opposition par02: pronoun we
- partisan_attack_on_opposition par05: pronoun we
- partisan_attack_on_opposition par08: pronoun they
- foreign_enemy for10: pronoun they
