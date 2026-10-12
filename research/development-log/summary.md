# Summary: stepwise improvement of the proposition-anchor axis

Date: 2026-09-24. Branch `feature/v2-v3-benchmark-pipeline`. Protocol: `docs/development-protocol.md`. Step reports: `research/development-log/stage-*.md`.

> **Correction, 2026-10-02.** The RoBERTa comparator was originally scored with a softmax across its three labels; the checkpoint is a multi-label model with independent sigmoid outputs. The rows marked *sigmoid* are the corrected comparator (run `results/roberta-correction/rc-20261002T1415Z-sigmoid/`, same model, windows, pooling, splits and seeds); rows marked *softmax* are kept as the superseded record. Axis and NLI numbers are unchanged. The corrected comparator beats axis + NLI on EU-OOD macro-F1 (.555 against .545).

## Result

V4 improves on V3 on both held-out sets, but only when the axis is combined with an NLI second stage. The axis alone improves slightly and not significantly.

- **English holdout** (84 GPD speeches from 14 countries, never used for a V4 decision): axis + NLI reaches Spearman .518, against .376 for the V3 configuration (paired Δ +.142 [+.004, +.332], leader-clustered). The axis alone reaches .417 (Δ +.041 [−.044, +.129]).
- **EU-OOD** (1,762 translated sentences, scored once): axis + NLI reaches 3-vs-0 AUROC .942 against .890, and macro-F1 .545 under the Chalkidis et al. (2025) protocol. V3 scored .528, the fine-tuned RoBERTa .555 (corrected sigmoid; .512 under the superseded softmax), and the paper's zero-shot LLMs .590 (Gemini 2.5 Flash) and .613 (Qwen3-14B).

## Step ledger (development data: English-dev 174 speeches, Agora 5,272 paragraphs)

| Step | Change | English-dev ρ | Agora ρ | Decision |
|---|---|---:|---:|---|
| – | V3 configuration on the development split | .462 | .357 | reference |
| 1 | 60-word window cap (splitter defect) | .483 | .359 | adopted (defect fix) |
| 1b | Remove metadata headers, URL lines and bundled Chinese text | .483 | .359 | adopted (data fix, no effect) |
| 2 | Label-free whitening and nuisance residualization | ≤ .488 | ≤ .362 | rejected: all 20 cells fail |
| 3 | Minimal-pair anchors, balanced for length, pronouns and sentiment | .444 | .347 | rejected |
| 4 | Three-sentence windows, mean of top 8 | .517 | .359 | adopted |
| 5 | Encoders: Qwen3-Embedding 0.6B/4B/8B, multilingual-e5-instruct | ≤ .515 | ≤ .392 | rejected: none passes on English |
| 6 | NLI second stage (BART-large-MNLI) combined with the axis | .537 | .382 | adopted |

## Held-out results

| Configuration | English holdout ρ [95% CI] | Extreme AUROC | Extreme AP | Leader-term ρ | EU AUROC 3-vs-0 | EU within-speaker AUROC | EU macro-F1 (LOSO) |
|---|---|---:|---:|---:|---:|---:|---:|
| V3 configuration (B0) | .376 [.121, .598] | .856 | .524 | .289 | .890 | .861 | .528 |
| V4 axis only (B4) | .417 [.162, .618] | .936 | .508 | .251 | .890 | .860 | .533 |
| **V4 axis + NLI (B6)** | **.518 [.283, .700]** | **.984** | **.804** | .342 | **.942** | **.927** | **.545** |
| NLI only (diagnostic) | .520 [.309, .699] | 1.000 | 1.000 | .458 | .945 | .929 | – |
| RoBERTa fine-tuned (comparator), sigmoid | .440 [.142, .678] | .883 | .620 | .060 | .900 | .854 | .555 |
| RoBERTa fine-tuned, softmax (superseded) | .464 [.238, .670] | .856 | .484 | .147 | .902 | .862 | .512 |

References: the English-holdout human-coder Spearman is .748 (one coder pair) and word count alone reaches .261; speaker identity alone reaches EU AUROC .837.

Paired differences on the English holdout (27 leaders, 4,000 draws):

| Comparison | Δρ | 95% CI | p |
|---|---:|---|---:|
| V4 axis − V3 configuration | +.041 | [−.044, +.129] | .34 |
| **V4 axis + NLI − V3 configuration** | **+.142** | **[+.004, +.332]** | **.04** |
| V4 axis + NLI − V4 axis | +.101 | [+.009, +.241] | .02 |
| V4 axis + NLI − RoBERTa (sigmoid) | +.079 | [−.080, +.251] | .35 |
| V4 axis − RoBERTa (sigmoid) | −.023 | [−.175, +.130] | .74 |
| V4 axis + NLI − RoBERTa (softmax, superseded) | +.055 | [−.094, +.205] | .49 |
| V4 axis − RoBERTa (softmax, superseded) | −.047 | [−.203, +.081] | .49 |

EU-OOD under the paper protocol (leave-one-speaker-out thresholds):

| Configuration | Neutral F1 | Anti-elitism F1 | People-centrism F1 | Macro-F1 | AE AP | PC AP |
|---|---:|---:|---:|---:|---:|---:|
| V3 configuration | .917 | .260 | .405 | .528 | .197 | .352 |
| V4 axis only | .908 | .284 | .406 | .533 | .196 | .352 |
| **V4 axis + NLI** | .896 | .285 | **.455** | .545 | .291 | **.384** |
| RoBERTa (our run), sigmoid | .924 | .332 | .410 | **.555** | .317 | .347 |
| RoBERTa (our run), softmax (superseded) | .928 | .295 | .312 | .512 | .284 | .329 |
| Gemini 2.5 Flash, zero-shot (paper) | – | – | – | .590 | – | – |
| Qwen3-14B, zero-shot (paper) | – | – | – | .613 | – | – |

Per-speaker macro-F1 for axis + NLI: Le Pen .481, Weidel .315, Zemmour .468, Kickl .524, Tsipras .594. Weidel has no anti-elitism sentences in the gold labels, so her score reflects neutral and people-centrism only.

The V3 configuration reproduces V3's published EU macro-F1 (.528) and the superseded softmax RoBERTa run reproduced its published .512, which confirmed that the final harness matches the V3 protocol. The corrected sigmoid comparator scores .555 under the same harness; the paper's own RoBERTa figure (.508) is not a validation of either activation.

## What V4 established

1. **Pipeline defects cost accuracy; construct refinements did not add it.** The two adopted axis changes are a bug fix (window cap) and pooling depth (top 8 of three-sentence windows). Every change to the representation itself was rejected on the development data: anchors, geometry and encoder.
2. **The benchmark labels reward hostile tone.** Projecting nuisance directions out of the axes (step 2) removed the E axis's coupling to negative affect (+.34 → about 0) and lowered agreement by about .06. Affect-matched minimal-pair anchors (step 3) reduced the coupling only a little (+.34 → +.28) and also lowered agreement. In e5-large-v2, the anti-elite claim and its hostile register share a direction.
3. **The GPD rubric rewards breadth.** Speech scores improve steadily as more windows are pooled. This fits a rubric that grades how pervasive populism is across a speech.
4. **Instruction templates hurt axis projection.** Model-card instructions lowered agreement by .05–.08. Anchors and windows must be encoded identically. Qwen3-Embedding-8B is the strongest anti-elitism (E) encoder, and significantly better on Agora, but weaker on English people-centrism.
5. **An explicit claim check complements the geometry.** NLI entailment against ten written anchor claims, pooled over the top windows, adds information the axis lacks. On the development data it was weaker than the axis alone (English .468 against .517); on the English holdout, NLI alone matches the combination (.520 against .518). The combination is the pre-declared configuration and the one reported.

## Limits

- **Small English holdout.** It has 84 speeches from 27 leaders and only 4 extreme positives. The intervals are wide (±.2), and the extreme AUROC and AP figures rest on 4 positives. The significant V3→V4 difference (+.142) has a lower bound near zero (+.004).
- **NLI is stronger on the holdout than on the development data.** NLI alone scores .520 on the holdout against .468 on the development split. This discrepancy is unexplained, and a single 84-speech holdout cannot tell whether it is a real property or chance.
- **EU still trails zero-shot LLMs.** EU-OOD remains .045–.068 below the paper's zero-shot LLMs. The anti-elitism F1 barely moved (.260 → .285); the EU gain is in people-centrism (.405 → .455) and in ranking quality (AE AP .197 → .291).
- **The English holdout was not blind to V3.** It was visible to V3's configuration choices; it is unseen by V4 decisions only. EU-OOD was scored five times during V3 and once in V4.
- **Development decisions rest on two datasets.** They used 174 English speeches (57 leaders) and Agora (6 politicians). The Agora intervals rest on six clusters and understate uncertainty.
- **The NLI stage is less transparent than the axis.** It is a 400M-parameter cross-encoder. It remains inspectable per window, but only because its hypotheses are readable claims.

## Provenance

- Final scoring: `scripts/final_holdout.py --final` (then `scripts/v4/final_holdout_v4.py`), commit `8f95b3c`. Outputs are in `results/final/` (`final_results.json`, `final_results.md`, `eu_paper_protocol.json`, one folder per configuration). The B4 and final manifests record `dirty: true`. The modified tracked file was the holdout access log, which the B0 run wrote during the same job. `results/final/PROVENANCE.md` gives the evidence that the executed code is `8f95b3c` and the limits of that evidence.
- Holdout access log: `results/holdout_access_log.json`, 4 entries: the declared step 1 V3 reproduction on all 258 speeches, then B0, B4 and B6 in the single final run.
- GPU use for the final step: EU encodes about 1 minute; NLI 23 s; RoBERTa over English, EU and Agora about 4 minutes. Peak VRAM was 3.2 GB.
- Total V4 GPU time on pvgpu1: about 2.5 hours, most of it the step 5 encoder comparison (the Qwen3-8B encode took 27 minutes at 15 GB).
