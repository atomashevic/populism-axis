# Stage 5: encoder comparison

Date: 2026-09-24. Code commits `198502d`, `52cda58`, `b95ecfa`.

## Purpose

V3 chose `intfloat/e5-large-v2` from four encoders. It concluded that capacity plateaus above about 300M parameters, but it encoded Qwen3-Embedding without the instruction prefix its model card specifies. This step compares instruction-tuned and larger encoders on the carried V4 configuration, with each encoder's documented template and with plain text.

## Change

- `src/populism_axis/v4_encoders.py`:
  - Qwen3 models run in bfloat16.
  - An instruction template gets its own token budget on top of the window limit.
  - Non-finite embeddings abort the run.
  - Plain-template specs were added for multilingual-e5-instruct and Qwen3-4B/8B, along with a Qwen3-Embedding-8B spec.
- `scripts/encode_windows.py` counts truncation against the effective token limit.
- `scripts/sweep_encoders.py` runs the grid.
- Qwen3-Embedding-8B (15 GB) was downloaded to pvgpu1 with the project lead's approval, after Qwen3-4B showed that capacity helps within the family.

## Config

- Carried configuration from step 4 (B4): V3 anchors, V3 axis rule, PE soft, three-sentence windows (stride 1, 60 words per sentence, 256 tokens), mean of the top 8 windows. Only the encoder changes.
- The step 4 sensitivity rows (one-sentence windows with top 5 or top 8) were also run for the smaller encoders.
- Every anchor and window is encoded with the same template. With the instruction template, anchors and windows both carry it; the plain template adds nothing. An asymmetric variant, with the instruction on anchors only as model cards suggest for queries, was run once for Qwen3-0.6B.
- gte-large-en-v1.5 was dropped. Its remote model code fails with an index-out-of-bounds CUDA error under transformers 5.6.

## Results on development sets

Full grid: `results/step5_encoders/grid.md`. Carried window setting, system PE soft. Δ is against B4 (e5-large-v2); CIs are 95% paired cluster-bootstrap.

| Encoder, template | Params | English-dev ρ | Δ English [95% CI] | Agora ρ | Δ Agora [95% CI] | Rule |
|---|---:|---:|---|---:|---|---|
| **e5-large-v2, `query:` (carried)** | 335M | **.517** | – | .359 | – | – |
| Qwen3-Emb-0.6B, instruction | 596M | .422 | −.095 [−.158, −.032] | .340 | −.019 [−.040, +.008] | fail |
| Qwen3-Emb-0.6B, instruction on anchors only | 596M | .401 | – | .304 | – | – |
| Qwen3-Emb-0.6B, plain | 596M | .504 | −.013 [−.075, +.054] | .344 | −.015 [−.034, +.013] | fail |
| multilingual-e5-large-instruct, instruction | 560M | .437 | −.080 [−.156, −.011] | .342 | −.017 [−.050, +.001] | fail |
| multilingual-e5-large-instruct, plain | 560M | .491 | −.025 [−.080, +.030] | .348 | −.011 [−.043, +.005] | fail |
| Qwen3-Emb-4B, plain | 4.0B | .515 | −.001 [−.061, +.067] | .367 | +.008 [−.023, +.043] | fail |
| Qwen3-Emb-8B, plain | 7.6B | .502 | −.014 [−.086, +.050] | **.392** | **+.033 [+.005, +.061]** | fail |

The asymmetric row was run outside the grid script, so it has no paired comparison.

Secondary metrics and single channels, carried window setting:

| Encoder, template | Extreme AUROC | Leader-term ρ | Agora high-vs-low AUROC | E English / Agora | P English / Agora | PEM English / Agora |
|---|---:|---:|---:|---|---|---|
| e5-large-v2 | .921 | .308 | .874 | .483 / .319 | .492 / .327 | .513 / .346 |
| Qwen3-0.6B, plain | .921 | .289 | .852 | .484 / .328 | .451 / .281 | .512 / .346 |
| me5-instruct, plain | .927 | .246 | .864 | .489 / .314 | .448 / .314 | .511 / .332 |
| Qwen3-4B, plain | **.943** | **.349** | .938 | .479 / .359 | .436 / .276 | .519 / .375 |
| Qwen3-8B, plain | .923 | .289 | **.939** | **.507** / **.378** | .444 / .305 | .512 / **.397** |

With one-sentence windows (the step 4 sensitivity rows), every new encoder lost .04–.16 against e5 on English.

### What the grid shows

- **The instruction prefix hurts this method.** For both instruction-tuned models, the model-card instruction lowers English-dev ρ by .05–.08 against plain text. Applying it to the anchors only, as model cards recommend for retrieval queries, is worse still (.401). An axis projection compares anchors and windows in one space, so they must be encoded identically, and the instruction pulls every sentence toward one task-shaped region. The protocol declared adding the instruction a defect fix. The data show the opposite: V3's omission was not a defect for this method. The plain template is the right one.
- **Capacity helps E and Agora, not English PE.** Within Qwen3 (plain), the E channel on Agora rises steadily (.328 → .359 → .378), as do Agora ρ (.344 → .367 → .392) and Agora high-vs-low AUROC (.852 → .938 → .939). Qwen3-8B's E channel is also the best on English (.507). English PE does not follow (.504 → .515 → .502), because the P channel is weaker in every Qwen model than in e5 (English P .436–.451 against .492).
- **The two benchmarks disagree about the encoder.** Qwen3-8B gives the first significant Agora gain in V4 (+.033 [+.005, +.061]) while being slightly worse on English. The adoption rule gives English priority, so this does not count as a win.
- **Qwen3-4B has the best secondary English metrics:** extreme AUROC .943 and leader-term ρ .349, against .921 and .308 for e5.

## Paired difference vs previous step

| Candidate | English-dev Δ [95% CI] | Agora Δ [95% CI] |
|---|---|---|
| Qwen3-4B, plain | −.001 [−.061, +.067] | +.008 [−.023, +.043] |
| Qwen3-8B, plain | −.014 [−.086, +.050] | +.033 [+.005, +.061] |

Paired comparisons for every cell are in the grid folders (`paired_vs_step4.json`).

## Decision

**Keep e5-large-v2 (B4).** No encoder cell passes the adoption rule; every English-dev difference is below +.01. The carried configuration is unchanged (`results/step5_encoders/carried_config.json`).

Two findings are recorded for the final assessment and for later work:

- **Qwen3-8B is the strongest E-channel encoder** and the best on Agora. If the project later decides that paragraph-level Agora agreement matters as much as speech-level English agreement, it is the natural candidate. A mixed design, with the E channel from Qwen3-8B and the P channel from e5, is untested and would need its own pre-declared step.
- **The instruction template should not be used** for axis projection with these models.

## Provenance

| Artifact | Commit | Host | GPU time | Peak VRAM |
|---|---|---|---:|---:|
| Qwen3-0.6B, instruction (spw3 + spw1) | `198502d` | pvgpu1 | about 11 min | 2.4 GB |
| Qwen3-0.6B, plain (spw3 + spw1) | `198502d` | pvgpu1 | about 9 min | 2.3 GB |
| me5-instruct, instruction (spw3 + spw1) | `198502d` | pvgpu1 | about 5 min | 1.3 GB |
| me5-instruct, plain (spw3 + spw1) | `52cda58` | pvgpu1 | about 4 min | 1.3 GB |
| Qwen3-4B, plain (spw3) | `52cda58` | pvgpu1 | 18 min | 8.4 GB |
| Qwen3-8B, plain (spw3) | `b95ecfa` | pvgpu1 | 27 min | 15.0 GB |
| Grid (CPU) | `b95ecfa` | local | – | – |

- A first Qwen3-4B run with the instruction template was stopped about 5 minutes in, once the 0.6B results showed the instruction hurts; its partial output was deleted.
- One launch was postponed by the load guard.
- Every job exited and returned the GPU to its idle baseline.
- English windows over 256 tokens with the Qwen tokenizer: 148 of 33,657.

```bash
# pvgpu1 (example)
scripts/remote_job.sh step5_encode_8b -- bash -c "python scripts/encode_sentences.py --encoder qwen3-emb-8b-noinstr --sentences anchors/proposition_anchors.json --set-id v3_proposition && python scripts/encode_windows.py --encoder qwen3-emb-8b-noinstr --datasets english,agora --sentences-per-window 3 --stride 1 --max-words 60 --max-length 256 --clean-text"
# local
uv run --extra cpu python scripts/sweep_encoders.py --reuse
```
