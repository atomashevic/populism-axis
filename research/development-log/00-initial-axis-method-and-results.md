# Initial populism axis: method and results

Date: 2026-07-27. Final configuration: **proposition anchors + `intfloat/e5-large-v2`, PE system, soft conjunction, top-3 windows.**

## Approach

Three theory channels, each realized as one linear direction in a frozen
contextual sentence-embedding space. A document score is the mean of its top-3
sentence windows, where a window scores as the weighted geometric mean of its
channel activations. Nothing is trained: the encoder is frozen, the anchors are
written text, and the only fitted quantity at evaluation time is a threshold.

The pipeline is inherited unchanged from the V2.1 static measure — same channel
structure, same activation, same conjunction, same top-k rule. Two things
changed: what defines an axis, and what represents a sentence.

## The axes and what they measure

Each channel is defined by 10 populist claims contrasted against 10 mainstream
claims, written from the ideational definition of populism and stored at
`anchors/proposition_anchors.json`. The axis is
`normalize(mean(populist) − mean(mainstream))`.

| Axis | Measures | Populist pole | Mainstream pole |
|---|---|---|---|
| **P** | people sovereignty | "The ordinary people are the only true source of legitimate political power." | "Complex policy questions require careful analysis by qualified specialists." |
| **E** | elite obstruction | "A corrupt elite has captured our institutions and turned them against the public." | "Our public servants have worked diligently to implement this programme." |
| **M** | moralized antagonism | "This is a battle between honest people and a thoroughly corrupt establishment." | "Reasonable people can disagree about the best way forward." |

The headline system is **PE**. M is retained as a diagnostic; adding it does not
improve results (PEM: English rho `.427` vs PE `.433`).

This replaced the V2.1 word-pair inventory (966 abstract noun pairs such as
`captured institutions / representative institutions`). Word pairs name a
*concept*; the sentences being scored make a *claim*. Under inspection the
word-pair P channel fired hardest on "we must deepen the practice of
participatory democracy" — mainstream proceduralism, the opposite of the target.

A polarity defect in the V2.1 inventory was found and fixed along the way: E was
written pathology-positive in 33/36 families while P and M were virtue-positive
in 0/36, and the scorer combined all three with an implicit `+1`. The frozen
primary candidate `raw_PM` therefore multiplied participatory virtue by
professed moral virtue. Correcting the sign inside the otherwise untouched
static pipeline moved `raw_PM` from Spearman `.019` to `.258` (paired
leader-clustered delta `+.239` [`+.067`, `+.425`], p=`.008`).

## Parallelism

Parallelism is the repository's reliability diagnostic (Boutyline & Johnston):
mean pairwise cosine among a channel's anchor difference vectors. Low values
mean the anchors are pulling in different directions.

| Channel | Index-paired (n=10) | All populist×mainstream contrasts (n=100) |
|---|---:|---:|
| P | `.237` | `.296` |
| E | `.239` | `.311` |
| M | `.224` | `.304` |

Between-channel axis cosines: P·E `+.503`, E·M `+.559`, P·M `+.266`. All three
are positive, i.e. the channels point toward a common populist pole — which is
what a coherent construct should look like, and what the V2.1 inventory did not
produce.

Two cautions. First, these values are *lower* than the V2.1 static parallelism
retention (`.684` for `raw_PM`), yet performance is far higher. Parallelism
measures internal coherence, not validity: the V2.1 geometry-only selection
rewarded P and M precisely because they were redundant virtue directions.
**Parallelism should not be used as a selection criterion again.** Second,
parallelism is not comparable across representations; contextual difference
vectors are not the same objects as static ones.

A multi-prototype variant (splitting E into 7 named sub-frames, pooled by max or
mean) was tested and **rejected**: EU macro-F1 `.449` (max) and `.492` (mean)
against `.528` for the single flat axis. Anti-elitism is not usefully multimodal
in this space.

## Working with a non-static embedding model

Static vectors are type-level, so V2.1 averaged word vectors over a window.
A contextual encoder is token-in-context, so the window itself is the unit:

1. Split each document into sentence windows.
2. Encode every window once with the frozen encoder (mean pooling over tokens;
   e5 requires the `query: ` prefix).
3. **Subtract the dataset mean embedding, then L2-normalize.** Contextual spaces
   are strongly anisotropic; without this the projection is dominated by a
   common component shared by every sentence. This is the contextual analogue of
   all-but-the-top.
4. Project each window onto each channel axis, then z-score within dataset so
   the activation threshold is on a common scale across encoders.
5. Activation = `softplus(z)²`, scaled by the nonzero median.
6. Window score = weighted geometric mean across required channels; document
   score = mean of the top 3 windows. The soft (softplus) conjunction replaces
   V2.1's hard hinge, which returned exactly zero if any channel failed to
   activate and destroyed the ranking when a claim split across sentences.

Encoder sweep (English + Agora only, EU held out): e5-large-v2 `.433` >
Qwen3-Embedding-0.6B `.394` > bge-large-en-v1.5 `.377` > bge-small-en-v1.5
`.310`. Capacity matters below ~300M; above that it plateaus. The 33M model is
not viable — it scores at the static baseline.

Explainability is preserved: every document score decomposes into named
channels, a linear projection, and the specific windows that drove it. For
`India_Modi_Campaign_1` (GPD rubric `1.80`, corpus maximum): P `+2.84`, E
`+2.80`, M `+3.56` on "Now we will not let the people sit in peace who have
plundered this country since independence."

## Absolute performance

| Dataset | n | Metric | Value | 95% CI |
|---|---:|---|---:|---|
| English GPD speeches | 258 | Spearman vs `averagerubric` | **`.4330`** | [`.303`, `.545`] (leader-clustered) |
| | | extreme AUROC (13 vs 145) | `.9162` | [`.837`, `.975`] |
| | | average precision | `.6135` | |
| | 111 | leader-term Spearman | `.3271` | [`.128`, `.501`] |
| EU-OOD sentences | 1,762 | macro-F1 (paper protocol) | **`.528`** | |
| | | AUROC 3-vs-0 | `.8904` | [`.775`, `.908`] (speaker-clustered) |
| | | within-speaker AUROC | `.8614` | |
| Agora paragraphs | 5,272 | Spearman vs `populism_human` | **`.3574`** | [`.267`, `.381`] (politician-clustered) |
| | | high-vs-low AUROC | `.8914` | |
| | | within-speaker AUROC | `.8795` | |

Reference points: the human coder ceiling on English GPD is Spearman `.7161`
(coder pair 1v2, n=211). A word-count-only predictor reaches `.2466`. Speaker
identity alone reaches EU AUROC `.8368` and Agora rho `.2338`.

## Comparative performance

### Against our own RoBERTa run and prior in-repo measures

> **Correction, 2026-10-02.** The RoBERTa numbers in this section use a softmax across the checkpoint's three labels; the checkpoint has independent sigmoid outputs. They are kept as the historical record and were not recomputed for these development-era subsets. Corrected comparator results for the ten benchmark rows: `docs/roberta-correction.md`.

All rows scored on the identical V3 texts with the identical harness.

| Method | ENG rho | ENG AUROC | ENG leader-term rho | EU macro-F1 | EU AUROC | EU within-spk | Agora rho | Agora AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Human coder ceiling | `.7161` | — | — | — | — | — | — | — |
| **Proposition PE (ours)** | **`.4330`** | **`.9162`** | **`.3271`** | **`.528`** | `.8904` | `.8614` | `.3574` | `.8914` |
| RoBERTa, anti-elitism top3 | `.4423` | `.8806` | `.2349` | — | — | — | `.3407` | `.9049` |
| RoBERTa, populist total max | `.4169` | `.8464` | `.1977` | `.512` | `.9021` | `.8623` | `.3674` | `.9680` |
| Sub-frame PE, mean (rejected) | — | — | — | `.492` | — | — | — | — |
| Sub-frame PE, max (rejected) | — | — | — | `.449` | — | — | — | — |
| Word-pair contextual PEM | `.2038` | `.8106` | — | `.387` | `.6993` | `.6857` | `.2831` | — |
| V2.1 static `raw_PM` sign-fixed | `.2581` | `.7687` | — | — | — | — | `.2188` | — |
| V2.1 static P/E frozen | — | — | — | `.408` | — | — | — | — |
| V2.1 static `raw_PEM` frozen | `.1679` | `.7358` | — | — | — | — | `.1812` | — |
| V1 static canonical | `.1065` | `.7188` | — | — | — | — | `.1752` | — |
| V2.1 static `raw_PM` frozen primary | `.0190` | `.4626` | — | — | — | — | `.0964` | — |
| Word-count null | `.2466` | `.7446` | — | — | — | — | — | — |
| Speaker-identity null | — | — | — | — | `.8368` | `.5000` | `.2338` | `.6870` |

Paired leader-clustered deltas, reference = proposition PE:

| Comparison | ΔSpearman | 95% CI | p |
|---|---:|---|---:|
| vs V1 static canonical | `+.3264` | [`+.178`, `+.456`] | `<.001` |
| vs V2.1 frozen primary `raw_PM` | `+.4140` | [`+.258`, `+.579`] | `<.001` |
| vs word-count null | `+.1863` | [`+.063`, `+.297`] | `.0025` |
| vs RoBERTa anti-elitism top3 | `−.0093` | [`−.101`, `+.075`] | `.81` |

Read: we are **statistically indistinguishable from the fine-tuned RoBERTa** on
English, ahead of it on extreme-contrast AUROC (`.916` vs `.881`) and on the
leader-term unit, and 3–4× the canonical V1 measure. V1 sits below a word
counter and below the EU speaker-only null.

### Against the published benchmarks (Chalkidis et al. 2025, arXiv 2507.19303)

EU-OOD, 3-way multi-label macro-F1, translated. Our harness reproduces the
paper's RoBERTa at `.512` against their published `.508` (Δ`.004`), which
validates the protocol and makes these rows directly comparable.

| System | macro-F1 | Source |
|---|---:|---|
| Qwen3-14B, zero-shot | **`.613`** | paper Table 12 |
| Gemini 2.5 Flash, zero-shot | **`.590`** | paper Table 12 |
| **Proposition PE (ours)** | **`.528`** | this work |
| RoBERTa fine-tuned | `.512` | our replication |
| RoBERTa fine-tuned | `.508` | paper Table 12 |
| V2.1 static P/E frozen | `.408` | this work |
| Proposition PE, bge-small 33M | `.406` | this work |
| Word-pair contextual PE | `.387` | this work |

Per-class F1 on EU-OOD: ours N `.918` / AE `.262` / PC `.405`; RoBERTa N `.928` /
AE `.295` / PC `.312`. **Our people-centrism exceeds fine-tuned RoBERTa;
anti-elitism is the deficit.**

Paper's in-domain TRUMP-2016 test set, for context (macro-F1): RoBERTa `.746`,
DeBERTa `.731`, TF-IDF-SVM `.630`, Qwen3-14B `.582` zero-shot / `.740`
fine-tuned, Gemini 2.5 Flash `.574`, Qwen3-32B `.542`, Llama-3.1-70B `.507`,
GPT-4.1 `.503`, Gemma3-27B `.472`, Llama-3.1-8B `.460`, Gemma3-12B `.437`,
random `.350`. Human student annotators score `.737` against an expert oracle.

The paper's own headline is that RoBERTa loses ~24% going out of domain
(`.746` → `.508`) while zero-shot LLMs hold steady. So matching RoBERTa is a
weaker achievement than it appears: RoBERTa is the system that fails to
transfer. We beat it on its weak axis and remain `−.062` behind Gemini.

### Agora

The AgoraSpeech CSV ships GPT-3.5 annotations (`populism_gpt`) alongside the
human labels, scoring Spearman `.722` against them. **This is not a usable
baseline**: the paper's method is GPT-first, human-validated-second, so the
human labels are corrections of the GPT labels and agreement is inflated by
construction.

## Status and limits

- English Spearman `.433` against a human ceiling of `.716` is ~60% of
  attainable agreement. Not a solved measurement problem.
- EU-OOD is 1,762 sentences from 5 speakers and 6 speeches; the paper flags it
  as small. Three evaluations have been spent on it (flat, sub-frame max,
  sub-frame mean), so the estimate should not be iterated against further.
- English GPD does not transfer as a development proxy for EU: sub-frames won on
  English (`.4515` vs `.4330`) and lost on EU (`.492` vs `.528`).
- The proposition anchors were written in one pass by one author from theory,
  with no reviewer panel. They need the review discipline the V2.1 inventory got.
- The encoder is a frozen 335M model. "Explainable" refers to the axis and the
  aggregation, not to the encoder.
- Reaching Gemini's `.590` needs AE F1 `.262` → ~`.447`; calibration offers only
  `+.021`, so it requires roughly 2× the anti-elitism ranking quality. Anchor
  engineering and encoder scale are both exhausted; the remaining candidate is
  NLI-entailment scoring, not built.
