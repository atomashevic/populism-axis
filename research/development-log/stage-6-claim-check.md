# Stage 6: NLI entailment second stage

Date: 2026-09-24. Code commits `c9bfdce` and the evaluation-script commit after it.

## Purpose

The axis measures how close a window is to the populist pole of an embedding direction. It cannot check whether the window actually asserts the populist claim. This step adds a second stage that asks exactly that: a natural-language-inference (NLI) model scores whether each of a document's top windows entails the populist anchor claims. The axis stays the primary measure; NLI is scored alone and combined with it.

## Change

- `src/populism_axis/contextual_axis.py`: `window_conjunction_scores` is factored out of `aggregate_scores`. The carried scores are bit-identical.
- `scripts/prepare_nli_requests.py` (CPU). Premises are the top-8 windows of each document under the carried configuration. Hypotheses are the five P and five E populist anchor sentences closest to their own axis, chosen label-free (`results/step6_nli/hypotheses.json`).
- `scripts/nli_stage.py` (pvgpu1): `facebook/bart-large-mnli`, fp16. Pairs are processed in order of premise length, and the output columns are fixed to contradiction, neutral, entailment.
- `scripts/evaluate_nli.py` (CPU), with the scoring rule pre-declared in the V4 plan:
  - window P score: highest entailment probability over the P hypotheses; E likewise;
  - window NLI score: √(P·E), a conjunction like the axis's;
  - document NLI score: mean over the top-K windows;
  - combined score: mean of the within-dataset z-scores of the axis and NLI document scores.

## Config

- Axis: the carried configuration B4 (e5-large-v2, V3 anchors, three-sentence windows, PE soft, top-8).
- NLI premises: 1,392 English-dev windows (174 speeches × 8) and 22,668 Agora windows. Many Agora paragraphs have fewer than 8 windows. The median premise is 59 and 49 words.
- Hypotheses:
  - P: "The ordinary people are the only true source of legitimate political power."; "This movement is nothing more and nothing less than the voice of ordinary people."; "It is the people, not the officials, who must decide the future of this country."; "Power must be returned to the hands of the people who built this nation."; "Our people are sovereign and no authority stands above them."
  - E: "Entrenched insiders block every reform that would help ordinary families."; "A corrupt elite has captured our institutions and turned them against the public."; "Bankers, lobbyists and career politicians have rigged the system against you."; "The people in power have betrayed the country they were elected to serve."; "They enrich themselves while the rest of the country is told to sacrifice."
- Pre-declared rows: axis only (B4), NLI only with K = 8, combined with K = 8. NLI only with K = 3 is a sensitivity row. The adoption rule applies to combined against axis only.

## Results on development sets

| Configuration | English ρ | English 95% CI | Extreme AUROC | Leader-term ρ | Agora ρ | Agora 95% CI | Agora high-vs-low AUROC |
|---|---:|---|---:|---:|---:|---|---:|
| B4 axis only (carried) | .517 | [.375, .645] | .921 | .308 | .359 | [.245, .380] | .874 |
| NLI only, top 8 | .468 | [.352, .576] | .939 | .398 | .342 | [.272, .359] | .752 |
| NLI only, top 3 | .333 | [.183, .478] | .724 | .258 | .332 | [.253, .346] | .753 |
| **Axis + NLI, top 8** | **.537** | [.421, .642] | **.951** | **.409** | **.382** | [.295, .397] | .859 |

Paired differences against the axis alone:

| Row | English-dev Δ [95% CI] | Agora Δ [95% CI] |
|---|---|---|
| NLI only, top 8 | −.049 [−.165, +.076] | −.017 [−.047, +.027] |
| NLI only, top 3 | −.183 [−.346, −.019] | −.027 [−.058, +.008] |
| **Combined, top 8** | **+.020 [−.041, +.085]** | **+.024 [+.006, +.049]** |

### What the results show

- **NLI alone is weaker than the axis, but the two are complementary.** NLI alone loses on both datasets. Combined with the axis, it improves both. Window-level agreement between the axis score and the NLI score is moderate (Spearman .39 on English, .42 on Agora), so NLI adds information the axis does not carry.
- **The combination improves every English metric.** Spearman rises .517 → .537, extreme AUROC .921 → .951, extreme average precision reaches .642, and leader-term ρ rises .308 → .409. On Agora, Spearman gains +.024, which is significant; high-vs-low AUROC falls slightly, .874 → .859.
- **Entailment is sparse and works as a ranking signal.** Median entailment for a window–hypothesis pair is .001, the 99th percentile is .064, and fewer than 0.3% of windows exceed .5 on either channel. No window exceeds .5 on both. BART-MNLI rarely judges a speech window to entail these strongly worded claims, but the ordering of small probabilities still tracks populism.
- **Deeper pooling matters again.** NLI over the top 3 windows is much worse than over the top 8 (English .333 against .468), which matches the step 4 finding.
- **Two hypotheses dominate.** P05 ("…the voice of ordinary people") and E04 ("The people in power have betrayed the country…") are the best-matching hypotheses for most windows (English: 885 of 1,392 and 944 of 1,392). These are the most general claims in the set; the channel scores therefore rest mostly on two broad propositions.

### Evidence (English-dev, top combined scores)

| Speech | GPD | Top window (abridged) | Best P hypothesis (entailment) | Best E hypothesis (entailment) |
|---|---:|---|---|---|
| Iran_Ahmadinejad_Campaign_1 | 1.67 | "…the people are the true rulers of the country…" | P01 "It is the people, not the officials, who must decide…" (.74) | E04 (.001) |
| Philippines_Duterte_Famous_1 | 1.15 | "I was elected by the people, not by the military, not by the police…" | P08 "Our people are sovereign…" (.19) | E04 (.04) |
| Philippines_Arroyo_Famous_2 | .05 | "It is immoral for the government to grow unresponsive, even corrupt…" | P05 (.32) | E04 (.07) |
| UK_Brown_Campaign_1 | .00 | "…there are some who let our country down…" | P05 (.03) | E04 (.21) |

The stage catches explicit sovereignty claims, as in the Ahmadinejad window. It also rewards moralized criticism of government that coders did not score as populist, as in the Arroyo and Brown windows.

## Decision

**Adopt axis + NLI (combined, top 8)** as the carried configuration B6. It passes the adoption rule: English-dev Δ +.020 (at least +.01), Agora Δ +.024 (at least −.01), and neither interval lies below zero. The English interval includes zero; the Agora gain is significant. The axis-only score B4 remains the primary interpretable measure and is scored beside B6 in the final holdout step. Carried configuration: `results/step6_nli/carried_config.json`.

## Provenance

- NLI job: pvgpu1, commit `c9bfdce`, clean tree. English 13,920 pairs in 25 s; Agora 226,680 pairs in 299 s; peak VRAM 3.2 GB. The GPU returned to idle afterwards.
- The request files are gitignored and were copied to pvgpu1 by tar over ssh. Their summary and hypotheses are tracked.
- The DeBERTa-v3-large NLI model was not used: its download was offered but not approved, so the cached BART-large-MNLI model was used as planned.

```bash
uv run --extra cpu python scripts/prepare_nli_requests.py
# pvgpu1
scripts/remote_job.sh step6_nli -- python scripts/nli_stage.py --names english_dev,agora --model facebook/bart-large-mnli --batch-size 64
# local
uv run --extra cpu python scripts/evaluate_nli.py
```
