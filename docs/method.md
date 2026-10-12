# The populism axis and the claim check

The measure has two stages and trains no model. The first stage, the **populism axis**, places each passage of a text on two directions in a pretrained sentence-embedding space, one for people-centrism and one for anti-elitism, and combines the two projections as a soft conjunction. The second stage, the **claim check**, scores the passages the axis ranked highest with a natural-language-inference model's entailment probabilities for the written populist claims. The final score averages the standardized scores of the two stages.

## Construct

Following the ideational definition, populism is measured as the conjunction of two dimensions: people-centrism (P), written as the claim that the ordinary people are the legitimate source of political authority, and anti-elitism (E), written as the claim that a corrupt or self-serving elite obstructs their will. Each dimension is defined by ten populist and ten mainstream anchor propositions written from that definition (`anchors/proposition_anchors.json`); the code and output files call the dimensions channels. Examples:

| Dimension | Populist pole | Mainstream pole |
|---|---|---|
| P | "The ordinary people are the only true source of legitimate political power." | "Complex policy questions require careful analysis by qualified specialists." |
| E | "A corrupt elite has captured our institutions and turned them against the public." | "Our public servants have worked diligently to implement this programme." |

A third dimension, moralised antagonism (M), is kept in the anchor file as a diagnostic and does not enter the score.

## Stage 1: the axis

1. **Windows.** Split the text into three-sentence windows with stride one; sentences longer than 60 words are chunked. A speech yields about 130 windows; a single sentence yields one.
2. **Encoding.** Encode every window and every anchor sentence with `intfloat/e5-large-v2` (mean pooling, "query: " prefix, 256 tokens). Anchors and windows are rendered identically; instruction templates lower agreement and are not used.
3. **Axes.** For each dimension, the axis is the unit vector from the mean of the mainstream anchors to the mean of the populist anchors.
4. **Centering.** Subtract the corpus mean embedding from every window and L2-normalise. The embedding space is anisotropic; without this step the projection is dominated by a component shared by every sentence.
5. **Projection and activation.** Project each window on both axes, z-score within the corpus, and activate with softplus(z / .25)² divided by the nonzero median.
6. **Conjunction and pooling.** The window score is the geometric mean of the P and E activations. The document score is the mean of the top 8 windows for speeches and paragraphs, or the top window for sentences and tweets.

## Stage 2: the claim check

The axis measures alignment with a direction, which does not show whether a window supports a claim or only discusses it. The claim check sends the same top-K windows as premises to `facebook/bart-large-mnli` against ten hypotheses, the five P and five E anchor sentences closest to their own axis (`anchors/claim_hypotheses.json`). Per window, P is the maximum entailment over the P claims and E likewise; the window score is √(P·E); the document score is the mean over the top-K windows. The entailment probabilities are small (median .001) and the signal is their ordering, not confident entailments.

## Combination

The final score is the mean of the within-corpus z-scores of the two document scores, weight .5 each. On single-component benchmarks only the E dimension and the E claims, or the P dimension and the P claims, are scored; the claim-check premises are still the windows selected by the PE axis (`docs/theory.md` section 6.4). The weight .5 was declared in advance and is on or next to the development-set optimum (English-dev peaks at .6, +.004); the spent evaluation sets favour a little more weight on the claim check, and a fitted fusion of the two inputs cannot gain more than about .04 anywhere (`research/development-log/fusion-weight-sweep.md`).

## Standardisation

Every statistic above is label-free but computed on the corpus being scored: the centering mean, the projection scales, the activation scale and the document-level scales of the combination. Freezing all of them on the English development set changes axis rankings by less than one point of rank correlation on every benchmark; only the document-level scales of the combination are corpus-dependent, and by at most .03 (`research/development-log/frozen-reference-scales.md`). `scripts/score_texts.py --reference frozen` uses the stored statistics, which makes single texts scorable.

## What is inspectable

Every score decomposes into named windows, each with its P and E projections, its conjunction value, and the best-matching claim with its entailment probability (`evidence.json` from the scorer; `evidence_english.json` under `results/final/`). The construct is the written anchors. The two things that are not inspectable are the encoder's placement of a window in the space, which couples the elite claim to its hostile register, and the entailment model's judgement, whose low-tail ordering is model-specific.

## Cost

No training. Per speech, about 130 encoder passes of a 335M-parameter model and 80 cross-encoder passes of a 407M-parameter model. On a 6-core laptop CPU that is one to two minutes per speech and a few seconds per sentence; on an A30 GPU about a minute per thousand speeches for the encoder and 600 to 750 pairs per second for the claim check. Cheaper entailment models were tested and rank the low tail differently; none replicates the result (`research/development-log/claim-check-replication.md`).
