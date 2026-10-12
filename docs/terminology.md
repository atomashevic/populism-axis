# Terminology and prose conventions

Prepared 2026-10-12 to align the manuscript-facing text (`abstract.md`, `README.md`, `docs/method.md`, `docs/theory.md`) with the usage of the Journal of Computational Social Science (JCSS). No number, result, claim strength or method was changed; the development log is kept as written.

## Reference articles

- **Method and genre:** Ohamadike, N., Durrheim, K., & Primus, M. (2026). Anchoring race: improving the construction of race dimensions in word embeddings. *Journal of Computational Social Science, 9*, article 20. https://doi.org/10.1007/s42001-025-00449-w (open access). No JCSS article measures populism in text with embeddings or NLI. This is the closest match: a full methods article that builds a construct direction from written anchors in a pretrained embedding space, projects texts on it, checks the quality of the axis and validates the projections against human ratings.
- **Populism vocabulary:** Sermpezis, P., et al. (2026). AgoraSpeech: a multi-annotated comprehensive dataset of political discourse through the lens of humans and AI. *Journal of Computational Social Science, 9*, article 36. https://doi.org/10.1007/s42001-026-00469-0 (open access). It is the source of our AgoraSpeech benchmark and defines populism by two criteria, people-centrism and anti-elitism, that "must be simultaneously present".

Taylor and Stoltz (2021, JCSS 4, 231–242), already cited as R102, is the closest word-level ancestor but is a short note, not a validation study.

## Term map

| Our term before | Term now | Basis |
|---|---|---|
| channel (P, E, M) | dimension; the code and output files keep "channel" | Ohamadike et al. call each construct direction a "dimension"; one term per concept, where the reference article mixes dimension, axis and direction |
| people sovereignty, people-sovereignty claims | people-centrism (P), written as popular-sovereignty claims | AgoraSpeech and the populism literature name the criterion people-centrism; the anchors operationalize its sovereignty form (`docs/theory.md` section 2.1) |
| elite obstruction | anti-elitism (E), written as elite-obstruction claims | AgoraSpeech |
| populist and mainstream propositions | anchor propositions at the populist and mainstream poles | Ohamadike et al.: "anchor terms", "poles" |
| frozen sentence encoder, frozen embedding space | pretrained sentence-embedding model or space, with fixed weights | Ohamadike et al.: "pre-trained models"; "frozen" is kept only for the fixed configuration and the stored reference statistics, which the code names that way |
| the claim check tests whether passages assert the claims | the claim check scores support for the written claims (an entailment probability) | Median entailment is .001 and no window exceeds .5 on both dimensions (`docs/theory.md` section 4.3); matches correction M4 in `docs/manuscript-corrections-proposed.md` |
| requires both; conjunction of projections | soft conjunction (geometric mean of the activated projections) | The geometric mean is compensatory (`docs/theory.md` section 3.4); AgoraSpeech's "both criteria must be simultaneously present" is the strict form |
| channel projections and matched claims | P and E projections and best-matching claims | |
| statistically supported | paired 95% interval excludes zero | States the test instead of a verdict |
| failures | limits | |
| four languages | English and, through English translation, German, Portuguese and other languages | The fourth language was never named; all non-English benchmarks except the Portuguese sensitivity route are read in translation |
| loss against XLM-T on tweets | numeric deficit; paired interval includes zero (−.052, [−.102, +.002]) | `research/development-log/external-antielite-tweets.md` |
| development-set optimum (fusion weight) | declared in advance, on or next to the development-set optimum | English-dev peaks at .6 (+.004), `research/development-log/fusion-weight-sweep.md` |
| holds up across | applied unchanged to | |

"Populism axis" and "claim check" stay as the names of the two stages, each defined once where it first appears.

## Prose conventions taken from the comparison

1. **One term per concept.** The reference article uses dimension, axis and direction interchangeably, and anchors, anchor pairs and anchor groups. We use dimension for the construct component, direction or axis for the vector built from its anchors, and anchor propositions for the sentences.
2. **No bold for emphasis in running text.** JCSS articles use bold only in formulas and labels. Bold stays for run-in list labels, anchor-rating labels and reference tags.
3. **State design choices as decisions.** "We write ten populist and ten mainstream anchor propositions" rather than a justification of why the choice is defensible.
4. **Qualify with the evidence, once.** A limit names its mechanism and its consequence, as AgoraSpeech's limitations section does ("translation", "paragraph-level analysis"), and is not repeated as a stack of negations. Qualifications that keep a claim precise stay: the holdout's exposure, Agora as a development set, the paired intervals, the absence of pre-registration.
5. **No intensifiers or verdict adjectives.** Cut "particularly", "especially", "unusually", "honest", "strongest case", "defensible", and the AgoraSpeech register ("meticulously", "rigorous", "robust", "comprehensive").
6. **Numbers with their interval.** Report a difference with its paired 95% interval in brackets or as "CI a to b", as Ohamadike et al. report agreement statistics.

## Not changed here

- The manuscript draft itself is not in this repository; the same term map applies to it.
- `research/development-log/` is kept as written, including `summary.md:12`, which says EU-OOD was "scored once"; its own Limits section and `abstract.md` give the full exposure history.
