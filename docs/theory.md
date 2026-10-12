# Theoretical foundations of the populism axis and NLI claim check

Research synthesis, 24 September 2026. Method inspected at commit `4ad98d92045b001848553a2d1656c1946d95285b` on `tidy-main`. The target is the existing proposition-anchor system, including its BART-large-MNLI stage. No anchors, scoring rules, or benchmark configurations were changed for this report. Revised the same day with sections 2.6, 3.7 and 8, the computational-populism lineage, and references R55 onward.

This is a theory-informed measure of people-versus-elite claims, implemented through semantic contrasts and a second, entailment-based score. Established research supports its conceptual components, its focus on propositions, its separation of components from their conjunction, and its attempt to distinguish semantic resemblance from textual support. The exact encoder, anchor wording, activation, window size, pooling depth, and fusion weight remain operational choices. The literature supports this architecture more strongly than it supports any claim that the complete numerical pipeline follows uniquely from theory.

Three independent research strands informed this synthesis: sociology and political theory; communication, discourse and pragmatics; and NLP and semantic measurement. Literature claims are linked to primary publications or author manuscripts. The anchor mappings below are interpretive applications of those sources, not claims that the authors wrote or validated these exact sentences.

## 1. The most useful theoretical lineage

| Requirement | Closest supporting literature | What it supports here |
|---|---|---|
| Specify the content of populism before measuring it | [Mudde 2004][R01]; [Hawkins 2009][R02] | P and E as substantive claims, with a moral opposition embedded in their meaning. |
| Measure claims and their variation across texts | [Bonikowski and Gidron 2016][R03]; [de Vreese et al. 2018][R04] | A continuous passage/document measure that does not assign a fixed identity to a politician. |
| Represent relations in propositions | [Aslanidis 2018][R05]; [Popping 2018][R06] | Clause-level propositions interpreted in context, with identifiable people-side and elite-side roles. |
| Preserve the conjunction of components | [Wuttke, Schimpf and Schoen 2020][R07] | P and E should not become interchangeable indicators in a simple additive definition. |
| Operationalize semantic contrasts geometrically | [Kozlowski, Taddy and Evans 2019][R08]; [SemAxis][R09]; [FrameAxis][R10] | Positive/contrast anchor sets, centroid differences, and directional projections. |
| Connect semantic projection to construct measurement | [Deng and Li 2026][R49]; [Hurtado Bodell et al. 2026][R52] | Theory specifies the target and its indicators; projection provides a measurement operation. |
| Examine axis quality separately from predictive performance | [Boutyline and Johnston 2026][R21]; [Taylor et al. 2026][R51]; [Pokropek 2026][R53] | Anchor reliability, internal structure, and comparability require their own evidence. |
| Ask whether text supports a specified claim | [Dagan, Glickman and Magnini 2006][R11]; [Yin, Hay and Roth 2019][R12] | Treat a passage as premise and a construct proposition as hypothesis. |
| Keep semantic association distinct from commitment | [de Marneffe, Manning and Potts 2012][R13] | Motivation for a second stage that considers negation, attribution and contextual support. |
| Place the measure in the computational populism lineage | [Rooduijn and Pauwels 2011][R67]; [Gründl 2022][R70]; [Bonikowski, Luo and Stuhler 2022][R78]; [Jung et al. 2025][R85] | Dictionaries, supervised classifiers and LLM prompting are the three prior generations; section 8 places the axis relative to each. |
| Justify open models and validation beyond correlation | [Jankowski and Huber 2023][R75]; [Birkenmaier, Lechner and Wagner 2024][R87]; [Ollion et al. 2024][R88]; [Barrie, Palmer and Spirling][R90] | The transparency and measurement-practice gap stated in section 8.3. |

The most direct substantive predecessor is Aslanidis's proposition-based semantic analysis. The closest computational predecessors are semantic-axis measurement and zero-shot classification through NLI. Together they justify the design more specifically than a general appeal to discourse theory or to transformer models.

## 2. What the construct is

### 2.1 An ideational core expressed in discourse

Mudde's definition identifies a morally polarized people/elite division and the priority of the people's general will. Hawkins turns a closely related understanding into a discourse-coding procedure. These provide the substantive basis for the repository's P dimension, popular authority, and E dimension, elite interference with that authority. [R01] [R02]

Hawkins and Rovira Kaltwasser state the ideational family's measurement commitment directly: populism is a set of ideas expressed in discourse, so it is measured in texts rather than inferred from organization or strategy. [R59] Rooduijn's comparison of six populist actors across regions and periods identifies four elements they share: the central position of the people, criticism of the elite, the people perceived as a homogeneous entity, and the proclamation of a serious crisis. The P and E dimensions operationalize the first two. Homogeneity enters only through the wording of several P propositions, which speak of "the people" as one actor with one will, and crisis is not scored at all. The measure is therefore a measure of the first two of Rooduijn's four elements, and should be described that way. [R55] On the attitudes side, Schulz et al. separate anti-elitism, a preference for popular sovereignty, and belief in the homogeneity and virtue of the people as three first-order dimensions. Their sovereignty dimension is the closest survey analogue of the P dimension, which is consistent with the Portuguese finding that P plus the claim check ranks the separately coded sovereignty category better than the broader people-centrism category (section 6.2). [R56]

Elite obstruction has a communication-science counterpart in blame attribution. Hameleers, Bos and de Vreese define populist communication by the attribution of blame to elites, and show experimentally that emotionalized blame moves both blame perceptions and populist attitudes. This supports reading the E propositions as attributions of responsibility for harm to the people, and it also predicts the confound documented in the development log: blame is expressed with hostile affect, so a direction fitted to blame claims will carry negative tone unless the tone is explicitly removed. [R58]

The current system compresses several conceptual attributes into two dimensions. P includes sovereignty, claims of authentic representation, and confidence in ordinary people's judgment. E includes corruption, self-interest, capture, betrayal, and obstruction. Moral evaluation therefore already appears inside the scored P/E inventory. Keeping M as a diagnostic does not remove all morality from the construct. It does mean that the scoring rule does not separately require moral homogeneity or Manichean opposition.

The object of measurement is populist claims-making in a text. Bonikowski and Gidron specifically locate populism in political claims and show why its expression can vary with a politician's position and circumstances. This supports scoring speeches, passages and individual statements without treating party identity as the definition. [R03]

The measure is narrower than a comprehensive account of populist politics. It does not measure organization, leadership, mobilization, performance, audience reception, or institutional consequences. It is a proposition-based measure of people-versus-elite populist rhetoric, with P and E also available as component scores.

### 2.2 Why separate dimensions are useful

People sovereignty and elite obstruction answer different questions: who ought to authorize politics, and who is said to prevent that authority from operating? Separating them preserves an interpretable distinction that a single generic populism similarity score would obscure. Communication research also distinguishes combinations of populist content and documents fragmented expressions of its components. [Jagers and Walgrave 2007][R14]; [Engesser et al. 2017][R15]

This leads to two legitimate measurement targets:

- **Full local people-versus-elite rhetoric:** use the P/E composite.
- **A component such as anti-elitism or people-centrism:** evaluate the matching dimension, with a component-specific interpretation.

Success on the second target is evidence about that component. It does not, by itself, validate the first. The distinction matters for the tweet, PopBERT and Portuguese benchmarks.

A direct local-co-occurrence precedent appears in Rooduijn, van der Brug and de Lange: their coding identifies populist paragraphs through the joint presence of people-centrism and anti-elitism. The current method generalizes that conjunction into a continuous window score; it changes the subsequent document aggregation. [R47] (p. 35)

### 2.3 What discourse theory adds

Laclau's account makes the construction of a people and an antagonistic frontier central to populism. This supports looking for a relational organization of meaning rather than a fixed list of political topics. It also explains why actors such as bankers, bureaucrats or party insiders can occupy an elite role in different settings. The embedding axis is an operational approximation to recurring semantic contrasts; it does not implement Laclau's theory of equivalential chains, political identification, or hegemony. [Laclau 2005][R16]

Aslanidis's framing approach provides a closer bridge to measurement: a populist frame can be identified in discourse without first establishing that a speaker possesses a stable ideology. His later semantic-analysis proposal makes this bridge explicit through coded propositions. [Aslanidis 2016][R17]; [Aslanidis 2018][R05]

### 2.4 The contrast poles are substantive choices

The positive and negative poles do different work. Positive anchors describe the target claims. Negative anchors determine which alternatives the projection treats as distant from those claims, so they are part of the construct.

The P contrast mixes constitutional mediation, pluralist deliberation and expert authority. The E contrast mixes institutional accountability with praise of government performance. The M contrast mostly expresses legitimate disagreement and good-faith opposition. These are recognizable alternative political registers, but none is the complete logical complement of populism.

Caramani adds a qualification: populism and technocracy differ over authority while sharing a tendency to posit a unitary general interest, whereas party government accommodates plural interests. Bickerton and Invernizzi Accetti likewise analyze their shared challenge to party democracy. Expert language therefore supplies a meaningful P contrast without functioning as a universal nonpopulist class. [R18] [R19]

### 2.5 Theory-guided computational measurement

Hurtado Bodell et al. distinguish data-first, theory-first and theory-data integration approaches to computational text analysis. In their theory-first account, researchers specify the categories or relationships to measure before applying a classifier. This provides a clear methodological location for the written P/E construct: the substantive target is specified by researchers, while pretrained representations supply the computational resources. The project's later empirical choices remain part of its development history. The review also treats validation as necessary when no task-specific training set is required. [R52] (sections 2.2-2.3)

The corresponding measurement claim concerns expressed rhetoric, not a speaker's unobserved psychological disposition. Deng and Li's WEPA study likewise interprets its text projections as expressed construct salience. Its reflective psychological-indicator model and additive treatment of self-efficacy dimensions differ from the present conjunctive definition. What transfers is the procedure, theory-guided anchors followed by an explicit projection rule, not the theory of the measured construct. [R49] (pp. 3, 6, 9)

Boutyline and Arseniev-Koehler make the semantic rationale more precise: measure a feature by constructing multi-keyword endpoint scales and subtracting them. Jointly selecting contrasts can cancel unrelated meanings; asymmetric endpoint content can bias the direction. This directly supports treating both proposition poles as substantive construct decisions. Their recommendation concerns word-based scales, so moving to sentences remains an extension. [R36] ("Measure Features of Concepts, Not Their Contexts")

### 2.6 Adjacent approaches, adjacent constructs and the other measurement traditions

Three conceptual alternatives to the ideational approach define what the measure does not claim. Weyland's political-strategic definition locates populism in a personalistic leader's unmediated mobilization of support; Moffitt and Tormey's account locates it in a performed political style; Laclau's, discussed above, in a logic of articulation. A text measure of claims cannot observe organization or performance, so agreement with holistic coders who weigh those features is bounded from above. [R60] [R61] [R16]

Two adjacent constructs explain the discriminant failures recorded in the development log. Mudde and Rovira Kaltwasser distinguish exclusionary populism, which adds nativist boundaries to the people, from inclusionary populism, which does not; the exclusion is host-ideology content, not populism. Hunger and Paxton's review of 884 abstracts finds that political science routinely conflates populism with what it travels with. The PE axis's strong response to the Portuguese exclusionism category is this conflation appearing in an embedding space: hostility toward an out-group and opposition to an elite share a register that the encoder does not separate. Stating the construct as vertical people-versus-elite opposition, and reporting the exclusionism response as a limit, is the measurement-validity answer to that finding. [R62] [R63] [R37]

The measure also belongs to one of three measurement traditions that rarely meet. Text measures, from holistic grading to classifiers, score what actors say. Expert surveys such as the POPPA survey score what experts believe parties are, on multiple dimensions. Attitude scales score what citizens think, and Castanho Silva et al. show that seven such scales differ in dimensionality and cross-national validity. Olivas Osuna and Rama document that supply-side and demand-side instruments were built without regard to each other, so the assumed correspondence between populist attitudes and populist voting is partial and country-specific. A text measure with a stated two-component construct is easier to align with the multidimensional survey instruments than a single holistic score, which is one argument for reporting P and E separately. [R64] [R65] [R66]

## 3. From propositions to a formal measurement model

### 3.1 A conceptual specification

The following notation is a reconstruction for this report, not a formula claimed by the cited theorists. For a speaker's passage \(w\), let:

\[
P^*(w)=\text{the passage advances the people's overriding political authority},
\]

\[
E^*(w)=\text{the passage advances elite obstruction or betrayal of that people}.
\]

A minimal local target is

\[
C^*(w)=P^*(w)\land E^*(w).
\]

The interpretation additionally requires that the claims concern compatible people/elite referents and are attributable to the speaker. A strict moralized version would also require moral polarization or an equivalent constraint on the meanings of \(P^*\) and \(E^*\). The present inventory largely takes the latter route, embedding moral language in its propositions.

This specification explains three design requirements: inspectable P and E evidence, a preference for their local combination, and sensitivity to whether a proposition is actually being advanced. The system approximates all three. It does not explicitly bind referents, extract semantic roles, or resolve every attribution.

### 3.2 A close formal predecessor

Aslanidis operationalizes populist discourse using clause-level subject-verb-object relations and distinguishes full from partial frames. In simplified notation, his index is

\[
\mathrm{PDI}=\frac{N_{\mathrm{full}}+c(N_{P\mathrm{-only}}+N_{E\mathrm{-only}})}{N_{\mathrm{clauses}}},
\qquad c=\tfrac13.
\]

Full frames connect the relevant people and elite roles; partial frames receive reduced weight. This directly precedes the idea of preserving components while giving their combination special status. The repository replaces explicit clause coding with continuous semantic and NLI signals. It also uses top-window concentration instead of clause-frequency normalization, so it estimates a different document property. [R05] (sections 4-6)

Popping's account of semantic text analysis adds modality and reasons to predicate-centered representations and discusses context needed to interpret statements. This provides a language-method rationale for retaining assertions and their surrounding sentences. It does not establish that sentence embeddings recover all of these structures. [R06]

### 3.3 The exact axis

Let \(f(s)\) denote the pretrained E5 representation of sentence or window \(s\), and let \(\widehat f(s)=f(s)/\|f(s)\|\). For each \(c\in\{P,E,M\}\), the code first normalizes the individual anchor embeddings and forms

\[
\mu_c^+=\frac1{10}\sum_{i=1}^{10}\widehat f(a_{ci}^+),\qquad
\mu_c^-=\frac1{10}\sum_{i=1}^{10}\widehat f(a_{ci}^-),
\]

\[
d_c=\frac{\mu_c^+-\mu_c^-}{\|\mu_c^+-\mu_c^-\|}.
\]

For windows, the raw embedding corpus mean \(\bar f_{\mathcal C}\) is subtracted before normalization:

\[
x_w=\frac{f(w)-\bar f_{\mathcal C}}{\|f(w)-\bar f_{\mathcal C}\|},\qquad r_c(w)=x_w^\top d_c.
\]

The projection is exactly a scaled contrast in average similarity:

\[
r_c(w)=\frac{
\frac1{10}\sum_i x_w^\top\widehat f(a_{ci}^+)
-\frac1{10}\sum_i x_w^\top\widehat f(a_{ci}^-)
}{\|\mu_c^+-\mu_c^-\|}.
\]

The identity states what the axis measures: how much more the centered passage representation aligns with the target propositions than with their written alternatives. An anchor's contribution to the numerator can be inspected. The encoder's representation remains learned and opaque.

Kozlowski et al., SemAxis and FrameAxis establish related projection and contrast constructions using word embeddings. Applying that geometry to complete political propositions in a contextual sentence space is this project's extension. Grand et al. provide complementary evidence that semantic projection can recover human feature judgments in other domains. None of these studies establishes the validity of these particular populism anchors. [R08] [R09] [R10] [R20]

Deng and Li's WEPA is a close implementation precedent. Their WEPA equations define

\[
\mu_\pm=|A_\pm|^{-1}\sum_{a\in A_\pm}e(a),\qquad
d=\frac{\mu_+-\mu_-}{\|\mu_+-\mu_-\|},\qquad
s(T)=\left(|W_T|^{-1}\sum_{w\in W_T}e(w)\right)^\top d.
\]

Here \(e\) is a word embedding and \(W_T\) the text's token multiset. This is the same normalized centroid-difference and projection structure. WEPA uses domain-trained static GloVe; its equations do not specify this project's individual-anchor normalization or centered, normalized windows. It supplies a close methodological precedent without validating the present political propositions. [R49] (section 2.5.1, p. 6, equations 1-4)

Two recent studies with Boutyline extend the evidence into transformer models, using different representations. Kozlowski, Dai and Boutyline (2025) examine fixed token embedding matrices. They compare 301 words on 28 semantic scales with human ratings and identify meaningful relationships among semantic directions. Their formula averages individually normalized antonym-pair differences,

\[
d_f=\frac1N\sum_{j=1}^{N}\frac{a_j^+-a_j^-}{\|a_j^+-a_j^-\|},\qquad N=10.
\]

This is related to, but not identical with, the present unit-anchor centroid construction: normalizing each offset changes its relative weight. Their whitening experiment also shows that an apparently useful geometric transformation can weaken correspondence with human judgments; it does not test this project's simpler corpus-mean subtraction. [R54] (sections 3.2-4.1, equation 1, Figure 3)

Kozlowski and Boutyline (2026) instead extract contextual features from transformer hidden states. Their axis is the mean of ten unnormalized positive-minus-negative feature vectors, algebraically a difference of centroids. They compare projections of 360 words on 32 semantic scales with human ratings and relate axis cosines to human interscale correlations. This supports the possibility of meaningful, nonorthogonal semantic directions beyond static word embeddings. Their Llama-based feature extraction and Qwen replication do not test E5 sentence representations or BART entailment. Both studies are preprints. They use different word/scale subsets of the same underlying human-rating survey, so their validation evidence overlaps despite the different model representations and extraction procedures. [R50] (sections 2.2-2.3 and 3, equation 1, Figures 3-4)

For this project, nonorthogonality is therefore an empirical question about shared content. Related P/E/M directions could reflect a coherent discursive configuration, shared evaluative tone, or both. The cited studies motivate inspecting those alternatives; they do not decide between them. Making the axes orthogonal would change which semantic contrasts the projections measure, while retaining them does not establish discriminant validity.

Multiple anchors make each pole broader than a single slogan. Their arithmetic mean also assigns equal formal weight to every item and allows semantically redundant items to dominate a narrow region of meaning. Anchor coverage and axis reliability therefore need assessment in addition to agreement with document labels; section 3.6 separates these questions.

### 3.4 What the activation and conjunction do

The current PE score uses within-corpus standardized projections \(z_c(w)\), temperature \(t=.25\), and activation

\[
B_c(w)=\left[\log\left(1+e^{z_c(w)/t}\right)\right]^2,\qquad
s_c=\operatorname{median}_{u\in\mathcal C:B_c(u)>0}B_c(u),\qquad
A_c(w)=B_c(w)/s_c.
\]

The window score is

\[
G(w)=\sqrt{A_P(w)A_E(w)}
=\exp\left\{\tfrac12\log A_P(w)+\tfrac12\log A_E(w)\right\}.
\]

For a fixed activation sum, the geometric mean is maximized when the two dimensions are balanced. This gives a precise rationale for preferring passages with evidence on both dimensions over equally strong but concentrated evidence on one dimension. The positive softplus also preserves rankings among weak passages.

The rule is therefore a soft conjunction that rewards balance. Wuttke et al. argue that populism's concept structure requires attention to joint component presence; they distinguish a strict minimum-based operationalization from compensatory aggregation, including geometric aggregation. Their theory motivates conjunction but cannot be cited as proving this implementation is noncompensatory. [R07] (pp. 359-361)

For example, \(\sqrt{100\times .01}=\sqrt{1\times1}=1\): a sufficiently strong activation can compensate for a weak one. Softplus activations are positive and unbounded, so the rule has no exact logical veto. Even on \([0,1]\), \(\sqrt{ab}\) is not a fuzzy t-norm: \(\sqrt{a\cdot1}=\sqrt a\ne a\). It is also not the probability that two claims are jointly true.

The squaring, median scaling and temperature are implementation decisions. Theory does not determine them. They produce smooth, comparably scaled dimension intensities and keep the aggregation rule readable.

### 3.5 Context and document aggregation

The implementation forms overlapping three-sentence windows, stride one, after chunking sentences longer than 60 words. E5 uses the same `query: ` rendering for anchors and windows and a 256-token limit. The window keeps a claim and its local completion together. Tóth, Goyanes and Demeter provide a close precedent for this window length. They compare sentences, three-sentence units, and paragraphs. Their middle unit places one sentence on either side of a focal sentence to recover context and pronoun referents. Among dictionary-selected candidate units from 18 Orbán speeches, the share coded as explicit populism more than doubles between the sentence and three-sentence units. Their study supports this contextual design while recommending analysis at several levels; it does not establish a universally optimal window. [R22] (sections 4-7)

For document \(D\), let \(I_K(D)\) index the \(\min(K,|D|)\) highest-scoring windows under \(G\). Then

\[
S_A(D)=\frac1{|I_K(D)|}\sum_{w\in I_K(D)}G(w),
\]

with \(K=8\) for speeches/paragraphs and \(K=1\) for sentence/tweet scoring.

The resulting estimand is the average strength of the document's most salient detected populist passages. Sparse local expressions make this plausible, and pooling several windows reduces dependence on a single peak. It is not the proportion of a speech that is populist. Overlapping windows may repeatedly represent the same passage, and longer documents have more opportunities to produce high values.

The GPD rubric considers how consistently populist discourse organizes a speech, including whether pluralist elements temper it. That motivates attending to multiple passages, but does not prescribe strongest-window pooling; holistic pervasiveness and fixed top-K intensity remain different targets. [GPD codebook][R23] The repository's development grid also identifies pooling depth, rather than window width alone, as the clearer source of improvement. The value eight was selected empirically.

### 3.6 Anchor reliability and the structure of the measure

Reliability concerns how consistently an anchor set defines its intended direction. Boutyline and Johnston's *Forging Better Axes* formalizes parallelism as agreement among offsets from paired opposing anchors:

\[
\operatorname{Par}(S)=\frac{1}{n(n-1)}\sum_i\sum_{j\ne i}
\cos(z_i-a_i,z_j-a_j),
\]

where \((a_i,z_i)\) are \(n\) opposing anchor pairs. This gives an internal-reliability rationale for multiple indicators of a direction. Their published results favor parallelism over synonymy and antonymy for predicting agreement with human ratings, but identify its strongest use in selecting tractable dimensions and embedding models; guidance for choosing individual keywords is relatively weak. The formula is verified against the 2025 manuscript and 2026 replication, with the interpretation checked against the published abstract. [R21] (manuscript pp. 12-13, equations 6-8)

There is an important distinction for this implementation. Let \(p_i\) and \(n_i\) denote normalized vectors in the positive and negative anchor poles. For equally sized poles and any permutation \(\pi\),

\[
\frac1m\sum_{i=1}^{m}(p_i-n_{\pi(i)})=\bar p-\bar n.
\]

Thus the implemented centroid difference does not require the two arrays to be matched sentence pairs. A diagnostic based on individual pair directions would require a defensible pairing or a stated pairing procedure. The identity above is an algebraic observation about this project, not evidence that every pairing has the same diagnostic quality. No reliability coefficient for these proposition anchors has been established by the cited word-anchor studies.

Taylor et al. supply a complementary anchor reliability coefficient, **relco**. They divide anchor lists into subsets, construct simulated documents with varying anchor content, and regress the resulting concept-engagement scores on an anchor-inclusion score. Repeated random partitions yield a mean slope, with uncertainty and word-contribution diagnostics. In their inspected preprint, the core regression is

\[
\widehat c_d=\widehat\alpha+\widehat\beta x_d,\qquad
\operatorname{relco}=\frac1B\sum_{b=1}^{B}\widehat\beta_b.
\]

Here \(x_d\) is the scaled anchor-inclusion score; notation is simplified from their equations 3-4. This motivates testing whether an axis responds consistently to controlled changes in construct-relevant content. Their word-based simulation would need substantive adaptation for political propositions, since token replacement can alter assertion, negation and attribution. It is a validation precedent, not a test already passed by this system. [R51] (preprint pp. 13-14)

Pokropek provides a different formal diagnostic: confirmatory factor analysis of similarities among selected word embeddings, with model-implied structure \(\Sigma=\Lambda\Psi\Lambda^\top+\Theta_\epsilon\). Here \(\Lambda\) contains loadings, \(\Psi\) factor covariances and \(\Theta_\epsilon\) residual variances/covariances. Model fit tests the hypothesized internal structure; multigroup models address comparability. The paper centers each word vector across its coordinates before calculating similarities, which differs from this project's subtraction of a corpus mean vector. Its reflective latent-factor model also differs from the P/E conjunction. It supports explicit tests of internal structure and invariance, without making the current axes latent factors or establishing their validity. [R53] (sections 4-6 and 8.6)

### 3.7 Document-level projection precedents and recent validity work on embedding measures

The closest sociological precedent for scoring whole documents on a written direction is Taylor and Stoltz's integration of semantic directions with concept mover's distance. Their problem was that words for opposing concepts sit close together in an embedding space, so a document's closeness to "life" and to "death" is nearly the same; projecting on a direction between the two poles restores the contrast. That is the word-level version of the present design, in which a passage is scored on the direction between populist and mainstream propositions rather than on its similarity to populist propositions alone. [R102] Duan et al.'s vec-tionaries move the other way, from a validated dictionary to an embedding-weighted measure of moral appeals with valence and ambivalence scores; they show that a dictionary's content can be carried into an embedding space without retraining. [R103] Luongo, Marocco and Milano's clinical study is the nearest recent implementation: items from validated depression, anxiety and worry scales define semantic axes in a Sentence-BERT space, texts are projected on them, and the projections are validated against the clinical instruments with split-half reliability and attenuation corrections. Their scale items play the role the proposition anchors play here. [R107]

Four recent papers state the validity requirements such measures face. Fang, Nguyen and Oberski apply the classical construct-validity framework to text embeddings and find that convergent and discriminant validity differ by model, so validity must be checked per embedding, not assumed. [R104] Rodriguez and Spirling show that pretrained embeddings match or exceed human coders on political tasks, and give the applied researcher's checklist for choosing among them. [R105] Li et al. name the "proxy presumption", the reliance on geometric properties such as cosine distance as direct measures of social concepts, and propose a construct validity protocol with discriminant, incremental and predictive tests and a counterfactual neutralization step against confounds such as topic and style. [R106] Plisiecki argues that embeddings optimized for prediction entangle meaning with non-semantic signal and proposes geometry-first and post-hoc transformations for scientific use. [R108] Erk and Apidianaki show that seed-based interpretable dimensions can be improved with human ratings when the seeds alone do not work well. [R109] Taken together, these papers say what section 7.3 says: the axis needs discriminant tests against tone and out-group hostility, reliability tests over anchor subsets, and a documented confound analysis, and the development log's stages 2 and 3 are the first entries in that record rather than the last.

## 4. Why the NLI stage belongs in the method

### 4.1 Similarity and textual support are different relations

A sentence can discuss elite corruption while rejecting the allegation, quoting an opponent, reporting somebody else's belief, or describing a hypothetical case. Such passages can remain near corruption-related anchors in a semantic space. The distinction between a text resembling a proposition and supporting it is the motivation for textual entailment as an NLP task. [R11]

Boutyline and Arseniev-Koehler illustrate the same measurement problem: a word can become closer to both intelligence and unintelligence because that characteristic is discussed more often. Proximity then fails to establish which attribute is assigned. Explicit propositions and a separate NLI score address an analogous inferential gap here. That connection is our interpretation, not a tested NLI result from their review. [R36] ([section 14](https://www.annualreviews.org/content/journals/10.1146/annurev-soc-090324-024027#sec14))

Yin et al. show how natural-language hypotheses can specify classes for zero-shot classification. Here the hypotheses do more: they expose the political claim being tested. This is a strong reason to use written sovereignty and obstruction propositions instead of an opaque class identifier. [R12]

Pragmatics adds a necessary qualification. Commitment is sensitive to the source of a claim and to context, modality and embedding. A standard NLI prediction is an approximation to this problem; it is not a complete model of political endorsement. [R13]

### 4.2 The exact implemented check

The premise is a PE-axis-selected window. The hypotheses are the five positive P anchors and five positive E anchors closest to their corresponding axes. Selection was geometric and label-free. It was not an independent literature-based judgment of which five propositions best cover each dimension.

For hypothesis \(h\), the BART-MNLI implementation retains all three output classes:

\[
q(w,h)=\frac{e^{\ell_{\rm entailment}(w,h)}}{
e^{\ell_{\rm contradiction}(w,h)}+e^{\ell_{\rm neutral}(w,h)}+e^{\ell_{\rm entailment}(w,h)}}.
\]

This differs from zero-shot implementations that discard the neutral logit and normalize only entailment against contradiction. Here lack of evidence can remain neutral, which is appropriate for strong substantive claims. MultiNLI supplies the three-way training task; BART supplies the pretrained architecture subsequently fine-tuned on it. [Williams, Nangia and Bowman 2018][R24]; [Lewis et al. 2020][R25]; [BART-MNLI model card][R26]

Within dimensions,

\[
Q_c(w)=\max_{h\in H_c}q(w,h).
\]

The maximum treats the hypotheses as alternative routes to expressing one component. A passage need not entail every sovereignty formulation or accuse every possible elite. It is a prototype-pooling convention, not a calibrated probability of the disjunction of the hypotheses. The selected hypotheses are related claims with different logical strength, not interchangeable paraphrases. Laurer et al. give a social-science precedent for operationalizing a coding scheme through class hypotheses. [R32] (section 3.1.2) Burnham et al.'s Political DEBATE models extend that precedent: DeBERTa entailment classifiers trained on a political NLI corpus reach zero- and few-shot performance on stance, topic, hate-speech and event tasks at laptop speed, which establishes NLI as a practical instrument for political text at the scale of this project. The claim-check replication found that three general-purpose DeBERTa models order the low tail differently from BART; a politically tuned entailment model is the natural next candidate and is listed in section 7.3. [R101]

Across dimensions and then across the axis-selected windows,

\[
Q_{PE}(w)=\sqrt{Q_P(w)Q_E(w)},\qquad
S_N(D)=\frac1{|I_K(D)|}\sum_{w\in I_K(D)}Q_{PE}(w).
\]

The architecture thus uses alternatives within a component and joint support across components. The geometric mean retains the qualification discussed above.

Finally,

\[
S(D)=\tfrac12 Z_{\mathcal C}(S_A(D))+\tfrac12 Z_{\mathcal C}(S_N(D)).
\]

This is equal-weight fusion of two standardized document signals. The NLI stage is not a veto: low NLI support does not prevent a high axis score from contributing. The two models also share the same written construct and candidate passages, so they are complementary measurements rather than independent validators.

### 4.3 What the existing NLI evidence establishes

The development report records median pairwise entailment near .001 and a 99th percentile near .064. Fewer than 0.3% of windows exceed .5 on either dimension, and none exceeds .5 on both. The useful signal is therefore mainly the relative ordering of low scores. We therefore describe it as an NLI-based claim-support score. Saying that BART confirms these claims would be stronger than the outputs warrant.

Low entailment is compatible with the specificity of the anchors. A passage criticizing an institution need not entail that every reform is blocked; a passage praising voters need not entail that no authority stands above them. This is a plausible explanation, not a demonstrated causal diagnosis of the low scores.

The benefit over axis-only scoring on several corpora supports complementary predictive information. It does not show which linguistic mechanism supplied the gain. Likewise, the weaker replication by three DeBERTa alternatives establishes model dependence in this task; it does not establish that BART has the best general understanding of populism or the best-calibrated entailment probabilities.

### 4.4 Why the two-stage architecture is coherent

The axis offers an inspectable contrast score and selects candidate passages. The NLI model evaluates those passages against explicit statements. This division can be understood as candidate retrieval followed by pairwise assessment. Here the purpose is construct measurement, so the second-stage hypotheses are substantive claims. Sentence-BERT provides a precedent for efficient sentence-embedding comparison; this project adds a cross-encoder judgment of each selected premise-hypothesis pair. [Reimers and Gurevych 2019][R27]

The theoretical claim is that association and claim support are complementary evidence about a discursive construct. The strongest empirical claim is that their frozen combination transfers usefully across several operationalizations. Neither claim requires interpreting the final score as a calibrated probability.

## 5. Every anchor and its theoretical support

IDs below use zero-based positions, matching the NLI hypothesis IDs. `+` identifies the populist pole; `−` identifies the file's `mainstream` pole. A dagger marks a positive anchor used by NLI. **Direct** means the proposition expresses a recognized conceptual element. **Extension** means its phrasing or example is an operational interpretation. **Conditional** means it needs contextual information to distinguish populism from ordinary democratic, oppositional or moral language. These ratings concern theoretical fit, not measured classification accuracy.

### 5.1 People-sovereignty propositions

| ID | Exact proposition in the anchor file | Support and qualification |
|---|---|---|
| P+00† | The ordinary people are the only true source of legitimate political power. | Direct: popular authority; especially close to Bonikowski and Gidron. [R03] (abstract) [R01] |
| P+01† | It is the people, not the officials, who must decide the future of this country. | Direct: decision rights of citizens over officials; close to survey item POP2. [R29] (Table 1) |
| P+02 | We are here to give the country back to the people it belongs to. | Extension: restoration of authority to the people, in the register of Taggart's heartland, the populist image of a place that belongs to a virtuous population. National ownership alone remains ambiguous. [R28] (p. 104) [R57] |
| P+03 | The will of the ordinary citizen must come before the preferences of any institution. | Direct: popular will takes priority over institutional mediation. [R01] (p. 561) |
| P+04 | Nobody has the right to overrule what the people have decided. | Direct: rejection of restraints on the popular decision. This does not itself allege elite wrongdoing. [R01] (p. 561) |
| P+05† | This movement is nothing more and nothing less than the voice of ordinary people. | Conditional: authentic representation. It does not explicitly deny every rival the right to represent the people. [R18] [R31] |
| P+06† | Power must be returned to the hands of the people who built this nation. | Extension: restitution to a deserving popular subject; "the people who built this nation" is Taggart's heartland, the laborious and morally pure population of the populist imagination. [R28] [R30] [R57] |
| P+07 | The common people know perfectly well what is good for them. | Conditional: popular common sense and competence; also compatible with ordinary democratic confidence. [R30] [R18] |
| P+08† | Our people are sovereign and no authority stands above them. | Direct: unrestricted popular sovereignty. [R01] (pp. 543, 561) |
| P+09 | What the ordinary working family wants is what this government will do. | Conditional: responsiveness to ordinary families; exclusivity and a unified will are only implicit. [R29] (POP1) [R18] |

### 5.2 Elite-obstruction propositions

| ID | Exact proposition in the anchor file | Support and qualification |
|---|---|---|
| E+00† | A corrupt elite has captured our institutions and turned them against the public. | Direct: corrupt capture directed against the public. [R02] (p. 1044) |
| E+01 | The establishment in the capital works only for itself and its donors. | Direct: collective elite self-interest; donors are a contextual example. [R28] (pp. 103-104) |
| E+02† | Bankers, lobbyists and career politicians have rigged the system against you. | Direct: elite sectors jointly obstruct the public through a rigged system. [R02] (rubric) |
| E+03 | A small privileged clique makes every decision behind closed doors. | Extension: generalized power monopoly and secrecy; moral betrayal is less explicit. [R28] (pp. 103-104) |
| E+04† | The people in power have betrayed the country they were elected to serve. | Conditional: betrayal of representation, the blame-attribution form of populist communication; ordinary opposition can use the same accusation. [R18] (p. 63) [R58] |
| E+05† | Entrenched insiders block every reform that would help ordinary families. | Direct: systematic obstruction of ordinary people by insiders. The universal “every” makes the claim unusually strong. [R28] (p. 104) |
| E+06† | They enrich themselves while the rest of the country is told to sacrifice. | Conditional: self-enrichment against public sacrifice, blame attributed to an in-group's expense, provided “they” identifies an elite. [R30] [R58] |
| E+07 | An unelected bureaucracy overrules whatever the voters ask for. | Extension: bureaucratic displacement of voters; corruption is not explicitly asserted. [R01] (p. 561) [R18] (Table 2) |
| E+08 | The political class protects its own privileges above everything else. | Direct: a generalized political class protecting its interests. [R28] (p. 104) |
| E+09 | Powerful interests have bought the influence that should belong to citizens. | Direct: purchased influence displaces citizens. This instantiates capture; it is not a validated survey item. [R02] (p. 1044) |

### 5.3 P contrast propositions

| ID | Exact proposition in the anchor file | Support and qualification |
|---|---|---|
| P−00 | Complex policy questions require careful analysis by qualified specialists. | Extension: expert justification, not an assertion that citizens lack political authority. [R18] (Table 2) |
| P−01 | Our constitutional institutions provide the proper framework for these decisions. | Direct contrast: constitutional mediation of decisions. [R23] (pluralist rubric) |
| P−02 | This government will follow the recommendations of the independent commission. | Extension: delegated expert advice. Populist governments can also follow commissions. [R18] [R19] |
| P−03 | Sound policy is developed through deliberation among many stakeholders. | Direct contrast: plurality of stakeholders and deliberation. [R18] (Table 2) |
| P−04 | We must respect the procedures that our legal system prescribes. | Direct contrast: binding legal procedure. [R23] (pluralist rubric) |
| P−05 | Expert advice has guided our approach throughout this process. | Extension: expertise as a reason for policy. Advice can coexist with populist legitimation. [R18] [R19] |
| P−06 | The relevant ministries have reviewed the evidence and issued their guidance. | Extension: administrative evidence and guidance; mainly an institutional register. [R18] |
| P−07 | Representative institutions exist precisely to weigh these competing interests. | Direct contrast: competing interests mediated by representatives. [R18] (Table 2) |
| P−08 | We will proceed in accordance with established regulatory practice. | Extension: procedural regularity; limited evidence about the speaker’s wider worldview. [R23] |
| P−09 | A technical assessment will determine the appropriate course of action. | Extension: technical expertise determines action. A defensible technocratic contrast. [R18] (Table 2) |

### 5.4 E contrast propositions

| ID | Exact proposition in the anchor file | Support and qualification |
|---|---|---|
| E−00 | Our public servants have worked diligently to implement this programme. | Conditional contrast: competent public service. Institutional praise is not required by nonpopulism. [R02] (pluralist rubric) |
| E−01 | The agency has met its obligations and reported transparently on its work. | Extension: accountable administration counters indiscriminate institutional demonization. [R23] |
| E−02 | Officials across government cooperated effectively on this initiative. | Conditional contrast: cooperative administration; largely positive evaluation. [R02] (pluralist rubric) |
| E−03 | The civil service continues to deliver essential services to the public. | Conditional contrast: public service as beneficial; an operational contrast with obstruction, not a separate theoretical criterion. [R02] (pluralist rubric) |
| E−04 | Regulators acted within their mandate and published their findings. | Direct contrast in institutional legitimation: bounded mandate and public justification. [R23] |
| E−05 | Members of this chamber have debated the proposal in good faith. | Direct pluralist contrast: representatives deliberate in good faith. [R02] (pluralist rubric) |
| E−06 | The commission consulted widely before publishing its recommendations. | Extension: consultation with multiple constituencies. [R18] (Table 2) |
| E−07 | Our institutions have proved resilient and accountable in difficult times. | Extension: institutional accountability rather than capture. [R23] |
| E−08 | Public bodies are subject to audit and parliamentary oversight. | Direct institutional contrast: public oversight and accountability. [R23] |
| E−09 | The department has responded constructively to the concerns raised. | Conditional contrast: responsive administration; an evaluative opposite to obstruction. [R02] (pluralist rubric) |

### 5.5 Diagnostic moral-antagonism propositions

| ID | Exact proposition in the anchor file | Support and qualification |
|---|---|---|
| M+00 | This is a battle between honest people and a thoroughly corrupt establishment. | Direct: explicitly moral people-versus-establishment division. [R02] |
| M+01 | They lie to your face and they have been lying for decades. | Conditional: generalized deceit; “they” needs a political elite referent. [R02] (rubric); context caveat [R13] |
| M+02 | These people are not merely mistaken, they are morally rotten. | Direct moral-essentialist form, conditional on whom “these people” denotes. [R31] |
| M+03 | It is time to drain the swamp and throw out the crooks. | Conditional: expulsion of a corrupt group; a slogan needs actor/context interpretation. [R02] |
| M+04 | Their greed and their treachery have brought this country to its knees. | Conditional: treachery and greed explain collective harm; targets remain implicit. [R28] |
| M+05 | We are the decent ones and they are the parasites feeding on us. | Direct polarized moral identification, conditional on the vertical people/elite roles. [R30] |
| M+06 | Everything they say is a fraud designed to deceive ordinary citizens. | Conditional: sweeping deception against citizens; source and target require context. [R02] (rubric); context caveat [R13] |
| M+07 | They have sold out the nation for their own miserable profit. | Conditional: national betrayal for profit can also target a foreign adversary. [R30] |
| M+08 | There can be no compromise with people this dishonest. | Direct anti-compromise form; an ordinary corruption allegation can still be context-specific. [R29] (POP7, Table 1) |
| M+09 | Their corruption is not an accident, it is who they are. | Direct essentialized corruption; actor scope determines whether this is anti-elitism. [R31] |

### 5.6 M contrast propositions

| ID | Exact proposition in the anchor file | Support and qualification |
|---|---|---|
| M−00 | Reasonable people can disagree about the best way forward. | Direct pluralist contrast: legitimate disagreement. [R02] (pluralist rubric) |
| M−01 | I want to acknowledge the constructive contribution of the opposition. | Direct pluralist contrast: constructive opposition. [R23] |
| M−02 | We should seek common ground wherever it can honestly be found. | Direct pluralist contrast: legitimate compromise. [R29] (pluralism items) |
| M−03 | My colleagues hold a different view and they argue it sincerely. | Direct pluralist contrast: sincere competing views. [R02] (pluralist rubric) |
| M−04 | This is a difficult trade-off with legitimate arguments on both sides. | Direct pluralist contrast: legitimate alternatives and trade-offs. [R18] (Table 2) |
| M−05 | I thank the honourable member for raising a serious point. | Conditional: civility alone does not establish pluralism. [R04] |
| M−06 | Our disagreements are real but they are conducted in good faith. | Direct pluralist contrast: disagreement without moral exclusion. [R02] (pluralist rubric) |
| M−07 | Let us debate the substance rather than question anyone's motives. | Extension: substantive debate over motive attribution. [R04] |
| M−08 | Both parties share a commitment to the wellbeing of this country. | Direct pluralist contrast: shared concern despite party competition. [R18] |
| M−09 | We will continue to work with all sides to reach a durable agreement. | Direct pluralist contrast: cooperation across legitimate sides. [R23] |

### 5.7 What this mapping establishes

The positive inventory has recognizable theoretical content throughout, with direct support for P+00/P+01/P+03/P+04/P+08 and E+00/E+02/E+05/E+08/E+09. Several other items are plausible manifestations whose specificity depends on context. The sentence wording is the project’s own operationalization.

The contrast inventory is a set of substantive comparison poles. P contrasts popular primacy with mediated or expert decision-making; E contrasts elite obstruction with institutional service; M contrasts moral exclusion with legitimate disagreement. The E contrast is the most vulnerable to conflating construct content with favorable versus unfavorable evaluation.

No item was found to be an established, externally validated text-classification anchor. Survey-item precedents support content, not interchangeability between attitudes and rhetoric. In particular, the interest-group item POP8 in Akkerman et al. was considered but removed from their retained scale; it should not be presented as validation of E+09. [R29]

Müller’s exclusive-representation criterion is a useful boundary for P+05: claiming to voice ordinary people is weaker than claiming that only this movement can represent them. The current P dimension therefore operationalizes popular primacy more clearly than a strict monopoly on moral representation. [R31]

Excluding M from the final score has a practical rationale: some moral content is already carried by P/E, while several M items are generic accusations without explicit elite roles. This does not establish that moral polarization is theoretically dispensable. The historical choice to exclude M was also empirical.

## 6. Construction history and what the results support

### 6.1 The actual development path

The [initial-method report](../research/development-log/00-initial-axis-method-and-results.md) describes a transition from static word-pair directions to proposition anchors in a frozen contextual space. Inspection had shown that abstract nominal phrases could activate participatory-democracy or professed-virtue language without the intended claim. The proposition inventory made the asserted relation explicit. Its initial P/E, soft-conjunction, top-three configuration was subsequently developed under the [stepwise protocol](../docs/development-protocol.md).

The later stages corrected segmentation and text cleaning, tested alternative geometry and matched-pair anchors, compared context/pooling rules and encoders, and added NLI. The original proposition anchors and E5 encoder were retained; the adopted window/pooling rule became three sentences and eight windows. Alternative geometry and revised anchors failed the development adoption rule. BART-MNLI supplied a complementary second-stage score. The .5 fusion weight was predeclared and retained. It should not be described as a parameter deduced from political theory or as a uniquely identified empirical optimum.

These choices can be motivated theoretically now without rewriting their history: the construct was specified from theory, and its measurement was then developed empirically. Labels did not train a populism classifier or directly estimate the anchor vectors, but development labels influenced representation and aggregation choices. “No populism-specific model training” is more precise than “no labels were used.” Both pretrained models themselves underwent substantial training, including inference-task training for BART-MNLI.

### 6.2 The benchmarks test related but different targets

| Benchmark | What the reference labels represent | Fit to this method |
|---|---|---|
| English Global Populism Database | Holistic speech-level assessment of populist discourse | Closest to the moral people/elite configuration; holistic consistency is broader than peak-passage intensity. [R02] [R23] |
| EU-OOD | Sentence-level anti-elitism, people-centrism, both, or neither | Direct test of named components and their co-occurrence. Sentences are English translations. [R38] |
| AgoraSpeech | Graded paragraph labels, initially generated by ChatGPT and then human-validated | Relevant paragraph-level criterion, with dependent human/AI annotation provenance. The repository retains 5,272 paragraphs. [R39] |
| US governors | Holistic speech scores using the same broad rubric family | Useful external transfer within a similar language, genre and construct. [R40] |
| Party tweets | Generalized anti-elite appeals, distinguishing general from specific targets | A component test whose required generalization is more specific than the current E signal. [R41] |
| PopBERT | Moralizing people-centric and anti-elite sentence labels | Close component validation; the evaluated route uses released English translations of German sentences. [R42] |
| Portuguese parliament | Separate people-centrism, anti-elitism, sovereignty and exclusionism indicators | Especially informative about construct overlap and mismatch. The primary route translates Portuguese to English. [R43] |

Chalkidis et al. explicitly distinguish the two components and their co-occurrence. The source is now published at EACL 2026; the repository cites its 2025 preprint. The final paper's table numbering differs, so manuscript citations should specify the version when quoting its comparator tables. [R38]

The Portuguese data separate these categories. Their separately coded sovereignty category fits the stronger P propositions better than broad references to beneficiaries such as “the Portuguese.” In the project's external report, P plus NLI reaches AUROC .832 for sovereignty, versus .695 for people-centrism. This is consistent with the chosen sovereignty construct, although only 23 sovereignty positives make it a limited diagnostic. It supports a precise P label more than a claim that P captures every definition of people-centrism.

### 6.3 Current ranking results

The table reports the retained corpus-standardized system. Values are taken from the [clean-run reference-scale reconstruction](../results/frozen_reference/results.md), the [development NLI report](../research/development-log/stage-6-claim-check.md), and the clean external evaluation artifacts listed below. They are existing results, not new model runs.

| Evaluation | Unit and evaluated n | Metric | Axis | Axis + NLI |
|---|---|---|---:|---:|
| English holdout | 84 speeches | Spearman | .417 | .518 |
| EU-OOD, both components versus neither | 1,597 of 1,762 sentences | AUROC, code 3 versus code 0 | .890 | .942 |
| AgoraSpeech, development | 5,272 paragraphs | Spearman | .359 | .382 |
| US governors 2010-2018 | 288 speeches | Spearman | .458 | .526 |
| Party tweets, general anti-elitism | 1,059 tweets | E AUROC | .754 | .756 |
| PopBERT, anti-elitism | 1,759 sentences | E AUROC | .781 | .806 |
| PopBERT, people-centrism | 1,759 sentences | P AUROC | .754 | .771 |
| Portugal, anti-elitism | 6,214 paragraphs | E AUROC | .770 | .798 |
| Portugal, people-centrism | 6,214 paragraphs | P AUROC | .668 | .695 |

The EU extreme-contrast AUROC excludes sentences with only one component. It should not be described as overall sentence-classification accuracy. The separate EU classification protocol uses component thresholds and is a different task. AUROC, Spearman and F1 are not interchangeable performance scales.

The clean external results are the strongest evidence. On US governors, the NLI gain is +.068 Spearman, with a paired cluster-bootstrap interval of [.031, .106]. The combined score is .526 [.438, .605]. It is not distinguishable from the released supervised SetFit speech ranking on their matched cases: the paired difference is +.006 [−.078, .093]. The supervised system remains stronger on binary classification. On Portugal, the NLI gains are +.029 [.021, .036] for anti-elitism and +.028 [.018, .043] for people-centrism. On general anti-elitism tweets, +.002 [−.007, .010] provides no clear evidence of improvement.

These results favor the two-stage score in several settings, without establishing universal superiority or identifying the linguistic cause of the gains. The tweet contrast suggests a boundary worth studying; brevity, fragmented expression, and the general-versus-specific coding distinction are plausible contributors, not experimentally separated explanations.

### 6.4 Exact component scoring matters

For single-component evaluations, the axis statistic pools standardized P or E projections, without the PE softplus conjunction. The NLI statistic takes the maximum entailment score over that component's hypotheses. The premises are still selected by the PE axis. This distinction has little effect on a literal one-window item but can matter for longer paragraphs or tweets. Consequently, the component benchmarks validate dimension readouts within the shared pipeline, not wholly independent P-only and E-only retrieval systems.

The tweet report supplies a useful discriminant boundary: E plus NLI reaches .818 AUROC for any elite criticism, but .756 for generalized anti-elite criticism. General and specific criticism are poorly separated by the axis. The paper's theoretically important generalization requirement is therefore only partly operationalized. In Portugal, the PE axis also responds strongly to exclusionism. Hostile outgroup talk and vertical elite opposition are not fully separated.

These are meaningful limitations for the theoretical case. They also explain why transferable performance can coexist with disagreement against a particular coding manual. A stable construct specification can be useful even when it deliberately covers a narrower or differently organized claim family. Measurement validity still requires stating that family accurately. [Adcock and Collier 2001][R37]

### 6.5 Centering, reference scales and transfer

E5's model documentation supports consistent input rendering and relative interpretation of similarities. General embedding research supplies precedents for investigating common components and anisotropy, including mean removal. It does not establish that the project's exact centering rule removes all unwanted variation. Anchors are individually normalized without centering; windows are centered and then normalized. [R33] [R34] [R35]

The clean reference-scale analysis gives encouraging portability evidence: freezing axis-level statistics on English development data leaves axis-only rank agreement at least .989 across the tested corpora. The combined score is more sensitive because document-level standard deviations alter the effective fusion weights. Critically, that analysis retains the original corpus-selected NLI premises. It tests fixed scaling more completely than it tests the entire pipeline after passage reselection.

The contrast between context and conceptual content discussed in sections 2.5 and 4.1 also constrains transfer claims. Stable rankings can preserve a stable nuisance association. Human ratings and matched discriminant cases remain necessary to determine what the score carries across corpora.

The evaluations of the fixed English configuration on translated text provide evidence of transfer through English representations. They do not establish multilingual invariance. The Portuguese report's original-language multilingual sensitivity route performs better, so translation and encoder choice remain consequential.

Both routes have precedents in political text analysis. De Vries, Schoonvelde and Schumacher show that machine translation preserves the term-document structure that bag-of-words models need for comparative work, and Licht shows that multilingual sentence embeddings give cross-lingual classifiers that are at least as reliable as translation-based ones when training data are scarce. The present measure uses each route on different benchmarks, translation for PopBERT, the tweets and the Portuguese primary route, and a multilingual encoder with the same English propositions for the Portuguese sensitivity route. Baden et al. list the dominance of English-language tools as one of three gaps in computational text analysis for social science; a construct written once and carried across languages by either route is one answer to that gap, and the Portuguese comparison of the two routes is the evidence that the answer is not yet complete. [R100] [R99] [R86]

### 6.6 Provenance and uncertainty boundaries

The English holdout was available during the initial configuration's development and was held out only from the later stepwise choices. EU-OOD also had earlier exposure. Both are spent for further selection. The four external benchmarks are stronger transfer checks, although the US corpus uses a related rubric and one presidential speech overlaps GPD; the headline governor subset avoids that particular overlap. Agora is development data with only six politician clusters and human labels corrected from AI annotations.

For the main ranking table, the clean reconstruction is recorded at `fddd3b8`, `dirty: false`; development NLI evaluation at `9cc1780`; and external final evaluations at `3814010` (US), `61f3cbf` (tweets), `aaa4efd` (PopBERT), and `244c25e` (Portugal), all with `dirty: false`. Some original final-evaluation manifests are marked dirty, including `results/final/final.manifest.json` and the B4 final manifest. `results/final/PROVENANCE.md` traces the dirty runs to the tracked code of `8f95b3c`; for the B4 and final manifests the only modified tracked file was the holdout access log, written by the same job. It does not recompute the B0/B4 axis scores. Accordingly, the ranking claims here rely on the clean reconstruction and clean external evaluations, not on an assertion that every historical final artifact has clean provenance. No new scoring was undertaken in this research task.

## 7. A defensible theory-to-method argument

The complete supporting argument has six steps:

1. **The construct is a configuration of political claims.** Specify popular primacy and elite obstruction before scoring texts. This follows the ideational/discursive family and avoids using party identity as the measurement rule.
2. **Claims require propositional content.** Write anchors with actors, relations, evaluation and modality. Semantic clause analysis provides a direct precedent; contextual encoders offer a scalable approximation.
3. **Meaning can be measured relative to explicit alternatives.** The centroid-difference axis has precedents in cultural measurement, theory-guided psychological measurement and transformer feature analysis. Its application to full propositions keeps the operational construct available for inspection. [R08] [R49] [R50]
4. **The components should be locally connected.** Combine P and E before document pooling. Three-sentence context has a direct communication-science precedent. The geometric mean rewards balanced evidence while remaining a soft approximation to logical conjunction.
5. **Semantic resemblance leaves an inferential question unanswered.** Reuse the construct propositions as NLI hypotheses and obtain an additional passage-support signal. The empirical evidence establishes usefulness of the present BART readout, with no claim of certified endorsement.
6. **Validate against several explicit operationalizations.** Preserve component/full-construct distinctions and report where agreement depends on genre, labeling practice, translation, or aggregation.

The methodological contribution is the arrangement of these parts into an inspectable measure without new populism-specific classifier fitting. It is unnecessary to claim that any single theory predicts E5, BART, .25, eight windows, or equal-weight fusion.

### 7.1 Suggested manuscript wording

> We measure populist rhetoric on two dimensions, people-centrism and anti-elitism, written as claims of popular authority and of elite obstruction. The construct combines an ideational definition of populism with a discourse-level unit of analysis: texts vary in the extent to which they advance a people-versus-elite opposition. We represent each component through written propositions and substantive comparison propositions. Following semantic-axis methods, the difference between the two proposition centroids defines a direction in a frozen sentence-embedding space. This extends a projection construction used in cultural and psychological measurement to political claims. [R01] [R03] [R08] [R09] [R49]
>
> We score local passages on both directions and combine their activations with a geometric mean, which favors balanced component evidence. Overlapping three-sentence windows provide context for claims that are distributed across adjacent sentences. The document score summarizes its strongest local expressions. These choices approximate a conjunctive account of populist discourse while retaining graded information; they do not implement a strict necessary-condition classifier or a prevalence estimator. [R05] [R07] [R22]
>
> Because semantic alignment does not establish textual support, we obtain a complementary natural-language-inference score using the same construct propositions as hypotheses. This stage provides relative model support for specified claims in the selected passages. The final measure combines standardized geometric and inference-based scores, and we assess their agreement with human-coded rhetoric across corpora. The theoretical framework specifies the content and component structure; the representation and aggregation settings were retained through empirical development. [R12] [R13] [R32]

### 7.2 Claims that the evidence can carry

| Formulation | Assessment |
|---|---|
| “Theory-informed proposition-based measurement of populist rhetoric” | Strongly supported. |
| “Two interpretable directions for popular primacy and elite obstruction” | Supported as an operational construction; they are not statistically independent or pure latent factors. |
| “The axis construction has precedents in both static and transformer representation spaces” | Supported by the cited constructions; their evidence does not transfer automatically to these political propositions. |
| “Using multiple anchors establishes reliability or measurement invariance” | Not established; these require additional diagnostics and human validation. |
| “Three-sentence context has prior methodological support in populism research” | Directly supported by Tóth et al.; optimality here remains empirical. |
| “A soft conjunction of component evidence” | Supported with the mathematical definition and compensation qualification. |
| “NLI contributes evidence about support for written claims” | Supported by the task formulation and project validation. |
| “BART verifies that the speaker endorses the populist proposition” | Not established by current outputs or diagnostics. |
| “Every parameter and anchor was derived from theory” | Not supported by the construction history. |
| “The score measures the prevalence of populism in a speech” | Does not describe fixed top-K pooling. |
| “The same score is invariant across languages and corpora” | Not established by translated ranking results. |

### 7.3 The few tests that would strengthen this case most

These are prospective validation needs, not changes to the fixed configuration:

- **Independent content review:** assess the existing anchors for sovereignty, generalized elite obstruction, moral homogeneity, and ordinary-democratic alternatives, without seeing model outcomes. WEPA provides a precedent for expert anchor screening followed by geometric diagnostics. [R49]
- **Anchor stability and internal structure:** assess sensitivity to defensible anchor subsets and paraphrases, and examine within-dimension coherence and between-dimension overlap. Parallelism, relco and embedding-based CFA offer different diagnostic precedents; each needs adaptation before application to this proposition inventory. [R21] [R51] [R53]
- **Source and polarity challenges:** compare the same claim when endorsed, denied, quoted, hypothesized, presupposed, or expressed as a rhetorical question. CommitmentBank research makes this a targeted test of the NLI rationale. [R45] [R48]
- **Matched discriminant cases:** contrast calm populist claims with hostile nonpopulist criticism, generalized with actor-specific elite criticism, and vertical elite opposition with horizontal exclusion.
- **Aggregation interpretation:** assess top-K intensity separately from prevalence and holistic consistency, controlling document length and duplicated evidence from overlapping windows.
- **Fresh validation after any revision:** use unspent texts with independent labels; preserve the current scores as the frozen reference.
- **The two comparators the lineage now demands:** an open-weight LLM scored under Jung et al.'s rubric-and-anchor prompting on the same held-out speeches, since that is the current best-performing alternative on the Global Populism Database, and a politically tuned entailment model such as Political DEBATE as the claim-check replication candidate. Both are pre-declarable and neither has been run. [R85] [R101]

NLI benchmark work shows why a high aggregate task score cannot substitute for targeted reasoning tests. These proposed checks would test the specific mechanism claimed for the second stage. [R46] More generally, substantive validation remains part of automated text measurement even when its computation is transparent. [R44]

## 8. The computational measurement of populism: lineage and research gap

### 8.1 Four generations of text measures

The table lists the computational measures of populism in text that the literature search located, in the order of their method family. Unit is the level at which the method scores; "labels" says whether populism labels enter the measure (training a classifier or fitting a dictionary), "open" whether the scoring model is publicly available and fixed.

| Study | Method family | Unit and material | Language(s) | Populism labels | Open scoring model |
|---|---|---|---|---|---|
| [Rooduijn and Pauwels 2011][R67] | Dictionary versus classical content analysis | Paragraphs of party manifestos | EN, NL, DE, IT | Dictionary written by hand; validated against coders | Yes (word list) |
| [Hawkins 2009][R02]; [Hawkins et al. 2019][R69] | Holistic grading, Global Populism Database | Whole speeches, 0–2 rubric | Many, coded in the original | Human coders | Not a model |
| [Hawkins and Castanho Silva 2018][R68] | Two automated techniques tested against holistic grading across party systems | Speeches and manifestos | Multiple | Supervised | Partly |
| [Aslanidis 2018][R05] | Semantic text analysis, clause-level subject–verb–object coding, populist discourse index | Clauses of grassroots texts | EN, EL | Human coders | Not a model |
| [Tao et al. 2024][R81] | Semantic role labelling plus a populism dictionary, rule-based | Chinese online posts | ZH | Dictionary | Yes (rules) |
| [Gründl 2022][R70]; [Breyer 2023][R71] | Ideational dictionary (people-centrism, anti-elitism, sovereignty) | Social media posts; parliamentary speeches | DE | Dictionary; expert ratings for validation | Yes (word list) |
| [Dai, working paper][R72]; [Dai and Kustov 2022][R73] | Supervised classifier on doc2vec / word embeddings, active learning | Party manifestos and news; US campaign speech paragraphs | EN, ZH | Training labels | Model not released as a fixed instrument |
| [Di Cocco and Monechi 2022][R74]; critique [Jankowski and Huber 2023][R75] | Supervised classifier on manifesto sentences, party-level score | Sentences and manifestos | Multiple, translated | Party-level labels as training targets | Code released; validity contested |
| [Ulinskaitė and Pukelis 2021][R76] | BERT classifier as a screening tool | Paragraphs | LT | Training labels | Not stated |
| [Huguet Cabot et al. 2021][R77] | RoBERTa multitask classifier, populist attitudes with emotions | Reddit comments | EN | 6,861 annotated comments | Dataset and code |
| [Bonikowski, Luo and Stuhler 2022][R78] | RoBERTa with active learning; populism, nationalism, authoritarianism | Sentences of US presidential campaign speeches, 1952–2020 | EN | Training labels | Code and data |
| [Klamm, Rehbein and Ponzetto 2023][R79] | Supervised and semi-supervised classifiers of references to the people and the elite | Sentences of parliamentary debates | DE, EN | Hierarchical annotations (MoPe) | Dataset |
| [Erhard et al. 2025][R42] | PopBERT, fine-tuned GBERT multilabel classifier | Bundestag sentences | DE | 8,795 five-coder sentences | Yes (weights) |
| [van der Veen et al. 2024][R40] | SetFit sentence classifier, shares aggregated to speech | US governor and presidential speeches | EN | Rubric sentences from the coded speeches | Replication package |
| [Licht et al. 2025][R41] | Fine-tuned XLM-T, general versus specific anti-elite criticism | Party tweets, 20 countries | Multiple | 5,401 crowd-coded tweets | Yes (weights) |
| [Makhortykh et al. 2025][R80] | 66 dictionary, classic ML and deep-learning models plus 330 ensembles for populist radical right content | Germanophone test sets | DE | Training labels | Mixed |
| [Behler and Guenther 2025][R82] | Validated BERT classifier, anti-elitism and people-centrism | Reichstag and Bundestag speeches, 1918–1933 and 1991–today | DE | Training labels | Working paper |
| [Wang, Yu, Wang and He 2025][R83] | Fine-tuned RoBERTa on PopBERT labels plus LIWC features | US inaugural and State of the Union addresses; released GPT-4o translations of PopBERT | EN (translated from DE) | PopBERT labels | Data released |
| [Schuette and Bauer 2026][R84] | Comparison of ML, deep-learning and LLM classifiers under a narrowed definition | Tweets | DE | New coding scheme | Preprint |
| [Chalkidis, Brandl and Aslanidis 2026][R38] | Fine-tuned RoBERTa versus zero-shot and fine-tuned LLMs, fine-grained forms | Sentences of Trump campaign speeches; EU-OOD transfer set | EN (EU sentences translated) | Training labels | Weights and data |
| [Sermpezis et al. 2025][R39] | GPT annotation validated by humans | Paragraphs of Greek speeches | EL | GPT-first labels | Data |
| [Jung et al. 2025][R85] | Rubric-and-anchor chain-of-thought prompting of proprietary and open-weight LLMs, replicating GPD scores | Whole speeches | Multiple | Rubric and anchor texts in the prompt; GPD scores for validation | Prompts; models partly proprietary |
| This repository | Proposition anchors in a pretrained sentence-embedding space, soft conjunction, NLI claim check | Speeches, paragraphs, sentences, tweets | EN; DE, PT and 20-country tweets through translation or a multilingual encoder | None in the measure; development labels for configuration | Yes (two public checkpoints, written anchors) |

Two non-text traditions sit beside these: expert surveys such as POPPA, and their machine-learning extensions to more parties and years, and attitude scales. They are listed in section 2.6 and are not text measures. [R64] [R65]

### 8.2 What the lineage shows

- **Every generation has traded transparency for accuracy.** Dictionaries are fully inspectable and cheap; Rooduijn and Pauwels found them somewhat less reliable than hand coding, Gründl improved their breadth and validity, and they remain confined to the vocabulary they list. Supervised classifiers, from doc2vec to PopBERT, are accurate within their training corpus and encode that corpus: Jankowski and Huber showed that the manifesto classifier had learned party names and language rather than populist rhetoric, Chalkidis et al. found that the Trump-speech RoBERTa loses about a quarter of its accuracy out of domain, and Makhortykh et al. found no single model or ensemble that resolves the task. LLM prompting is the most accurate and flexible generation, and Jung et al. reach coder-level agreement with the Global Populism Database by putting the rubric and its anchor texts into the prompt, but the scoring model is then a very large network that interprets the rubric, and proprietary models change between versions. [R67] [R70] [R75] [R38] [R80] [R85]
- **Units and languages are fragmented.** Each study scores one unit, sentences, paragraphs, tweets or whole speeches, in one or two languages, with its own definition and coding scheme, and the results are not comparable across studies. Hunger and Paxton's review documents the same fragmentation at the conceptual level. No prior text measure of populism has been evaluated with one fixed configuration across speeches, paragraphs, sentences and tweets in more than one language. [R63]
- **Classifiers made the construct implicit.** In dictionaries the construct is the word list; in classifiers it is implicit in the training labels; in LLM prompting it returns as a rubric in natural language but is interpreted by an opaque model. The proposition anchors restore a written construct, forty sentences, and keep the scoring linear and inspectable, while the sentence encoder supplies the contextual sensitivity that word lists lack. In that sense the axis is the successor of Gründl's ideational dictionary in a contextual space, with Aslanidis's propositions as its unit of meaning, rather than a competitor of the classifiers.
- **The gap.** A measure of populist claims that (i) is specified by written propositions with theoretical support for each, (ii) uses no populism labels in the measure and only public, fixed models, (iii) scores every text unit with the same configuration, (iv) transfers across languages by a stated pathway, and (v) reports its validity beyond correlation with a single label set, including where it fails, has not been published. The repository's development log is an attempt to fill it, and its known failures, tone, exclusionism, general-versus-specific criticism, are the places where the next round of validation must go.

### 8.3 The alternative framing: transparency and measurement practice

The same gap can be stated as a methods problem rather than a populism problem. Grimmer and Stewart's rule that all automated text measures must be validated has been the field's standard for a decade, and three reviews find it unevenly followed. Baden et al. identify a validity gap in computational text analysis, in which method development has prioritized technology over the operationalization of social-scientific measurements. Birkenmaier, Lechner and Wagner's systematic review of validation practices finds them scarce, inconsistent and without a shared vocabulary. For populism specifically, Jankowski and Huber showed that correlation with expert scores had certified a classifier that measured party identity, and their additional checks, feature importance, a coding-error case study and reshuffled training data, are the checks that a correlation cannot replace. [R44] [R86] [R87] [R75]

The LLM generation adds a second problem: the model itself is not available for inspection or replication. Ollion et al. and Palmer, Smith and Spirling argue that proprietary models cannot be audited and that their use in research needs explicit justification; Barrie, Palmer and Spirling document, in a rolling replication over six months, that language-model annotations vary unacceptably between runs; a 2026 comment in Nature Reviews Psychology recommends open-weight models and an open baseline for any proprietary result; Baumann et al. quantify "LLM hacking", showing that configuration choices alone can make nearly any hypothesis appear significant and that accidental errors affect a third of hypotheses even with the best models; Desai, Card and Jacobs find validation of LLM measurements across eight social-science journals inconsistent and limited; Törnberg sets out best practices for annotation with these models; Argyle et al. ask researchers to keep the goal of inference in view. [R88] [R89] [R90] [R95] [R93] [R94] [R91] [R96] [R97]

The present measure meets several of these requirements through its design. The scoring models are two public checkpoints with fixed weights; the construct is forty sentences in a tracked file; the configuration was chosen on development data under a written stepwise protocol, and the final configuration was scored on the held-out sets once at the end of the stepwise development, with every access logged (earlier exposure is stated in section 6.6); every result carries a run manifest with the commit and input hashes; and every score decomposes into passages, projections and matched claims. What it does not answer is also on the record: the encoder's placement of a passage is opaque, the entailment model's low-tail ordering is model-specific, and the corpus-level standardization is label-free but corpus-dependent. The methods section should name these three as not inspectable; sections 3.7 and 7.3 locate the remaining validation work there.

### 8.4 Where the measure stands against the strongest recent alternatives

On the Global Populism Database, the strongest published alternative is now Jung et al.'s prompted LLM, which replicates GPD scores at coder-level agreement, against the present measure's Spearman of .52 on the held-out speeches (held out from the stepwise choices only; section 6.6) where a human coder pair reaches .75. The two have not been compared on the same speeches and the LLM route depends on the prompt, the model version and, for the proprietary models, the provider. On EU-OOD, Chalkidis et al.'s zero-shot Qwen3-14B and Gemini 2.5 Flash lead the present measure by .05 to .07 macro-F1, and the fine-tuned RoBERTa, scored with its independent per-label sigmoid outputs, is .01 ahead on macro-F1 (.555 against .545) and behind on AUROC (.900 against .942, without a paired test). On the English holdout the present measure leads that RoBERTa by .08 Spearman, with a paired interval that includes zero ([−.08, +.25]). On the four external benchmarks, the present measure exceeds that RoBERTa on all but Portuguese anti-elitism, with paired intervals that exclude zero on each of those five rows and on the Portuguese loss; it ties the in-domain SetFit classifier on ranking US governors, and trails the in-domain XLM-T on tweets by .05. In summary, the measure sits between fine-tuned encoders and prompted LLMs in accuracy, and it is cheaper to run and easier to inspect than either. Whether the LLM route's advantage survives an open-weight model, a fixed prompt and a spent-set protocol is the comparison section 7.3 asks for. [R85] [R38] [R40] [R41]

## References and verification notes

References are grouped by their role in the argument. “Full text” means relevant passages were checked in publisher text, an author upload, or an accessible copy of the original publication. “Abstract/record” means the claim is restricted to what that material establishes. The report is a targeted supporting synthesis, not a systematic review or an estimate of scholarly consensus. No verbatim scholarly quotation is needed for the argument; exact anchor sentences above come from this repository.

### Populism, discourse and measurement theory

- **R01.** Mudde, C. (2004). [The populist Zeitgeist][R01]. *Government and Opposition, 39*(4), 541-563. Full text; definition pp. 543-544, institutional limits p. 561.
- **R02.** Hawkins, K. A. (2009). [Is Chávez populist? Measuring populist discourse in comparative perspective][R02]. *Comparative Political Studies, 42*(8), 1040-1067. Full text; conceptual account pp. 1042-1044 and rubric pp. 1063-1064.
- **R03.** Bonikowski, B., & Gidron, N. (2016). [The populist style in American politics: Presidential campaign discourse, 1952-1996][R03]. *Social Forces, 94*(4), 1593-1621. Publisher abstract and indexed author text checked; definition p. 1593. [Author PDF](https://scholar.harvard.edu/sites/scholar.harvard.edu/files/bonikowski/files/bonikowski_and_gidron_-_the_populist_style_in_american_politics.pdf).
- **R04.** de Vreese, C. H., Esser, F., Aalberg, T., Reinemann, C., & Stanyer, J. (2018). [Populism as an expression of political communication content and style: A new perspective][R04]. *The International Journal of Press/Politics, 23*(4), 423-438. Full text; sections on communication and political actors. [University copy](https://pure.uva.nl/ws/files/31385795/Populism_as_an_expression_of_political_communication.pdf).
- **R05.** Aslanidis, P. (2018). [Measuring populist discourse with semantic text analysis: An application on grassroots populist mobilization][R05]. *Quality & Quantity, 52*, 1241-1263. Full author-uploaded published text; sections 4-6, coding table p. 1257, index p. 1258. [Author upload](https://www.researchgate.net/publication/317258772_Measuring_Populist_Discourse_with_Semantic_Text_Analysis_an_Application_on_Grassroots_Populist_Mobilization_Quality_Quantity).
- **R06.** Popping, R. (2018). [Measuring populist discourse using semantic text analysis: A comment][R06]. *Quality & Quantity, 52*, 2163-2172. Full text; predicate, modality and context, sections 2-5. [Open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6096898/).
- **R07.** Wuttke, A., Schimpf, C., & Schoen, H. (2020). [When the whole is greater than the sum of its parts: On the conceptualization and measurement of populist attitudes and other multidimensional constructs][R07]. *American Political Science Review, 114*(2), 356-374. Full text; aggregation pp. 357-361. Geometric aggregation is discussed as compensatory, p. 359.
- **R14.** Jagers, J., & Walgrave, S. (2007). [Populism as political communication style: An empirical study of political parties' discourse in Belgium][R14]. *European Journal of Political Research, 46*(3), 319-345. Publisher abstract/record; communication lineage and component distinctions. Its thin definition is broader than PE.
- **R15.** Engesser, S., Ernst, N., Esser, F., & Büchel, F. (2017). [Populism and social media: How politicians spread a fragmented ideology][R15]. *Information, Communication & Society, 20*(8), 1109-1126. Full author copy; conceptualization pp. 1111-1112 and fragmentation pp. 1121-1122.
- **R16.** Laclau, E. (2005). [Populism: What's in a name?][R16] In F. Panizza (Ed.), *Populism and the Mirror of Democracy* (pp. 32-49). Verso. Full chapter; demands and antagonistic frontier, pp. 37-39 and 44-47. No DOI verified.
- **R17.** Aslanidis, P. (2016). [Is populism an ideology? A refutation and a new perspective][R17]. *Political Studies, 64*(1_suppl), 88-104. Publisher abstract checked; discursive framing rationale.
- **R18.** Caramani, D. (2017). [Will vs. reason: The populist and technocratic forms of political representation and their critique to party government][R18]. *American Political Science Review, 111*(1), 54-67. Full text; Table 2, p. 61. [University copy](https://dl1.cuni.cz/pluginfile.php/1585207/mod_resource/content/1/08_caramani_2017.pdf).
- **R19.** Bickerton, C., & Invernizzi Accetti, C. (2017). [Populism and technocracy: Opposites or complements?][R19] *Critical Review of International Social and Political Philosophy, 20*(2), 186-206. Publisher abstract/metadata and related author chapter abstract checked. [Author manuscript](https://files.core.ac.uk/download/pdf/42339037.pdf).
- **R22.** Tóth, T., Goyanes, M., & Demeter, M. (2024). [Extend the context! Measuring explicit and implicit populism on three different textual levels][R22]. *Communications, 49*(2), 222-242. Full publisher-provided text via [ResearchGate](https://www.researchgate.net/publication/366974229_Extend_the_context_Measuring_explicit_and_implicit_populism_on_three_different_textual_levels), sections 4-7, pp. 230-238. Issue year 2024.
- **R23.** Team Populism. (2019, March 5). [Coding rubric and anchor texts for the Global Populism Database: A Team Populism project with The Guardian][R23]. Primary codebook. PDF retrieved from BYU; rubric and training explanation pp. 1-3 checked.
- **R28.** Stanley, B. (2008). [The thin ideology of populism][R28]. *Journal of Political Ideologies, 13*(1), 95-110. Full text; four elements and their relations pp. 102-104. [University copy](https://is.muni.cz/el/1423/podzim2016/POL333/um/Stanley_Thin_Ideology_Populism.pdf).
- **R29.** Akkerman, A., Mudde, C., & Zaslove, A. (2014). [How populist are the people? Measuring populist attitudes in voters][R29]. *Comparative Political Studies, 47*(9), 1324-1353. Full author upload; Table 1, p. 1331. Survey content precedent, not text-anchor validation.
- **R30.** Brubaker, R. (2017). [Why populism?][R30] *Theory and Society, 46*, 357-385. Full original article checked; discussion of vertical/horizontal oppositions and common-sense ordinary people. Its repertoire approach is broader than PE.
- **R31.** Müller, J.-W. (2017). [What Is Populism?][R31] Penguin Books. Full 2017 edition checked, pp. 19-20 and 101-102. First published in 2016; those page locators refer to the inspected Penguin edition.
- **R37.** Adcock, R., & Collier, D. (2001). [Measurement validity: A shared standard for qualitative and quantitative research][R37]. *American Political Science Review, 95*(3), 529-546. Publisher abstract and [author-hosted article](https://polisci.berkeley.edu/sites/default/files/people/u3827/APSR2001-Validity.pdf) checked; distinction between concepts, operationalization and context.
- **R47.** Rooduijn, M., van der Brug, W., & de Lange, S. L. (2016). [Expressing or fuelling discontent? The relationship between populist voting and political discontent][R47]. *Electoral Studies, 43*, 32-40. Full [institutional copy](https://pure.uva.nl/ws/files/2805049/178139_1_s2.0_S0261379416301445_main.pdf), p. 35, checked for the paragraph-level co-occurrence rule.

### Semantic geometry, language and NLP

- **R08.** Kozlowski, A. C., Taddy, M., & Evans, J. A. (2019). [The geometry of culture: Analyzing the meanings of class through word embeddings][R08]. *American Sociological Review, 84*(5), 905-949. Full publisher text; “Cultural Dimensions of Word Embeddings,” Figure 2 and surrounding construction.
- **R09.** An, J., Kwak, H., & Ahn, Y.-Y. (2018). [SemAxis: A lightweight framework to characterize domain-specific word semantics beyond sentiment][R09]. *ACL*, 2450-2461. Full paper; sections 3.1.2-3.1.3, equations 1-2, p. 2452.
- **R10.** Kwak, H., An, J., Jing, E., & Ahn, Y.-Y. (2021). [FrameAxis: Characterizing microframe bias and intensity with word embedding][R10]. *PeerJ Computer Science, 7*, e644. Full methods checked; microframe construction and equations 1-3.
- **R11.** Dagan, I., Glickman, O., & Magnini, B. (2006). [The PASCAL Recognising Textual Entailment Challenge][R11]. In *Machine Learning Challenges*, LNCS 3944, 177-190. Publisher abstract and record checked for the directional inference-task definition.
- **R12.** Yin, W., Hay, J., & Roth, D. (2019). [Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach][R12]. *EMNLP-IJCNLP*, 3914-3923. Full paper; section 5, pp. 3918-3919, and section 6.2.
- **R13.** de Marneffe, M.-C., Manning, C. D., & Potts, C. (2012). [Did it happen? The pragmatic complexity of veridicality assessment][R13]. *Computational Linguistics, 38*(2), 301-333. Primary article passages checked; sections 1-2 and source-specific commitment.
- **R20.** Grand, G., Blank, I. A., Pereira, F., & Fedorenko, E. (2022). [Semantic projection recovers rich human knowledge of multiple object features from word embeddings][R20]. *Nature Human Behaviour, 6*, 975-987. Primary institutional abstract and open article checked. Validation concerns object features, not political claims.
- **R21.** Boutyline, A., & Johnston, E. (2026). [Forging better axes: Evaluating and improving the reliability of semantic dimensions in word embeddings][R21]. *Sociological Methods & Research*, advance online publication, 18 September 2026. Published abstract, [2025 author manuscript](https://osf.io/preprints/socarxiv/576h3_v2), pp. 12-13, and [2026 replication v1.0.0](https://doi.org/10.5281/zenodo.20765801) checked; final published full text inaccessible. Current coauthor name: Elle Johnston. The inspected manuscript is dated 19 January 2025, despite the library's earlier year label.
- **R24.** Williams, A., Nangia, N., & Bowman, S. R. (2018). [A broad-coverage challenge corpus for sentence understanding through inference][R24]. *NAACL-HLT*, 1112-1122. Full paper; MultiNLI task, genres and inference phenomena.
- **R25.** Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). [BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension][R25]. *ACL*, 7871-7880. Official paper record checked for architecture provenance.
- **R26.** Facebook AI. [BART-large-MNLI model card][R26]. Official checkpoint documentation, accessed 24 September 2026. Full card checked. Repository probabilities use all three labels; the card's manual zero-shot example uses two.
- **R27.** Reimers, N., & Gurevych, I. (2019). [Sentence-BERT: Sentence embeddings using Siamese BERT-networks][R27]. *EMNLP-IJCNLP*, 3982-3992. Official abstract/record checked for sentence embeddings and efficient similarity comparison.
- **R32.** Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2024). [Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI][R32]. *Political Analysis, 32*(1), 84-100. Full publisher text; sections 2 and 3.1.2, label verbalization and coding instructions.
- **R33.** Wang, L., et al. (2022). [Text embeddings by weakly-supervised contrastive pre-training][R33]. arXiv:2212.03533. Paper abstract and complete official [E5-large-v2 model card](https://huggingface.co/intfloat/e5-large-v2) checked; mean pooling, input prefixes and relative similarities.
- **R34.** Mu, J., & Viswanath, P. (2018). [All-but-the-top: Simple and effective post-processing for word representations][R34]. *ICLR*. Official conference abstract checked; common-mean and dominant-direction removal. The current project uses mean removal only.
- **R35.** Li, B., Zhou, H., He, J., Wang, M., Yang, Y., & Li, L. (2020). [On the sentence embeddings from pre-trained language models][R35]. *EMNLP*, 9119-9130. Official record and primary introduction checked; anisotropy and its proposed transformation.
- **R36.** Boutyline, A., & Arseniev-Koehler, A. (2025). [Meaning in hyperspace: Word embeddings as tools for cultural measurement][R36]. *Annual Review of Sociology, 51*, 89-107. Full publisher HTML checked, including "Measure Features of Concepts, Not Their Contexts" (section 11) and "You Are Not an Embedding Algorithm" (section 14). These support contrast construction and distinguish attributed features from contextual association.
- **R44.** Grimmer, J., & Stewart, B. M. (2013). [Text as data: The promise and pitfalls of automatic content analysis methods for political texts][R44]. *Political Analysis, 21*(3), 267-297. Publisher/author abstract and [author article](https://web.stanford.edu/~jgrimmer/tad2.pdf) passages checked; need for substantive validation.
- **R45.** Jiang, N., & de Marneffe, M.-C. (2019). [Evaluating BERT for natural language inference: A case study on the CommitmentBank][R45]. *EMNLP-IJCNLP*, 6086-6091. Primary abstract/introduction checked; commitment under negation, modality, questions and conditionals.
- **R46.** McCoy, R. T., Pavlick, E., & Linzen, T. (2019). [Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference][R46]. *ACL*, 3428-3448. Primary abstract/introduction checked; heuristics in NLI. Not a direct test of this BART checkpoint.
- **R48.** Simons, M., Tonhauser, J., Beaver, D., & Roberts, C. (2010). [What projects and why][R48]. *Proceedings of SALT 20*, 309-327. Primary proceedings and PDF section 3 checked; at-issue versus projective content under embedding.
- **R49.** Deng, X., & Li, Y. (2026). [Measuring psychological constructs from social media text using the Word Embedding Projection Approach][R49]. *Behavioral Sciences, 16*(5), 762. Full local publication PDF located through the Omabib record and checked; construct model p. 3, equations 1-4 p. 6, expert anchor screening p. 11, and sensitivity analysis p. 22. Evidence concerns word anchors in Chinese fitness discourse.
- **R50.** Kozlowski, A. C., & Boutyline, A. (2026). [Semantic structure of feature space in large language models][R50]. arXiv:2604.27169, v1. Full primary HTML checked; contextual feature extraction and axis construction, sections 2.1-2.3; human-rating and interaxis comparisons, section 3. Preprint; not a test of E5 or NLI.
- **R51.** Taylor, M. A., Stoltz, D. S., Harper, H., Kumar, S., Nandhikonda, S. R., & Burks, L. (2026). [A simulation-based slope metric for anchor list reliability in word embedding spaces][R51]. *Sociological Methodology, 56*(2), 147-176. Publisher abstract and current publication metadata checked. Full local preprint methods checked; regression and mean-slope equations 3-4, pp. 13-14. Page locators refer to that preprint, not the journal pagination.
- **R52.** Hurtado Bodell, M., Keuschnigg, M., Macanovic, A., & Menshikova, A. (2026). [Computational text analysis for building and testing social theory][R52]. *Kölner Zeitschrift für Soziologie und Sozialpsychologie*, online publication. Full publisher text checked, especially sections 2.2-2.3 on theory-first methods and theory-data integration.
- **R53.** Pokropek, A. (2026). [From keyword-based text measures to latent variables: Confirmatory factor analysis with word embeddings][R53]. *EPJ Data Science, 15*, article 51. Full publisher text checked; coordinate-wise vector centering and CFA, sections 4-6; invariance and mean-comparison limits, section 8.6. This is a proposed measurement framework, not validation of the present axes.
- **R54.** Kozlowski, A. C., Dai, C., & Boutyline, A. (2025). [Semantic structure in large language model embeddings][R54]. arXiv:2508.10003, v1, 4 August 2025. Full primary text checked; token embedding matrices, normalized pair offsets, sections 3.2-4.1. Distinct from R50's contextual hidden-state study. The arXiv record lists no journal reference; conference acceptance was not verified.

### Primary benchmark sources

- **R38.** Chalkidis, I., Brandl, S., & Aslanidis, P. (2026). [Identifying fine-grained forms of populism in political discourse: A case study on Donald Trump's presidential campaigns][R38]. *EACL*, 919-936. Full primary PDF checked, especially sections 1-2. Earlier repository reference: [2025 preprint](https://arxiv.org/abs/2507.19303).
- **R39.** Sermpezis, P., et al. (2025). [AgoraSpeech: A multi-annotated comprehensive dataset of political discourse through the lens of humans and AI][R39]. arXiv:2501.06265. Primary abstract and [2024 Zenodo dataset documentation](https://doi.org/10.5281/zenodo.13957177) checked. The release has 5,279 rows; the project's retained benchmark has 5,272.
- **R40.** van der Veen, O., Dzebo, S., Littvay, L., Hawkins, K. A., & Dar, O. (2024). [Classifying populist language in American presidential and governor speeches using automatic text analysis][R40]. arXiv:2408.15213. Full primary HTML checked; rubric-based speech task and thresholding. [Replication data](https://doi.org/10.7910/DVN/ZOHMVO).
- **R41.** Licht, H., Abou-Chadi, T., Barberá, P., & Hua, W. (2025). [Measuring and understanding parties' anti-elite strategies][R41]. *The Journal of Politics, 87*(1), 275-290. Publisher abstract and author-institution PDF p. 279 checked for generalized elite criticism. [Replication data](https://doi.org/10.7910/DVN/T0UKCN).
- **R42.** Erhard, L., Hanke, S., Remer, U., Falenska, A., & Heiberger, R. H. (2025). [PopBERT. Detecting populism and its host ideologies in the German Bundestag][R42]. *Political Analysis, 33*(1), 1-17. Full publisher text checked; conceptualization and component labels. First online 2024; issue year 2025.
- **R43.** Moniz, J., Brás, T., Santana Pereira, J., Belchior, A. M., & Martins, N. (2025). [Dataset on parties' use of populist rhetoric in Portugal during the COVID-19 pandemic (2020-2021)][R43]. Zenodo. Creator-institution record and indexed primary supplementary material checked; detailed benchmark coding and counts also verified against the project's external report.

### Populism theory: additional links (added in revision)

- **R55.** Rooduijn, M. (2014). [The nucleus of populism: In search of the lowest common denominator][R55]. *Government and Opposition, 49*(4), 573-599. Publisher abstract and record checked; the four shared characteristics are listed in the abstract.
- **R56.** Schulz, A., Müller, P., Schemer, C., Wirz, D. S., Wettstein, M., & Wirth, W. (2018). [Measuring populist attitudes on three dimensions][R56]. *International Journal of Public Opinion Research, 30*(2), 316-326. Publisher abstract checked; three first-order dimensions including popular sovereignty.
- **R57.** Taggart, P. (2000). *Populism*. Open University Press. Book; the heartland concept, chapter 8. No DOI. Secondary summaries checked; the original chapter should be consulted before quoting.
- **R58.** Hameleers, M., Bos, L., & de Vreese, C. H. (2017). ["They did it": The effects of emotionalized blame attribution in populist communication][R58]. *Communication Research, 44*(6), 870-900. Publisher abstract checked; blame attribution to elites as the defining feature and the experimental result.
- **R59.** Hawkins, K. A., & Rovira Kaltwasser, C. (2017). [The ideational approach to populism][R59]. *Latin American Research Review, 52*(4), 513-528. Publisher abstract checked.
- **R60.** Weyland, K. (2001). [Clarifying a contested concept: Populism in the study of Latin American politics][R60]. *Comparative Politics, 34*(1), 1-22. Record checked; political-strategic definition.
- **R61.** Moffitt, B., & Tormey, S. (2014). [Rethinking populism: Politics, mediatisation and political style][R61]. *Political Studies, 62*(2), 381-397. Publisher abstract checked.
- **R62.** Mudde, C., & Rovira Kaltwasser, C. (2013). [Exclusionary vs. inclusionary populism: Comparing contemporary Europe and Latin America][R62]. *Government and Opposition, 48*(2), 147-174. Publisher record checked.
- **R63.** Hunger, S., & Paxton, F. (2022). [What's in a buzzword? A systematic review of the state of populism research in political science][R63]. *Political Science Research and Methods, 10*(3), 617-633. Publisher abstract checked; 884 abstracts, conflation with host ideologies.
- **R64.** Meijers, M. J., & Zaslove, A. (2021). [Measuring populism in political parties: Appraisal of a new approach][R64]. *Comparative Political Studies, 54*(2), 372-407. Publisher record checked; POPPA expert survey.
- **R65.** Castanho Silva, B., Jungkunz, S., Helbling, M., & Littvay, L. (2020). [An empirical comparison of seven populist attitudes scales][R65]. *Political Research Quarterly, 73*(2), 409-424. Publisher abstract checked.
- **R66.** Olivas Osuna, J. J., & Rama, J. (2022). [Recalibrating populism measurement tools: Methodological inconsistencies and challenges to our understanding of the relationship between the supply- and demand-side of populism][R66]. *Frontiers in Sociology, 7*, 970043. Open full text available; abstract checked.

### Computational populism measurement: lineage (added in revision)

- **R67.** Rooduijn, M., & Pauwels, T. (2011). [Measuring populism: Comparing two methods of content analysis][R67]. *West European Politics, 34*(6), 1272-1283. Publisher abstract checked.
- **R68.** Hawkins, K. A., & Castanho Silva, B. (2018). [Textual analysis: Big data approaches][R68]. In K. A. Hawkins, R. E. Carlin, L. Littvay, & C. Rovira Kaltwasser (Eds.), *The ideational approach to populism: Concept, theory, and analysis* (pp. 27-48). Routledge. Publisher chapter record checked; page range from the publisher listing.
- **R69.** Hawkins, K. A., Aguilar, R., Castanho Silva, B., Jenne, E. K., Kocijan, B., & Rovira Kaltwasser, C. (2019). [Measuring populist discourse: The Global Populism Database][R69]. Paper presented at the EPSA Annual Conference, Belfast. BYU-hosted PDF checked.
- **R70.** Gründl, J. (2022). [Populist ideas on social media: A dictionary-based measurement of populist communication][R70]. *New Media & Society, 24*(6), 1481-1499. Publisher abstract checked; `popdictR` package.
- **R71.** Breyer, M. (2023). [Populist positions in party competition: Do parties strategically vary their degree of populism in reaction to vote and office loss?][R71] *Party Politics*, advance online 2022. Publisher abstract checked; dictionary measure on German and Austrian debates.
- **R72.** Dai, Y. (working paper). [Measuring populism in context: A supervised approach with word embedding models][R72]. Author's site lists it under working papers; a curated bibliography lists a journal version that could not be verified. Cite as a working paper.
- **R73.** Dai, Y., & Kustov, A. (2022). [When do politicians use populist rhetoric? Populism as a campaign gamble][R73]. *Political Communication, 39*(3), 383-404. Publisher abstract checked.
- **R74.** Di Cocco, J., & Monechi, B. (2022). [How populist are parties? Measuring degrees of populism in party manifestos using supervised machine learning][R74]. *Political Analysis, 30*(3), 311-327. Publisher record checked. Corrigendum and addendum: arXiv:2201.07972.
- **R75.** Jankowski, M., & Huber, R. A. (2023). [When correlation is not enough: Validating populism scores from supervised machine-learning models][R75]. *Political Analysis, 31*(4), 591-605. Publisher record and abstract checked; feature importance, coding-error case study and reshuffling simulations.
- **R76.** Ulinskaitė, J., & Pukelis, L. (2021). [Identifying populist paragraphs in text: A machine-learning approach][R76]. arXiv:2106.03161. Abstract checked.
- **R77.** Huguet Cabot, P.-L., Abadi, D., Fischer, A., & Shutova, E. (2021). [Us vs. them: A dataset of populist attitudes, news bias and emotions][R77]. *EACL 2021*. ACL Anthology record checked.
- **R78.** Bonikowski, B., Luo, Y., & Stuhler, O. (2022). [Politics as usual? Measuring populism, nationalism, and authoritarianism in U.S. presidential campaigns (1952-2020) with neural language models][R78]. *Sociological Methods & Research, 51*(4), 1721-1787. Publisher abstract checked.
- **R79.** Klamm, C., Rehbein, I., & Ponzetto, S. P. (2023). [Our kind of people? Detecting populist references in political debates][R79]. *Findings of EACL 2023*, 1227-1243. ACL Anthology record checked.
- **R80.** Makhortykh, M., de León, E., Christner, C., Sydorova, M., Urman, A., Adam, S., Maier, M., & Gil-Lopez, T. (2025). [Is a single model enough? The systematic comparison of computational approaches for detecting populist radical right content][R80]. *Quality & Quantity, 59*(2). Publisher record and abstract checked.
- **R81.** Tao, Y., Zhan, Z., Zhou, H., Kang, J., & Sun, S. (2024). [Measuring Chinese online populist discourse: An automated semantic text analysis method][R81]. *Chinese Journal of Communication, 18*, 121-141. Publisher abstract checked.
- **R82.** Behler, P. C., & Guenther, L. (2025). [From Weimar to today: Mapping populism across German parliaments][R82]. ECONtribute Discussion Paper No. 381. Title page and abstract checked in the PDF.
- **R83.** Wang, Y., Yu, R., Wang, Z., & He, J. (2025). [The sound of populism: Distinct linguistic features across populist variants][R83]. arXiv:2505.07874. Abstract checked. The GPT-4o translations of PopBERT used in this repository are in the accompanying dataset, Wang, Y. (2026), Harvard Dataverse, doi:10.7910/DVN/VJGS7H, record checked. The development log's PopBERT-EN report previously listed a different author group for this work; corrected in this revision.
- **R84.** Schuette, L., & Bauer, P. C. (2026). [Predicting populism: Methodological and substantive insights from supervised classification of social media data][R84]. SocArXiv preprint. Abstract checked from the project bibliography.
- **R85.** Jung, Y. J., Tamaki, E. R., Chatterley, J., Mitchell, G., Dzebo, S., Sandoval, C., Littvay, L., & Hawkins, K. A. (2025). [Populism meets AI: Advancing populism research with LLMs][R85]. arXiv:2510.07458. Abstract checked; rubric-and-anchor-guided chain-of-thought prompting, proprietary and open-weight models, parity with expert coders on GPD replication.

### Measurement practice, transparency and adjacent methods (added in revision)

- **R86.** Baden, C., Pipal, C., Schoonvelde, M., & van der Velden, M. A. C. G. (2022). [Three gaps in computational text analysis methods for social sciences: A research agenda][R86]. *Communication Methods and Measures, 16*(1), 1-18. Publisher abstract checked.
- **R87.** Birkenmaier, L., Lechner, C. M., & Wagner, C. (2024). [The search for solid ground in text as data: A systematic review of validation practices and practical recommendations for validation][R87]. *Communication Methods and Measures, 18*(3). Publisher abstract checked.
- **R88.** Ollion, É., Shen, R., Macanovic, A., & Chatelain, A. (2024). [The dangers of using proprietary LLMs for research][R88]. *Nature Machine Intelligence, 6*, 4-5. Publisher record checked.
- **R89.** Palmer, A., Smith, N. A., & Spirling, A. (2024). [Using proprietary language models in academic research requires explicit justification][R89]. *Nature Computational Science, 4*, 2-3. Publisher record checked.
- **R90.** Barrie, C., Palmer, A., & Spirling, A. [Replication for language models: Problems, principles, and best practice for political science][R90]. Working paper, conditionally accepted at the *American Journal of Political Science*. Author-hosted PDF and project page checked.
- **R91.** Törnberg, P. (2024). [Best practices for text annotation with large language models][R91]. *Sociologica, 18*(2), 67-85. Publisher record checked.
- **R92.** Törnberg, P. (2025). [Large language models outperform expert coders and supervised classifiers at annotating political social media messages][R92]. *Social Science Computer Review, 43*(6), 1181-1195. Publisher record checked.
- **R93.** Baumann, J., Röttger, P., Urman, A., Wendsjö, A., Plaza-del-Arco, F. M., Gruber, J. B., & Hovy, D. (2025). [Large language model hacking: Quantifying the hidden risks of using LLMs for text annotation][R93]. arXiv:2509.08825. Abstract checked; 13 million labels, 18 models, 2,361 hypotheses.
- **R94.** Desai, M., Card, D., & Jacobs, A. Z. (2026). [Validating LLMs in social science: Epistemic threats and emerging norms][R94]. arXiv:2607.07915. Abstract checked.
- **R95.** [Proprietary LLMs impede scientific transparency and reproducibility][R95]. (2026). *Nature Reviews Psychology, 5*, 513-514. Publisher summary checked; the author list was not retrievable from the publisher page during this revision and must be completed before citation.
- **R96.** Argyle, L. P., Busby, E. C., Gubler, J. R., Hepner, B., Lyman, A., & Wingate, D. (2025). [Arti-"fickle" intelligence: Using LLMs as a tool for inference in the political and social sciences][R96]. *Nature Computational Science*. Publisher record checked.
- **R97.** Ziems, C., Held, W., Shaikh, O., Chen, J., Zhang, Z., & Yang, D. (2024). [Can large language models transform computational social science?][R97] *Computational Linguistics, 50*(1), 237-291. ACL Anthology record checked; zero-shot LLMs do not beat the best fine-tuned models on classification.
- **R98.** Widmann, T., & Wich, M. (2023). [Creating and comparing dictionary, word embedding, and transformer-based models to measure discrete emotions in German political text][R98]. *Political Analysis, 31*(4), 626-641. Publisher record checked; the three-family comparison design.
- **R99.** Licht, H. (2023). [Cross-lingual classification of political texts using multilingual sentence embeddings][R99]. *Political Analysis, 31*(3), 366-379. Publisher abstract checked.
- **R100.** de Vries, E., Schoonvelde, M., & Schumacher, G. (2018). [No longer lost in translation: Evidence that Google Translate works for comparative bag-of-words text applications][R100]. *Political Analysis, 26*(4), 417-430. Publisher abstract checked.
- **R101.** Burnham, M., Kahn, K., Wang, R. Y., & Peng, R. X. (2025). [Political DEBATE: Efficient zero-shot and few-shot classifiers for political text][R101]. *Political Analysis, 34*(3), 329-343. Publisher record checked.
- **R102.** Taylor, M. A., & Stoltz, D. S. (2021). [Integrating semantic directions with concept mover's distance to measure binary concept engagement][R102]. *Journal of Computational Social Science, 4*, 231-242. Publisher record checked.
- **R103.** Duan, Z., Shao, A., Hu, Y., Lee, H., Liao, X., Suh, Y. J., Kim, J., Yang, K.-C., Chen, K., & Yang, S. (2025). [Constructing vec-tionaries to extract message features from texts: A case study of moral content][R103]. *Political Analysis, 33*, 425-445. Publisher record checked; an erratum exists.
- **R104.** Fang, Q., Nguyen, D., & Oberski, D. L. (2022). [Evaluating the construct validity of text embeddings with application to survey questions][R104]. *EPJ Data Science, 11*, 39. Publisher record checked.
- **R105.** Rodriguez, P. L., & Spirling, A. (2022). [Word embeddings: What works, what doesn't, and how to tell the difference for applied research][R105]. *The Journal of Politics, 84*(1), 101-115. Publisher record checked.
- **R106.** Li, B., Yu, T., Koa, K. J. L., & Huang, K.-W. (2026). [The proxy presumption: From semantic embeddings to valid social measures][R106]. arXiv:2605.07409. Abstract checked; Construct Validity Protocol.
- **R107.** Luongo, M., Marocco, D., & Milano, N. (2026). [Measuring psychological states through semantic projection: A theory-driven approach to language-based assessment][R107]. arXiv:2605.04873. Abstract checked.
- **R108.** Plisiecki, H. (2026). [The prediction-measurement gap: Toward meaning representations as scientific instruments][R108]. arXiv:2603.10130. Abstract checked.
- **R109.** Erk, K., & Apidianaki, M. (2024). [Adjusting interpretable dimensions in embedding space with human judgments][R109]. *NAACL 2024*. arXiv:2404.02619; abstract checked.

### Repository evidence inspected

- Definition and implementation: [method](../docs/method.md), [anchors](../anchors/proposition_anchors.json), [NLI hypotheses](../anchors/claim_hypotheses.json), [geometry](../src/populism_axis/geometry.py), [scorer](../scripts/score_texts.py), [NLI evaluation](../scripts/evaluate_nli.py).
- Construction and adoption: [development summary](../research/development-log/summary.md), [window/pooling study](../research/development-log/stage-4-windows-topk.md), [NLI development](../research/development-log/stage-6-claim-check.md), [model replication](../research/development-log/claim-check-replication.md).
- Transfer evidence: [US governors](../research/development-log/external-usgov.md), [party tweets](../research/development-log/external-antielite-tweets.md), [PopBERT](../research/development-log/external-popbert-en.md), [Portugal](../research/development-log/external-portugal.md), [reference-scale analysis](../research/development-log/frozen-reference-scales.md).

[R01]: https://doi.org/10.1111/j.1477-7053.2004.00135.x
[R02]: https://doi.org/10.1177/0010414009331721
[R03]: https://doi.org/10.1093/sf/sov120
[R04]: https://doi.org/10.1177/1940161218790035
[R05]: https://doi.org/10.1007/s11135-017-0517-4
[R06]: https://doi.org/10.1007/s11135-017-0651-z
[R07]: https://doi.org/10.1017/S0003055419000807
[R08]: https://doi.org/10.1177/0003122419877135
[R09]: https://aclanthology.org/P18-1228/
[R10]: https://doi.org/10.7717/peerj-cs.644
[R11]: https://doi.org/10.1007/11736790_9
[R12]: https://aclanthology.org/D19-1404/
[R13]: https://aclanthology.org/J12-2003/
[R14]: https://doi.org/10.1111/j.1475-6765.2006.00690.x
[R15]: https://doi.org/10.1080/1369118X.2016.1207697
[R16]: https://presencial.moodle.ufsc.br/mod/resource/view.php?id=1203007
[R17]: https://doi.org/10.1111/1467-9248.12224
[R18]: https://doi.org/10.1017/S0003055416000538
[R19]: https://doi.org/10.1080/13698230.2014.995504
[R20]: https://doi.org/10.1038/s41562-022-01316-8
[R21]: https://doi.org/10.1177/00491241261467930
[R22]: https://doi.org/10.1515/commun-2022-0009
[R23]: https://populism.byu.edu/0000017d-bf5a-dc9a-affd-bffe4df90001/populism-codebook-the-new-populism-201903-2-pdf
[R24]: https://aclanthology.org/N18-1101/
[R25]: https://aclanthology.org/2020.acl-main.703/
[R26]: https://huggingface.co/facebook/bart-large-mnli
[R27]: https://aclanthology.org/D19-1410/
[R28]: https://doi.org/10.1080/13569310701822289
[R29]: https://doi.org/10.1177/0010414013512600
[R30]: https://doi.org/10.1007/s11186-017-9301-7
[R31]: https://psi424.cankaya.edu.tr/uploads/files/M%C3%BCller,%20What%20Is%20Populism%20(2017%20--%20Penguin%20Books).pdf
[R32]: https://doi.org/10.1017/pan.2023.20
[R33]: https://arxiv.org/abs/2212.03533
[R34]: https://openreview.net/forum?id=HkuGJ3kCb
[R35]: https://aclanthology.org/2020.emnlp-main.733/
[R36]: https://doi.org/10.1146/annurev-soc-090324-024027
[R37]: https://doi.org/10.1017/S0003055401003100
[R38]: https://aclanthology.org/2026.eacl-long.40/
[R39]: https://arxiv.org/abs/2501.06265
[R40]: https://arxiv.org/abs/2408.15213
[R41]: https://doi.org/10.1086/730711
[R42]: https://doi.org/10.1017/pan.2024.12
[R43]: https://doi.org/10.5281/zenodo.14626861
[R44]: https://doi.org/10.1093/pan/mps028
[R45]: https://aclanthology.org/D19-1630/
[R46]: https://aclanthology.org/P19-1334/
[R47]: https://doi.org/10.1016/j.electstud.2016.04.006
[R48]: https://doi.org/10.3765/salt.v20i0.2584
[R49]: https://doi.org/10.3390/bs16050762
[R50]: https://arxiv.org/abs/2604.27169
[R51]: https://doi.org/10.1177/00811750261444338
[R52]: https://doi.org/10.1007/s11577-026-01065-w
[R53]: https://doi.org/10.1140/epjds/s13688-026-00654-1
[R54]: https://arxiv.org/abs/2508.10003
[R55]: https://doi.org/10.1017/gov.2013.30
[R56]: https://doi.org/10.1093/ijpor/edw037
[R57]: https://www.worldcat.org/isbn/0335200451
[R58]: https://doi.org/10.1177/0093650216644026
[R59]: https://www.cambridge.org/core/journals/latin-american-research-review/article/ideational-approach-to-populism/C57445C5C3F96F596C36D0565806E3CA
[R60]: https://doi.org/10.2307/422412
[R61]: https://doi.org/10.1111/1467-9248.12032
[R62]: https://www.cambridge.org/core/journals/government-and-opposition/article/exclusionary-vs-inclusionary-populism-comparing-contemporary-europe-and-latin-america/AAB33C1316BE16B8E4DE229519362E27
[R63]: https://www.cambridge.org/core/journals/political-science-research-and-methods/article/whats-in-a-buzzword-a-systematic-review-of-the-state-of-populism-research-in-political-science/D9CD5E7E13DFA30FD05D41F32E6C122B
[R64]: https://doi.org/10.1177/0010414020938081
[R65]: https://doi.org/10.1177/1065912919833176
[R66]: https://doi.org/10.3389/fsoc.2022.970043
[R67]: https://doi.org/10.1080/01402382.2011.616665
[R68]: https://doi.org/10.4324/9781315196923-2
[R69]: https://populism.byu.edu/0000017d-bf5d-d270-a1fd-ff5fcf750001/global-populism-database-paper-pdf
[R70]: https://doi.org/10.1177/1461444820976970
[R71]: https://doi.org/10.1177/13540688221097082
[R72]: https://daiyaoyao.com/files/Dai_Populism.pdf
[R73]: https://doi.org/10.1080/10584609.2022.2025505
[R74]: https://doi.org/10.1017/pan.2021.29
[R75]: https://doi.org/10.1017/pan.2022.32
[R76]: https://arxiv.org/abs/2106.03161
[R77]: https://aclanthology.org/2021.eacl-main.165/
[R78]: https://doi.org/10.1177/00491241221122317
[R79]: https://aclanthology.org/2023.findings-eacl.91/
[R80]: https://doi.org/10.1007/s11135-024-02034-1
[R81]: https://doi.org/10.1080/17544750.2024.2420622
[R82]: https://www.econtribute.de/RePEc/ajk/ajkdps/ECONtribute_381_2025.pdf
[R83]: https://arxiv.org/abs/2505.07874
[R84]: https://doi.org/10.31235/osf.io/kzhvp_v1
[R85]: https://arxiv.org/abs/2510.07458
[R86]: https://doi.org/10.1080/19312458.2021.2015574
[R87]: https://doi.org/10.1080/19312458.2023.2285765
[R88]: https://doi.org/10.1038/s42256-023-00783-6
[R89]: https://doi.org/10.1038/s43588-023-00585-1
[R90]: https://arthurspirling.org/documents/BarriePalmerSpirling_TrustMeBro.pdf
[R91]: https://doi.org/10.6092/issn.1971-8853/19461
[R92]: https://doi.org/10.1177/08944393241286471
[R93]: https://arxiv.org/abs/2509.08825
[R94]: https://arxiv.org/abs/2607.07915
[R95]: https://doi.org/10.1038/s44159-026-00593-8
[R96]: https://doi.org/10.1038/s43588-025-00843-4
[R97]: https://aclanthology.org/2024.cl-1.8/
[R98]: https://doi.org/10.1017/pan.2022.15
[R99]: https://www.cambridge.org/core/journals/political-analysis/article/abs/crosslingual-classification-of-political-texts-using-multilingual-sentence-embeddings/30689C8798F097EEBA514ABE4891A71B
[R100]: https://doi.org/10.1017/pan.2018.26
[R101]: https://doi.org/10.1017/pan.2025.10028
[R102]: https://doi.org/10.1007/s42001-020-00075-8
[R103]: https://doi.org/10.1017/pan.2025.6
[R104]: https://doi.org/10.1140/epjds/s13688-022-00353-7
[R105]: https://doi.org/10.1086/715162
[R106]: https://arxiv.org/abs/2605.07409
[R107]: https://arxiv.org/abs/2605.04873
[R108]: https://arxiv.org/abs/2603.10130
[R109]: https://arxiv.org/abs/2404.02619
