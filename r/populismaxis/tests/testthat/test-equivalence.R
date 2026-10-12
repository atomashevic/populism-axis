# Deterministic cross-language checks exercise both APIs without model weights.
test_that("R and Python preserve complete outputs across 36 scoring configurations", {
fixture <- .pa_fixture_module()
.pa_module <- .pa_test_internal(".pa_module")
.pa_unwrap <- .pa_test_internal(".pa_unwrap")
.pa_wrap <- .pa_test_internal(".pa_wrap")
py <- .pa_module()
reticulate::py_run_string("import hashlib\nimport numpy as np\nfrom unittest.mock import patch\nimport populism_axis as pa\nfrom populism_axis.definitions import canonical\nfrom populism_axis import run_manifest\norig_manifest=run_manifest.build_run_manifest\ndef fake_manifest(*a,**kw):\n kw['timestamp']='2026-10-09T00:00:00Z'\n return orig_manifest(*a,**kw)\ndef fake_encoder(*a, **kw):\n def encode(texts):\n  return np.vstack([np.random.default_rng(int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], 'little')).normal(size=1024).astype(np.float32) for t in texts])\n return encode, lambda:None\ndef fake_nli(model,premises,hypotheses,*a,**kw):\n return np.array([[.05+.9*(int.from_bytes(hashlib.sha256((p+h).encode()).digest()[:4],'little')/2**32) for h in hypotheses] for p in premises]),None\npatches=[patch('populism_axis.scoring.make_encoder',fake_encoder),patch('populism_axis.scoring.preflight_models'),patch('populism_axis.scoring.run_nli',fake_nli),patch('populism_axis.run_manifest.build_run_manifest',fake_manifest)]\nfor p in patches:p.start()\ndef same(a,b):\n import pandas as pd\n pd.testing.assert_frame_equal(a.scores,b.scores,check_exact=True)\n assert canonical(a.evidence)==canonical(b.evidence),'evidence mismatch'\n assert canonical(a.provenance)==canonical(b.provenance),'provenance mismatch'\n return True")
on.exit(reticulate::py_run_string('for p in reversed(patches): p.stop()'), add = TRUE)
b <- reticulate::py
texts <- vapply(seq_len(35), function(i) paste('Example document',i,'discusses institution',i*i,'. Citizens express view',i+2,'. Officials answer on topic',i+3,'. Further evidence for case',i+4,'.'),character(1))
ids <- c('001','NA','ж',as.character(4:35)); inp <- as.list(texts); idp <- as.list(ids)
components <- list(integrity=list(positive='Honest government',contrast='Dishonest government'),capacity=list(positive='Capable government',contrast='Incapable government'))
configs <- list(define_scale('trust',positive=c('Fair institutions.','Reliable institutions.'),contrast='Unfair institutions.'),define_scale('quality',components=components,combine='mean'),define_scale('quality',components=components,combine='conjunction'))
n <- 0L
for (s in configs) for (unit in c('speech','sentence')) for (pool in c('mean','top')) for (claim in c(FALSE,TRUE)) {
 ref <- fit_reference(texts,s,ids=ids,unit=unit,pooling=pool,claim_check=claim)
 expected_ref <- py$fit_reference(inp,scale=.pa_unwrap(s,'scale'),ids=idp,unit=unit,pooling=pool,claim_check=claim)
 stopifnot(identical(ref$identity,reticulate::py_to_r(expected_ref$identity)))
 actual <- score_texts(texts,ref,ids=ids); expected <- py$score_texts(inp,reference=expected_ref,ids=idp)
 stopifnot(reticulate::py_to_r(b$same(.pa_unwrap(actual,'result'),expected)))
 cutoff <- scores(actual)[[summary(s)$name]][7]; rule <- make_classifier(ref,cutoff,'low','high')
 expected_rule <- py$make_classifier(expected_ref,threshold=cutoff,below='low',at_or_above='high')
 stopifnot(identical(rule$identity,reticulate::py_to_r(expected_rule$identity)))
 stopifnot(reticulate::py_to_r(b$same(.pa_unwrap(apply_classifier(rule,actual),'result'),expected_rule$apply(expected))))
 n <- n+1L
}
for (unit in c('speech','sentence')) for (claim in c(FALSE,TRUE)) {
 ref <- fit_reference(texts,'populism',ids=ids,unit=unit,claim_check=claim)
 expected_ref <- py$fit_reference(inp,scale='populism',ids=idp,unit=unit,claim_check=claim)
 stopifnot(identical(ref$identity,reticulate::py_to_r(expected_ref$identity)))
 stopifnot(reticulate::py_to_r(b$same(.pa_unwrap(score_texts(texts,ref,ids=ids),'result'),py$score_texts(inp,reference=expected_ref,ids=idp))))
 n <- n+1L
 for (kind in c('frozen','corpus')) {
 actual <- score_texts(texts,kind,scale='populism',ids=ids,unit=unit,claim_check=claim)
 expected <- py$score_texts(inp,reference=kind,scale='populism',ids=idp,unit=unit,claim_check=claim)
 stopifnot(reticulate::py_to_r(b$same(.pa_unwrap(actual,'result'),expected)))
 n <- n+1L
 }
}
expect_identical(n, 36L)
np <- reticulate::import('numpy',convert=FALSE); pd <- reticulate::import('pandas',convert=FALSE)
ref <- .pa_wrap(fixture$reference(),'reference'); cut <- .12345678901234566
vals <- list(reticulate::py_to_r(np$nextafter(cut,-Inf)),cut,reticulate::py_to_r(np$nextafter(cut,Inf)))
raw <- fixture$result(); frame <- raw$scores; frame[['trust']] <- pd$Series(vals)
r <- .pa_wrap(py$ScoreResult(frame,raw$evidence,raw$provenance),'result'); rule <- make_classifier(ref,cut,'испод','изнад')
stopifnot(identical(apply_classifier(rule,r)$scores$trust__label,c('испод','изнад','изнад')),identical(scores(r)$trust,unlist(vals)))
})
