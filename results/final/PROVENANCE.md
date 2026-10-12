# Provenance of the final holdout and EU-OOD runs

Three manifests from the one-time final scoring on 2026-09-24 record `dirty: true`:

| Manifest | Script (path at the time) | Recorded commit | Recorded dirty | Written (UTC) |
|---|---|---|---|---|
| `results/final/B4_axis/run_manifest.json` | `scripts/v4/run_config_v4.py` | `8f95b3c` | true | 06:06:21 |
| `results/final/final.manifest.json` | `scripts/v4/final_holdout_v4.py` | `8f95b3c` | true | 06:07:03 |
| `results/step6_nli/requests_final.manifest.json` | `scripts/v4/prepare_nli_requests_v4.py` | `3fac28f` | true | 06:02:23 |

The flags are left as recorded. These runs did not execute from a clean tree. This note records what code they executed, from the evidence that survives, and where that evidence stops.

## Conclusion

All three runs executed the tracked code of commit `8f95b3c` (`[v4] final step: EU/holdout NLI requests and one-time final scoring script`, 2026-09-24 08:02:24 +0200). The dirty flags come from two different working-tree changes, neither of them a change to code that the runs did not later commit:

- **B4 and final manifests:** the only modified tracked file was `results/v4/holdout_access_log.json`, which the B0 run itself appended to at 06:05:38, after writing its own manifest. Every later run in the same job then found that file modified.
- **NLI requests manifest:** the run executed `prepare_nli_requests_v4.py` with the `--final` and `--dataset-dir` edits that were committed one second later as `8f95b3c`.

Untracked files are not covered by any of these checks (see "Limits").

## Evidence

1. **Job order and commit.** The local Claude Code transcript of session `d462f10b` (the session linked as `session_01WFBrDz2igpyCArbyiANLBh` in the commit messages) holds the commands. At 06:02:18 one shell command runs `prepare_nli_requests_v4.py --final` twice (English holdout, then EU), then runs `git add .gitignore scripts/v4/prepare_nli_requests_v4.py scripts/v4/final_holdout_v4.py && git commit` (giving `8f95b3c`) and pushes. At 06:04:53 one shell command copies the NLI and RoBERTa caches back from pvgpu1, prints `git status --short | grep -v "^??"`, and runs `scripts/v4/final_holdout_v4.py --final`. That status printed nothing, so tracked files matched `8f95b3c` immediately before the final job. No other tool call ran until the job returned at 06:07:03.
2. **Dirty transition inside the job.** `final_holdout_v4.py` launches B0 and then B4 as `run_config_v4.py` subprocesses. At `8f95b3c`, `run_config_v4.py` writes its manifest and then calls `log_access`, which appends to the tracked `results/v4/holdout_access_log.json`. B0's manifest (06:05:38) records `dirty: false`. B4's (06:06:21) and the final manifest (06:07:03) record `dirty: true`. Commit `1f2ce28` (08:07:59 +0200) adds the outputs and changes no code. Its tracked modifications are the access log (three entries stamped 06:05:38 and 06:06:21) and README.md and docs/results-index.md. Both of those quote the final numbers, which were edited at 06:07:58, after the job.
3. **NLI requests.** `git diff 3fac28f 8f95b3c` touches only `.gitignore`, `scripts/v4/prepare_nli_requests_v4.py` (+15/−4: `--dataset-dir` override, frozen-hypotheses check, `*_final` output names) and the new `scripts/v4/final_holdout_v4.py`. The transcript shows the `prepare_nli_requests_v4.py` edit applied at 06:01:17, and the empty status in (1) shows that nothing else tracked was modified. The manifest's `3fac28f` + dirty therefore means `8f95b3c`'s version of that script. The English-holdout invocation's manifest was overwritten by the EU invocation, which ran second and wrote the same filename. Only the EU manifest survives.
4. **GPU stages were clean.** The NLI scoring of the holdout and EU requests and the RoBERTa encode ran on pvgpu1 in one job. Its log `pvgpu1:~/populism-axis/logs/v4/final_nli_roberta.log` reads `# commit 8f95b3c… dirty=no`, from a git checkout, so the header is valid. That job's NLI manifest was later replaced on pvgpu1 by the tracked step-6 manifest (dev/Agora run, `c9bfdce`, clean), so only the log header documents it. The NLI caches on pvgpu1 and in this checkout have the same sha256 (`nli_english_holdout__bart-large-mnli.npz` `cf2b587c…`, `nli_eu__bart-large-mnli.npz` `75b1ae01…`). The EU window encodes ran at `3fac28f`, clean (`results/final/encode_manifests/`).
5. **Import closure.** An archive of `8f95b3c` imports `final_holdout_v4`, `run_config_v4` and `prepare_nli_requests_v4` without error. Every project module loaded resolves to a tracked file of that commit: `evaluate_contextual_axis_v3`, `evaluate_eu_paper_protocol`, `paired_comparison_v3`, `evaluate_nli_v4`, `run_config_v4`, `populism_axis.{contextual_axis, run_manifest, v4_geometry, v4_sentences}`.
6. **Reproduction of the evaluation stage.** On 2026-10-02 the current `scripts/final_holdout.py` (the renamed descendant of `final_holdout_v4.py`) was run on the stored B0 and B4 scores, the NLI caches and the legacy RoBERTa caches, with no axis subprocess and no access-log write. It reproduced `final_results.json`, `eu_paper_protocol.json` and `final_results.md` byte for byte (`results/roberta-correction/rc-20261002T1600Z-refresh-result-files/verification.json`). This confirms the B6 combination, the NLI pooling and all metrics from the stored inputs. It does not re-derive the B0/B4 axis scores, which would need a new holdout run.

Blob sha256 prefixes at `8f95b3c`: `final_holdout_v4.py` `079558f0…`, `run_config_v4.py` `c42acf30…`, `prepare_nli_requests_v4.py` `088c7119…`, `nli_stage_v4.py` `93a1be12…`, `encode_roberta_comparator_v3.py` `e033f1b5…`, `evaluate_contextual_axis_v3.py` `b9fabb80…`, `contextual_axis.py` `d90da975…`, `v4_geometry.py` `6a523ee3…`.

To inspect the executed code: `git show 8f95b3c:scripts/v4/final_holdout_v4.py`, or `git archive 8f95b3c`. The scripts were moved to `scripts/` without the `_v4` suffix in `d22cbb6`, and `final_holdout.py` was changed on 2026-10-02 to read the corrected sigmoid RoBERTa caches. The current files are not byte-identical to the executed ones.

## Limits

- **Untracked files are unrecorded.** The manifests' dirty check and the transcript's status checks both use `--untracked-files=no`. An untracked `.py` file that shadows a tracked module on `sys.path` cannot be ruled out from the record. Two observations argue against one: the import closure in (5) resolves entirely to tracked files of `8f95b3c` in a clean archive, and the transcript shows no file written under `scripts/` or `src/` between `3fac28f` and the job other than the two committed in `8f95b3c`. The working tree of 2026-09-24 no longer exists, so this cannot be checked directly.
- **The B0/B4 axis scores were not recomputed** (see 6). Their exact code is `8f95b3c` by the argument in (1)–(2), not by reproduction.
- **The English-holdout NLI requests manifest is lost** (see 3). Its inputs and code are those of the surviving EU invocation, apart from the dataset arguments.
- The session transcript is a local file on the laptop (`~/.claude/projects/-home-aleksandar-work-populism-axis/d462f10b-….jsonl`) and is not part of the repository.
