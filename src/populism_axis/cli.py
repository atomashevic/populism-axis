"""Installed command-line adapters for the public Python interface."""
import argparse
import json
from pathlib import Path
import sys

_TEMPLATE = """component,pole,proposition
trust,positive,Public institutions act in the public interest.
trust,contrast,Public institutions routinely betray the public.
trust,positive,Public institutions respond fairly to citizens.
trust,contrast,Public institutions privilege powerful insiders.
"""


def _save_path(path, overwrite):
    path = Path(path)
    if path.exists() and (not overwrite or not path.is_file()):
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    return path


def _load_scale(args):
    from .definitions import ScaleDefinition, read_scale_csv
    if args.scale is None:
        if args.name is not None or args.combine is not None:
            raise ValueError("--name and --combine require --scale with a proposition CSV.")
        return None
    if args.scale == "populism":
        if args.name is not None or args.combine is not None:
            raise ValueError("--name and --combine apply only to proposition CSV input.")
        return "populism"
    path = Path(args.scale)
    if path.suffix.lower() == ".csv":
        if not args.name:
            raise ValueError("A proposition CSV requires --name for the scale name.")
        return read_scale_csv(path, name=args.name, combine=args.combine)
    if args.name is not None or args.combine is not None:
        raise ValueError("--name and --combine apply only to proposition CSV input.")
    return ScaleDefinition.load(path)


def _validate_classifier(rule, reference, scale):
    from .reference import FittedReference
    from .scoring import score_fields
    if not isinstance(reference, FittedReference):
        raise ValueError("--classifier requires its saved fitted reference; corpus and frozen reference modes are incompatible.")
    if rule.reference_identity != reference.identity or rule.definition_identity != reference.scale.identity:
        raise ValueError("Classifier definition/reference mismatch. Use the exact fitted reference used to create this rule.")
    if scale is not None:
        from .definitions import populism_preset
        effective_scale = populism_preset() if scale == "populism" else scale
        if effective_scale.identity != rule.definition_identity:
            raise ValueError("Classifier definition mismatch: --scale differs from the rule's fitted definition.")
    if rule.score not in score_fields(reference.scale, reference.settings["claim_check"]):
        raise ValueError(f"Classifier score {rule.score!r} is unavailable for this fitted reference.")


def _execution(args):
    out = {"device": args.device, "threads": args.threads,
           "encoder_batch_size": args.encoder_batch_size, "nli_batch_size": args.nli_batch_size,
           "local_files_only": not args.allow_download}
    for name in ("unit", "pooling", "clean", "claim_check"):
        if getattr(args, name) is not None:
            out[name] = getattr(args, name)
    return out


def _scale_options(parser):
    parser.add_argument("--scale", help="Saved definition JSON, proposition CSV, or populism")
    parser.add_argument("--name", help="Scale name when --scale is a proposition CSV")
    parser.add_argument("--combine", choices=("mean", "conjunction"))


def _corpus_options(parser):
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column")
    _scale_options(parser)
    parser.add_argument("--unit", choices=("speech", "sentence"))
    parser.add_argument("--pooling", choices=("mean", "top"))
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--claim-check", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--encoder-batch-size", type=int)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def _parser():
    parser = argparse.ArgumentParser(prog="populism-axis", description="Proposition-defined continuous scales and explicit threshold labels.")
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="Print or save an editable proposition CSV")
    template.add_argument("--out", type=Path)
    template.add_argument("--overwrite", action="store_true")
    define = commands.add_parser("define", help="Validate and save a CSV definition without inference")
    define.add_argument("--input", required=True, type=Path)
    define.add_argument("--name", required=True)
    define.add_argument("--combine", choices=("mean", "conjunction"))
    define.add_argument("--out", required=True, type=Path)
    define.add_argument("--overwrite", action="store_true")
    fit = commands.add_parser("fit-reference", help="Fit and save a reference from at least 30 documents")
    _corpus_options(fit)
    fit.add_argument("--out", required=True, type=Path)
    score = commands.add_parser("score", help="Score a literal text CSV")
    _corpus_options(score)
    score.add_argument("--reference", required=True, help="frozen, corpus, or saved reference JSON")
    score.add_argument("--out-dir", required=True, type=Path)
    score.add_argument("--classifier", type=Path, help="Apply a compatible saved threshold rule after scoring")
    rule = commands.add_parser("make-classifier", help="Save an explicit threshold bound to a fitted reference")
    rule.add_argument("--reference", required=True, type=Path)
    rule.add_argument("--threshold", required=True, type=float)
    rule.add_argument("--below", required=True)
    rule.add_argument("--at-or-above", required=True)
    rule.add_argument("--score")
    rule.add_argument("--out", required=True, type=Path)
    rule.add_argument("--overwrite", action="store_true")
    apply = commands.add_parser("apply-classifier", help="Label existing saved scores without model inference")
    apply.add_argument("--classifier", required=True, type=Path)
    apply.add_argument("--input-dir", required=True, type=Path)
    apply.add_argument("--out-dir", required=True, type=Path)
    apply.add_argument("--overwrite", action="store_true")
    commands.add_parser("verify-assets", help="Verify the four bundled frozen assets without model inference")
    return parser


def _run(args):
    if args.command == "template":
        if args.out is None:
            print(_TEMPLATE, end="")
        else:
            path = _save_path(args.out, args.overwrite)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_TEMPLATE, encoding="utf-8")
        return
    if args.command == "verify-assets":
        from .assets import verify_assets
        print(json.dumps(verify_assets(), indent=2))
        return
    if args.command == "define":
        from .definitions import read_scale_csv
        _save_path(args.out, args.overwrite)
        scale = read_scale_csv(args.input, name=args.name, combine=args.combine)
        scale.save(args.out, overwrite=args.overwrite)
        print(json.dumps(scale.summary(), ensure_ascii=False, indent=2))
        return
    from .io import fit_reference_csv, score_csv, preflight_scores, write_scores
    from .reference import FittedReference, ThresholdRule, make_classifier
    if args.command == "fit-reference":
        _save_path(args.out, args.overwrite)
        scale = _load_scale(args)
        if scale is None:
            scale = "populism"
        reference = fit_reference_csv(args.input, scale=scale, text_column=args.text_column,
                                      id_column=args.id_column, **_execution(args))
        reference.save(args.out, overwrite=args.overwrite)
        print(json.dumps(reference.summary(), ensure_ascii=False, indent=2))
    elif args.command == "score":
        preflight_scores(args.out_dir, overwrite=args.overwrite)
        reference = args.reference if args.reference in ("corpus", "frozen") else FittedReference.load(args.reference)
        rule = ThresholdRule.load(args.classifier) if args.classifier else None
        scale = _load_scale(args)
        if rule is not None:
            _validate_classifier(rule, reference, scale)
        result = score_csv(args.input, reference=reference, scale=scale,
                           text_column=args.text_column, id_column=args.id_column, **_execution(args))
        if rule is not None:
            result = rule.apply(result)
        write_scores(result, args.out_dir, overwrite=args.overwrite)
        print(f"Wrote {len(result.scores)} documents to {args.out_dir}")
    elif args.command == "make-classifier":
        _save_path(args.out, args.overwrite)
        rule = make_classifier(FittedReference.load(args.reference), threshold=args.threshold,
                               below=args.below, at_or_above=args.at_or_above, score=args.score)
        rule.save(args.out, overwrite=args.overwrite)
        print(json.dumps(rule.summary(), ensure_ascii=False, indent=2))
    elif args.command == "apply-classifier":
        import pandas as pd
        from .scoring import ScoreResult
        preflight_scores(args.out_dir, overwrite=args.overwrite)
        result = ScoreResult(
            scores=pd.read_csv(args.input_dir / "scores.csv", dtype={"id": str}, keep_default_na=False),
            evidence=json.loads((args.input_dir / "evidence.json").read_text(encoding="utf-8")),
            provenance=json.loads((args.input_dir / "run_manifest.json").read_text(encoding="utf-8")),
        )
        result = ThresholdRule.load(args.classifier).apply(result)
        write_scores(result, args.out_dir, overwrite=args.overwrite)
        print(f"Wrote {len(result.scores)} labeled documents to {args.out_dir}")


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _run(args)
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"populism-axis: error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
