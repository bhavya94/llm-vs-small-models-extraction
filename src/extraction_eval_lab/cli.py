"""The `eel` command: run the experiments and print the results."""

import argparse

from extraction_eval_lab.data.generator import BookingRecipe
from extraction_eval_lab.data.schema import FIELDS
from extraction_eval_lab.evaluation.scorer import FieldReport
from extraction_eval_lab.extractors.rules import extract
from extraction_eval_lab.extractors.tagger import Tagger, TaggerTraining


def _sample(args: argparse.Namespace) -> None:
    for doc in BookingRecipe(n=args.n, seed=args.seed, layouts=args.layouts).sample():
        print(f"--- {doc.id}")
        print(doc.text.rstrip())
        print(f"gold: {doc.gold.model_dump_json()}")


def _rules(args: argparse.Namespace) -> None:
    reports = {}
    for layouts in ["seen", "unseen"]:
        docs = BookingRecipe(n=args.n, seed=args.seed, layouts=layouts).sample()
        reports[layouts] = FieldReport.from_pairs([extract(d.text) for d in docs], [d.gold for d in docs])
    print(f"rule-based extractor, {args.n} documents per layout group; per-field F1")
    print(f"  {'field':<18} {'seen':>6} {'unseen':>7}")
    for name in FIELDS:
        print(f"  {name:<18} {reports['seen'].fields[name].f1:>6.3f} {reports['unseen'].fields[name].f1:>7.3f}")
    for label, attr in [("exact match", "exact_match"), ("valid JSON", "valid")]:
        print(f"  {label:<18} {getattr(reports['seen'], attr):>6.3f} {getattr(reports['unseen'], attr):>7.3f}")


def _report_row(label: str, reports: dict) -> None:
    print(f"  {label:<18} {reports['seen'].fields[label].f1:>6.3f} {reports['unseen'].fields[label].f1:>7.3f}")


def _tagger(args: argparse.Namespace) -> None:
    train = BookingRecipe(n=args.train, seed=1, layouts="seen").sample()
    tagger = Tagger(TaggerTraining(epochs=args.epochs).run(train))
    reports = {}
    for layouts in ["seen", "unseen"]:
        docs = BookingRecipe(n=args.n, seed=3, layouts=layouts).sample()
        reports[layouts] = FieldReport.from_pairs([tagger.extract(d.text) for d in docs], [d.gold for d in docs])
    print(f"distilbert-base-cased token tagger, trained on {args.train} seen-layout documents for {args.epochs} "
          f"epochs; per-field F1 on {args.n} test documents per layout group")
    print(f"  {'field':<18} {'seen':>6} {'unseen':>7}")
    for name in FIELDS:
        _report_row(name, reports)
    for label, attr in [("exact match", "exact_match"), ("valid JSON", "valid")]:
        print(f"  {label:<18} {getattr(reports['seen'], attr):>6.3f} {getattr(reports['unseen'], attr):>7.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eel", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sample = commands.add_parser("sample", help="print generated documents with their gold JSON")
    sample.add_argument("--n", type=int, default=3)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--layouts", default="seen", choices=["seen", "unseen", "all"])
    sample.set_defaults(run=_sample)

    rules = commands.add_parser("rules", help="score the rule-based extractor on seen and unseen layouts")
    rules.add_argument("--n", type=int, default=100, help="documents per layout group")
    rules.add_argument("--seed", type=int, default=3)
    rules.set_defaults(run=_rules)

    tagger = commands.add_parser("tagger", help="train (once) and score the token tagger on seen and unseen layouts")
    tagger.add_argument("--train", type=int, default=600, help="seen-layout training documents")
    tagger.add_argument("--epochs", type=int, default=3)
    tagger.add_argument("--n", type=int, default=100, help="test documents per layout group")
    tagger.set_defaults(run=_tagger)
    args = parser.parse_args(argv)
    args.run(args)
