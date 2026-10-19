"""The `eel` command: run the experiments and print the results."""

import argparse

from extraction_eval_lab.data.generator import BookingRecipe


def _sample(args: argparse.Namespace) -> None:
    for doc in BookingRecipe(n=args.n, seed=args.seed, layouts=args.layouts).sample():
        print(f"--- {doc.id}")
        print(doc.text.rstrip())
        print(f"gold: {doc.gold.model_dump_json()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eel", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sample = commands.add_parser("sample", help="print generated documents with their gold JSON")
    sample.add_argument("--n", type=int, default=3)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--layouts", default="seen", choices=["seen", "unseen", "all"])
    sample.set_defaults(run=_sample)
    args = parser.parse_args(argv)
    args.run(args)
