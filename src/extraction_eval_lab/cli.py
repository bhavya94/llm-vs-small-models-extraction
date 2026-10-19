"""The `eel` command: run the experiments and print the results."""

import argparse
import os

from extraction_eval_lab.data.adversarial import CATEGORIES, AdversarialRecipe
from extraction_eval_lab.data.generator import BookingRecipe
from extraction_eval_lab.data.schema import FIELDS
from extraction_eval_lab.evaluation.judge import Agreement, RubricJudge
from extraction_eval_lab.evaluation.scorer import FieldReport, field_correct
from extraction_eval_lab.extractors.llm import LLMExtractor, estimate
from extraction_eval_lab.extractors.rules import extract
from extraction_eval_lab.extractors.sft import SFTExtractor, SFTTraining
from extraction_eval_lab.extractors.tagger import Tagger, TaggerTraining
from extraction_eval_lab.llm.client import CachedLLM, Meter


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


def _sft(args: argparse.Namespace) -> None:
    train = BookingRecipe(n=args.train, seed=1, layouts="seen").sample()
    adapter = SFTTraining(epochs=args.epochs).run(train)
    tests = {layouts: BookingRecipe(n=args.n, seed=3, layouts=layouts).sample() for layouts in ["seen", "unseen"]}
    reports = {}
    for label, extractor in [("base", SFTExtractor()), ("tuned", SFTExtractor(adapter=adapter))]:
        reports[label] = {layouts: FieldReport.from_pairs([extractor.extract(d.text) for d in docs],
                                                          [d.gold for d in docs])
                          for layouts, docs in tests.items()}
    print(f"SmolLM2-135M-Instruct, base vs LoRA SFT on {args.train} seen-layout documents for {args.epochs} epochs; "
          f"per-field F1 on {args.n} test documents per layout group")
    print(f"  {'field':<18} {'base seen':>10} {'base unseen':>12} {'SFT seen':>9} {'SFT unseen':>11}")
    for name in FIELDS:
        values = [reports[m][g].fields[name].f1 for m in ("base", "tuned") for g in ("seen", "unseen")]
        print(f"  {name:<18} {values[0]:>10.3f} {values[1]:>12.3f} {values[2]:>9.3f} {values[3]:>11.3f}")
    for label, attr in [("exact match", "exact_match"), ("valid JSON", "valid")]:
        values = [getattr(reports[m][g], attr) for m in ("base", "tuned") for g in ("seen", "unseen")]
        print(f"  {label:<18} {values[0]:>10.3f} {values[1]:>12.3f} {values[2]:>9.3f} {values[3]:>11.3f}")


def _llm(args: argparse.Namespace) -> None:
    tests = {layouts: BookingRecipe(n=args.n, seed=3, layouts=layouts).sample() for layouts in ["seen", "unseen"]}
    meter = Meter(input_price=args.input_price, output_price=args.output_price, cap=args.cap)
    texts = [d.text for docs in tests.values() for d in docs]
    print(f"{args.model} on {len(texts)} documents: estimated at most ${estimate(texts, meter, 400):.2f} before "
          f"caching; cap ${args.cap:.2f}")
    if args.dry_run:
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("set ANTHROPIC_API_KEY to run; --dry-run prints the estimate only")
    import anthropic

    extractor = LLMExtractor(CachedLLM(anthropic.Anthropic(), args.model, meter), repairs=args.repairs)
    reports = {layouts: FieldReport.from_pairs([extractor.extract(d.text) for d in docs], [d.gold for d in docs])
               for layouts, docs in tests.items()}
    print(f"per-field F1 on {args.n} test documents per layout group; {extractor.llm.calls} API calls, "
          f"{extractor.repaired} documents repaired, ${meter.spent:.2f} spent")
    print(f"  {'field':<18} {'seen':>6} {'unseen':>7}")
    for name in FIELDS:
        _report_row(name, reports)
    for label, attr in [("exact match", "exact_match"), ("valid JSON", "valid")]:
        print(f"  {label:<18} {getattr(reports['seen'], attr):>6.3f} {getattr(reports['unseen'], attr):>7.3f}")


def _adversarial(args: argparse.Namespace) -> None:
    sets = AdversarialRecipe(n=args.n).sample()
    train = BookingRecipe(n=600, seed=1, layouts="seen").sample()
    extractors = {"rules": extract, "tagger": Tagger(TaggerTraining().run(train)).extract,
                  "SFT": SFTExtractor(adapter=SFTTraining().run(train)).extract}
    if args.llm:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("set ANTHROPIC_API_KEY for --llm")
        import anthropic

        meter = Meter(input_price=args.input_price, output_price=args.output_price, cap=args.cap)
        texts = [d.text for docs in sets.values() for d in docs]
        print(f"{args.model} on {len(texts)} documents: estimated at most ${estimate(texts, meter, 400):.2f} before "
              f"caching; cap ${args.cap:.2f}")
        extractors["LLM"] = LLMExtractor(CachedLLM(anthropic.Anthropic(), args.model, meter)).extract
    print(f"{args.n} adversarial documents per category, built from seen layouts; "
          "targeted field right / whole document right")
    print(f"  {'category':<17} " + " ".join(f"{name:>13}" for name in extractors))
    for category, docs in sets.items():
        target = CATEGORIES[category][0]
        cells = []
        for run in extractors.values():
            correct = [field_correct(run(d.text), d.gold) for d in docs]
            field = sum(c[target] for c in correct) / len(docs)
            whole = sum(all(c.values()) for c in correct) / len(docs)
            cells.append(f"{field:>5.2f} / {whole:.2f}")
        print(f"  {category:<17} " + " ".join(f"{cell:>13}" for cell in cells))
    if args.llm:
        print(f"LLM spend: ${meter.spent:.2f} (cached replies are free)")


def _judge(args: argparse.Namespace) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("set ANTHROPIC_API_KEY to run the judge")
    import anthropic

    train = BookingRecipe(n=600, seed=1, layouts="seen").sample()
    tagger = Tagger(TaggerTraining().run(train))
    sources = {"tagger, unseen layouts": (tagger.extract, BookingRecipe(n=100, seed=3, layouts="unseen").sample()),
               "rules, adversarial": (extract, [d for docs in AdversarialRecipe(n=40).sample().values() for d in docs])}
    meter = Meter(input_price=args.input_price, output_price=args.output_price, cap=args.cap)
    judge = RubricJudge(CachedLLM(anthropic.Anthropic(), args.model, meter))
    print(f"{args.model} as a rubric judge, one true/false per field, against the exact scorer")
    print(f"  {'predictions':<24} {'fields':>7} {'right':>6} {'TPR':>6} {'TNR':>6} {'kappa':>6} {'unparsed':>9}")
    for label, (run, docs) in sources.items():
        predictions = [run(d.text) for d in docs]
        truth = [field_correct(p, d.gold) for p, d in zip(predictions, docs)]
        agreement = Agreement.of(truth, [judge.verdicts(d.text, p) for p, d in zip(predictions, docs)])
        share = sum(v for t in truth for v in t.values()) / (len(truth) * len(FIELDS))
        print(f"  {label:<24} {agreement.pairs:>7} {share:>6.2f} {agreement.tpr:>6.3f} {agreement.tnr:>6.3f} "
              f"{agreement.kappa:>6.3f} {agreement.unparsed:>9}")
    print(f"judge spend: ${meter.spent:.2f} over {judge.llm.calls} calls (cached replies are free)")


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

    sft = commands.add_parser("sft", help="LoRA fine-tune SmolLM2-135M (once); score base and tuned on both groups")
    sft.add_argument("--train", type=int, default=600, help="seen-layout training documents")
    sft.add_argument("--epochs", type=int, default=2)
    sft.add_argument("--n", type=int, default=100, help="test documents per layout group")
    sft.set_defaults(run=_sft)

    llm = commands.add_parser("llm", help="extract with an Anthropic model: estimate first, then a capped run")
    llm.add_argument("--model", default="claude-haiku-4-5-20251001")
    llm.add_argument("--n", type=int, default=100, help="test documents per layout group")
    llm.add_argument("--repairs", type=int, default=1)
    llm.add_argument("--cap", type=float, default=10.0, help="hard spending cap in dollars")
    llm.add_argument("--input-price", type=float, default=1.0, help="dollars per million input tokens")
    llm.add_argument("--output-price", type=float, default=5.0, help="dollars per million output tokens")
    llm.add_argument("--dry-run", action="store_true", help="print the cost estimate and stop")
    llm.set_defaults(run=_llm)

    adversarial = commands.add_parser("adversarial", help="every extractor on five kinds of adversarial documents")
    adversarial.add_argument("--n", type=int, default=40, help="documents per category")
    adversarial.add_argument("--llm", action="store_true", help="include the LLM (paid, capped, cached)")
    adversarial.add_argument("--model", default="claude-haiku-4-5-20251001")
    adversarial.add_argument("--cap", type=float, default=2.0)
    adversarial.add_argument("--input-price", type=float, default=1.0)
    adversarial.add_argument("--output-price", type=float, default=5.0)
    adversarial.set_defaults(run=_adversarial)

    judge = commands.add_parser("judge", help="an LLM rubric judge against the exact scorer: TPR, TNR, kappa")
    judge.add_argument("--model", default="claude-haiku-4-5-20251001")
    judge.add_argument("--cap", type=float, default=2.0)
    judge.add_argument("--input-price", type=float, default=1.0)
    judge.add_argument("--output-price", type=float, default=5.0)
    judge.set_defaults(run=_judge)
    args = parser.parse_args(argv)
    args.run(args)
