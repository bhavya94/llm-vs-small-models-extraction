"""A rubric judge: an LLM checks each field of an extraction against the source document, one yes/no per field.

The judge never sees the gold JSON. Comparing its verdicts with the exact scorer, which does, measures how far a
judge can be trusted where no gold exists: true positive rate (right fields it accepts), true negative rate
(wrong fields it catches) and Cohen's kappa.
"""

import json
from dataclasses import dataclass

import numpy as np
from metrics_from_scratch.agreement.kappa import cohen_kappa

from extraction_eval_lab.data.schema import FIELDS
from extraction_eval_lab.llm.client import CachedLLM

RUBRIC = ("You check an extraction against its source document. For each field, answer true only if the "
          "extracted value is exactly what the document states for that field: dates as YYYY-MM-DD, amounts as "
          "plain numbers, currency as its 3-letter code. A null is right only if the document does not state the "
          "field. A value the document mentions but then revises, negates or attributes to something else is "
          f"wrong. Reply with one JSON object with exactly these keys and true or false values: {', '.join(FIELDS)}.")


@dataclass
class RubricJudge:
    llm: CachedLLM

    def verdicts(self, text: str, predicted: dict | None) -> dict[str, bool] | None:
        """True or false per field, or None when the judge's reply is not the JSON the rubric asks for."""
        content = f"Document:\n{text}\n\nExtraction:\n{json.dumps(predicted or {}, sort_keys=True)}"
        reply = self.llm.ask(RUBRIC, [{"role": "user", "content": content}])
        try:
            parsed = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
        except ValueError:
            return None
        if set(parsed) != set(FIELDS) or not all(isinstance(v, bool) for v in parsed.values()):
            return None
        return parsed


@dataclass(frozen=True)
class Agreement:
    """How well judge verdicts match the exact scorer over many (document, field) pairs."""

    pairs: int
    tpr: float  # of fields the scorer marks right, the share the judge accepts
    tnr: float  # of fields the scorer marks wrong, the share the judge rejects
    kappa: float
    unparsed: int  # judge replies that were not usable JSON

    @classmethod
    def of(cls, truth: list[dict[str, bool]], verdicts: list[dict[str, bool] | None]) -> "Agreement":
        usable = [(t, v) for t, v in zip(truth, verdicts) if v is not None]
        actual = np.array([t[name] for t, _ in usable for name in FIELDS])
        judged = np.array([v[name] for _, v in usable for name in FIELDS])
        right, wrong = actual, ~actual
        return cls(pairs=len(actual),
                   tpr=float((judged & right).sum() / right.sum()) if right.any() else float("nan"),
                   tnr=float((~judged & wrong).sum() / wrong.sum()) if wrong.any() else float("nan"),
                   kappa=float(cohen_kappa(actual, judged)), unparsed=len(truth) - len(usable))
