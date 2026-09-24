"""Pure-mock tests for the bounded LLM discussion."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nuwa_agent.round_deliberation import RoundDeliberation, validate_decision  # noqa: E402


FALLBACK = {
    "temperature": 0.6,
    "mutate_fraction": 0.4,
    "substitutions_per_candidate": 1,
    "parent_focus": "balanced",
    "evidence_ids": ["baseline"],
    "rationale": "Baseline search parameters.",
}
OUTPUTS = [
    {"temperature": 0.8, "mutate_fraction": 0.3,
     "evidence_ids": ["hv_drop"], "rationale": "Explore more broadly."},
    {"mutate_fraction": 0.5, "substitutions_per_candidate": 2,
     "parent_focus": "te", "evidence_ids": ["te_weak"], "rationale": "Edit TE parents."},
    {"concerns": ["Keep feasibility in view."],
     "recommended_changes": ["Reduce mutation share."], "evidence_ids": ["feasibility"]},
    {"temperature": 0.75, "mutate_fraction": 0.25,
     "evidence_ids": ["hv_drop", "feasibility"], "rationale": "Balance exploration."},
    {"mutate_fraction": 0.35, "substitutions_per_candidate": 1,
     "parent_focus": "te", "evidence_ids": ["feasibility"],
     "rationale": "Smaller synonymous edits."},
    {"temperature": 0.75, "mutate_fraction": 0.35,
     "substitutions_per_candidate": 1, "parent_focus": "te",
     "evidence_ids": ["hv_drop", "feasibility"],
     "rationale": "Explore while preserving feasibility."},
]
SUMMARY = {"round": 2, "next_round": 3, "hv_drop": True,
           "te_weak": True, "feasibility": 0.6}


class FakeCompletions:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs[len(self.calls) - 1]
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=output))])


def make_agent(outputs):
    completions = FakeCompletions(outputs)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return RoundDeliberation(client=client, model="mock-model"), completions


class RoundDeliberationTests(unittest.TestCase):
    def test_real_prior_messages_reach_each_later_agent(self):
        raw = [json.dumps(item) for item in OUTPUTS]
        agent, completions = make_agent(raw)
        result = agent.decide(SUMMARY, FALLBACK)

        self.assertFalse(result["fallback_used"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["model"], "mock-model")
        self.assertEqual(result["decision"], validate_decision(OUTPUTS[-1]))
        self.assertEqual(len(completions.calls), 6)
        self.assertEqual([item["phase"] for item in result["transcript"]], [
            "generation_proposal", "editing_proposal", "constraint_review",
            "generation_revision", "editing_revision", "central_decision",
        ])
        for index, call in enumerate(completions.calls):
            self.assertEqual(call["model"], "mock-model")
            self.assertEqual([message["role"] for message in call["messages"]], ["system", "user"])
            payload = json.loads(call["messages"][1]["content"])
            self.assertEqual(payload["timeline"], {
                "evaluated_round": 2, "decision_applies_to_round": 3,
            })
            expected = [] if index < 2 else OUTPUTS[:index]
            self.assertEqual([message["message"] for message in payload["previous_agent_messages"]], expected)
            self.assertEqual(result["transcript"][index]["role"], result["transcript"][index]["agent"])
            self.assertEqual(result["transcript"][index]["system"], call["messages"][0]["content"])
            self.assertEqual(result["transcript"][index]["user"], call["messages"][1]["content"])
            self.assertEqual(result["transcript"][index]["output"], raw[index])

    def test_invalid_central_decision_synthesizes_valid_expert_outputs(self):
        outputs = [json.dumps(item) for item in OUTPUTS]
        invalid = dict(OUTPUTS[-1], temperature=1.5)
        outputs[-1:] = [json.dumps(invalid)] * 3
        agent, _ = make_agent(outputs)
        result = agent.decide(SUMMARY, FALLBACK)

        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["decision_source"], "synthesized_experts")
        self.assertEqual(result["decision"]["temperature"], OUTPUTS[3]["temperature"])
        self.assertEqual(result["decision"]["parent_focus"], OUTPUTS[4]["parent_focus"])
        self.assertEqual(result["decision"]["mutate_fraction"], 0.25)
        self.assertEqual(result["error"], "central_decision: DecisionValidationError")
        self.assertEqual(len(result["transcript"]), 6)

    def test_failed_reviewer_isolated_and_not_forwarded(self):
        outputs = ([json.dumps(item) for item in OUTPUTS[:2]]
                   + [RuntimeError("secret-key")] * 3
                   + [json.dumps(item) for item in OUTPUTS[3:]])
        agent, completions = make_agent(outputs)
        result = agent.decide(SUMMARY, FALLBACK)

        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["decision"], validate_decision(OUTPUTS[-1]))
        self.assertEqual(result["error"], "constraint_review: RuntimeError")
        self.assertEqual(len(completions.calls), 8)
        self.assertEqual(result["transcript"][2]["output"], None)
        later_payload = json.loads(completions.calls[5]["messages"][1]["content"])
        self.assertNotIn("constraint_review", [
            item["phase"] for item in later_payload["previous_agent_messages"]
        ])
        self.assertNotIn("secret-key", json.dumps(result))

    def test_empty_central_response_retries_and_recovers(self):
        outputs = [json.dumps(item) for item in OUTPUTS[:5]] + ["", json.dumps(OUTPUTS[-1])]
        agent, completions = make_agent(outputs)
        result = agent.decide(SUMMARY, FALLBACK)

        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["decision_source"], "central")
        self.assertEqual(result["decision"], validate_decision(OUTPUTS[-1]))
        self.assertEqual(len(completions.calls), 7)
        self.assertEqual(result["transcript"][-1]["attempts"][0]["status"], "error")

    def test_evidence_aliases_are_canonicalized_and_unknown_ids_are_dropped(self):
        outputs = [dict(item) for item in OUTPUTS]
        outputs[0]["evidence_ids"] = ["round_2_hv_drop_1", "made_up_99"]
        agent, _ = make_agent([json.dumps(item) for item in outputs])
        result = agent.decide(SUMMARY, FALLBACK)

        self.assertEqual(result["transcript"][0]["parsed_output"]["evidence_ids"], ["hv_drop"])

    def test_replacement_characters_are_sanitized(self):
        outputs = [dict(item) for item in OUTPUTS]
        outputs[0]["rationale"] = "repair��then explore"
        agent, _ = make_agent([json.dumps(item) for item in outputs])
        result = agent.decide(SUMMARY, FALLBACK)

        self.assertNotIn("�", result["transcript"][0]["output"])
        self.assertIn("—", result["transcript"][0]["parsed_output"]["rationale"])

    def test_nonfinite_and_wrong_numeric_types_are_rejected(self):
        for bad in (float("nan"), float("inf"), True, "0.6"):
            with self.subTest(bad=bad):
                decision = dict(FALLBACK, temperature=bad)
                with self.assertRaises(ValueError):
                    validate_decision(decision)
        decision = dict(FALLBACK, substitutions_per_candidate=2.0)
        with self.assertRaises(ValueError):
            validate_decision(decision)


if __name__ == "__main__":
    unittest.main()
