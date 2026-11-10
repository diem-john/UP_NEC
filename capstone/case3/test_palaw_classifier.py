import json
import unittest
from unittest.mock import patch

from palaw_classifier import ClassificationError, classify_and_route, parse_decision


def model_response(payload):
    return type("Response", (), {
        "message": type("Message", (), {
            "content": json.dumps(payload)
        })()
    })()


class DecisionSchemaTests(unittest.TestCase):
    def test_valid_inquiry_hr(self):
        result = parse_decision(json.dumps({
            "category": "INQUIRY",
            "confidence": 0.92,
            "routed_team": "HR",
            "reason": "The employee asks about emergency leave.",
            "next_action": "ROUTE_TO_OTHER_TEAM",
        }))
        self.assertEqual(result["category"], "INQUIRY")
        self.assertEqual(result["routed_team"], "HR")

    def test_valid_complaint_accounting(self):
        result = parse_decision(json.dumps({
            "category": "COMPLAINT",
            "confidence": 0.92,
            "routed_team": "ACCOUNTING",
            "reason": "The employee reports missing 13th-month pay.",
            "next_action": "ROUTE_TO_OTHER_TEAM",
        }))
        self.assertEqual(result["routed_team"], "ACCOUNTING")

    def test_valid_legal_action(self):
        result = parse_decision(json.dumps({
            "category": "INQUIRY",
            "confidence": 0.92,
            "routed_team": "LEGAL",
            "reason": "The employee asks about filing a formal DOLE complaint.",
            "next_action": "ROUTE_TO_LEGAL_AI_ASSISTANT",
        }))
        self.assertEqual(result["next_action"], "ROUTE_TO_LEGAL_AI_ASSISTANT")

    def test_legal_team_requires_legal_action(self):
        with self.assertRaises(ClassificationError):
            parse_decision(json.dumps({
                "category": "COMPLAINT",
                "confidence": 0.92,
                "routed_team": "LEGAL",
                "reason": "Legal concern.",
                "next_action": "ROUTE_TO_OTHER_TEAM",
            }))

    def test_non_legal_team_cannot_use_legal_action(self):
        with self.assertRaises(ClassificationError):
            parse_decision(json.dumps({
                "category": "INQUIRY",
                "confidence": 0.92,
                "routed_team": "HR",
                "reason": "Leave question.",
                "next_action": "ROUTE_TO_LEGAL_AI_ASSISTANT",
            }))

    def test_rejects_unsupported_team(self):
        with self.assertRaises(ClassificationError):
            parse_decision(json.dumps({
                "category": "INQUIRY",
                "confidence": 0.92,
                "routed_team": "FINANCE",
                "reason": "Pay question.",
                "next_action": "ROUTE_TO_OTHER_TEAM",
            }))

    def test_rejects_extra_fields(self):
        with self.assertRaises(ClassificationError):
            parse_decision(json.dumps({
                "category": "INQUIRY",
                "confidence": 0.92,
                "routed_team": "HR",
                "reason": "Leave question.",
                "next_action": "ROUTE_TO_OTHER_TEAM",
                "confidence": 0.9,
            }))

    def test_rejects_invalid_category(self):
        with self.assertRaises(ClassificationError):
            parse_decision(json.dumps({
                "category": "QUESTION",
                "confidence": 0.92,
                "routed_team": "HR",
                "reason": "Leave question.",
                "next_action": "ROUTE_TO_OTHER_TEAM",
            }))

    def test_rejects_confidence_out_of_range(self):
        payload = {
            "category": "INQUIRY", "confidence": 1.2,
            "routed_team": "HR", "reason": "Leave question.",
            "next_action": "ROUTE_TO_OTHER_TEAM",
        }
        with self.assertRaises(ClassificationError):
            parse_decision(json.dumps(payload))

    def test_rejects_boolean_confidence(self):
        payload = {
            "category": "INQUIRY", "confidence": True,
            "routed_team": "HR", "reason": "Leave question.",
            "next_action": "ROUTE_TO_OTHER_TEAM",
        }
        with self.assertRaises(ClassificationError):
            parse_decision(json.dumps(payload))

    def test_rejects_empty_reason(self):
        with self.assertRaises(ClassificationError):
            parse_decision(json.dumps({
                "category": "COMPLAINT",
                "confidence": 0.92,
                "routed_team": "ACCOUNTING",
                "reason": "  ",
                "next_action": "ROUTE_TO_OTHER_TEAM",
            }))

    def test_rejects_malformed_json(self):
        with self.assertRaises(ClassificationError):
            parse_decision("{not json}")

    def test_rejects_empty_input(self):
        with self.assertRaises(ClassificationError):
            classify_and_route("  ")

    def test_rejects_overlong_input(self):
        with self.assertRaises(ClassificationError):
            classify_and_route("x" * 2001)

    @patch("palaw_classifier.ollama.chat")
    def test_single_model_call_returns_combined_decision(self, chat):
        chat.return_value = model_response({
            "category": "COMPLAINT",
            "confidence": 0.92,
            "routed_team": "ACCOUNTING",
            "reason": "The employee reports missing 13th-month pay.",
            "next_action": "ROUTE_TO_OTHER_TEAM",
        })
        result = classify_and_route("Bakit wala akong 13th month pay?")
        self.assertEqual(result["category"], "COMPLAINT")
        self.assertEqual(result["routed_team"], "ACCOUNTING")
        self.assertEqual(chat.call_count, 1)


if __name__ == "__main__":
    unittest.main()
