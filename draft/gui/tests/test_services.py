import sqlite3
import tempfile
import unittest
from pathlib import Path

from services import PaLawEngine, TicketRepository


class SubmissionValidationTests(unittest.TestCase):
    def test_accepts_valid_submission(self):
        choice, description = PaLawEngine.validate_submission(
            "2", "My supervisor repeatedly changes my approved work schedule."
        )
        self.assertEqual(choice, "2")
        self.assertTrue(description.startswith("My supervisor"))

    def test_rejects_invalid_nature(self):
        with self.assertRaises(ValueError):
            PaLawEngine.validate_submission("9", "This is a sufficiently long concern.")

    def test_rejects_short_description(self):
        with self.assertRaises(ValueError):
            PaLawEngine.validate_submission("1", "Too short")

    def test_rejects_overlong_description(self):
        with self.assertRaises(ValueError):
            PaLawEngine.validate_submission("1", "x" * 10001)


class TicketRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "tickets.db"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE tickets (
                ticket_id TEXT PRIMARY KEY, date_created TEXT,
                classification TEXT, priority TEXT, complexity TEXT,
                primary_department TEXT, status TEXT, assigned_to TEXT,
                draft_answer TEXT)""")
            conn.executemany("INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                ("T-LOW", "2026-01-01", "INQUIRY", "LOW", "NOT COMPLEX", "HR", "OPEN", "HR", None),
                ("T-URGENT", "2026-01-02", "COMPLAINT", "URGENT", "COMPLEX", "LEGAL", "OPEN", "Legal", None),
            ])
        self.repository = TicketRepository(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_priority_order(self):
        rows = self.repository.list_priority_sorted()
        self.assertEqual([row["ticket_id"] for row in rows], ["T-URGENT", "T-LOW"])

    def test_get_by_id(self):
        row = self.repository.get_by_id("T-LOW")
        self.assertEqual(row["primary_department"], "HR")

    def test_missing_ticket_returns_none(self):
        self.assertIsNone(self.repository.get_by_id("NOT-FOUND"))


if __name__ == "__main__":
    unittest.main()
