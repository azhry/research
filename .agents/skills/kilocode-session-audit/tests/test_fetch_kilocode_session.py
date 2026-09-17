from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "fetch_kilocode_session.py"
SPEC = importlib.util.spec_from_file_location("fetch_kilocode_session", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
fetcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fetcher
SPEC.loader.exec_module(fetcher)


class SecretSanitizationTests(unittest.TestCase):
    def test_redacts_prefixed_secret_assignments(self) -> None:
        source = "LINEAR_API_KEY=linear-value GITHUB_TOKEN=github-value GOOGLE_STITCH_API_KEY=google-value"
        result = fetcher.sanitize(source)

        self.assertEqual(
            result,
            "LINEAR_API_KEY=[REDACTED] GITHUB_TOKEN=[REDACTED] GOOGLE_STITCH_API_KEY=[REDACTED]",
        )

    def test_redacts_nested_secret_keys(self) -> None:
        result = fetcher.sanitize({"metadata": {"authorization": "Bearer value", "safe": "visible"}})

        self.assertEqual(result, {"metadata": {"authorization": "[REDACTED]", "safe": "visible"}})


class SessionSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database = Path(self.temporary_directory.name) / "kilo.db"

    def create_database(
        self, sessions: list[tuple[str, str]], include_title: bool = True
    ) -> None:
        connection = sqlite3.connect(self.database)
        if include_title:
            connection.execute(
                "CREATE TABLE session (id TEXT, title TEXT, time_created INTEGER, time_updated INTEGER)"
            )
            connection.executemany(
                "INSERT INTO session (id, title, time_created, time_updated) VALUES (?, ?, 1, 2)",
                sessions,
            )
        else:
            connection.execute("CREATE TABLE session (id TEXT, time_created INTEGER)")
        connection.execute(
            "CREATE TABLE message (id TEXT, session_id TEXT, time_created INTEGER, time_updated INTEGER, data TEXT)"
        )
        for session_id, _ in sessions:
            connection.execute(
                "INSERT INTO message VALUES (?, ?, 3, 4, ?)",
                (
                    f"message-{session_id}",
                    session_id,
                    json.dumps({"text": "visible", "token": "secret"}),
                ),
            )
        connection.commit()
        connection.close()

    def run_fetcher(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(MODULE_PATH), *arguments, "--database", str(self.database)],
            capture_output=True,
            check=False,
            text=True,
        )

    def test_fetches_unique_session_by_exact_title(self) -> None:
        self.create_database([("session-1", "Investigate login")])

        result = self.run_fetcher("--session-title", "Investigate login", "--page", "1")

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["session_id"], "session-1")
        self.assertEqual(payload["record_count"], 2)
        self.assertNotIn("secret", result.stdout)

    def test_preserves_session_id_lookup(self) -> None:
        self.create_database([("session-1", "Investigate login")])

        result = self.run_fetcher("--session-id", "session-1")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["session_id"], "session-1")

    def test_title_lookup_is_case_sensitive(self) -> None:
        self.create_database([("session-1", "Investigate login")])

        result = self.run_fetcher("--session-title", "investigate login")

        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["error"], "session title not found")

    def test_rejects_ambiguous_title(self) -> None:
        self.create_database([
            ("session-1", "Investigate login"),
            ("session-2", "Investigate login"),
        ])

        result = self.run_fetcher("--session-title", "Investigate login")

        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["error"], "session title is ambiguous")
        self.assertNotIn("session-1", result.stdout)
        self.assertNotIn("session-2", result.stdout)

    def test_rejects_schema_without_title_column(self) -> None:
        self.create_database([], include_title=False)

        result = self.run_fetcher("--session-title", "Investigate login")

        self.assertEqual(result.returncode, 2)
        self.assertEqual(
            json.loads(result.stdout)["error"],
            "session title lookup unsupported by schema",
        )

    def test_rejects_schema_without_session_table(self) -> None:
        connection = sqlite3.connect(self.database)
        connection.execute("CREATE TABLE message (id TEXT, session_id TEXT)")
        connection.commit()
        connection.close()

        result = self.run_fetcher("--session-title", "Investigate login")

        self.assertEqual(result.returncode, 2)
        self.assertEqual(
            json.loads(result.stdout)["error"],
            "session title lookup unsupported by schema",
        )

    def test_requires_exactly_one_selector(self) -> None:
        self.create_database([("session-1", "Investigate login")])

        missing = self.run_fetcher()
        duplicate = self.run_fetcher(
            "--session-id", "session-1", "--session-title", "Investigate login"
        )

        self.assertEqual(missing.returncode, 2)
        self.assertIn(
            "one of the arguments --session-id --session-title is required",
            missing.stderr,
        )
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("not allowed with argument", duplicate.stderr)

    def test_reuses_snapshot_for_title_resolution(self) -> None:
        self.create_database([("session-1", "Original title")])
        snapshot = Path(self.temporary_directory.name) / "snapshot.db"

        first = self.run_fetcher(
            "--session-title", "Original title", "--snapshot-file", str(snapshot)
        )
        connection = sqlite3.connect(self.database)
        connection.execute("UPDATE session SET title = 'Changed title'")
        connection.commit()
        connection.close()
        second = self.run_fetcher(
            "--session-title", "Original title", "--snapshot-file", str(snapshot)
        )

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(json.loads(second.stdout)["session_id"], "session-1")


if __name__ == "__main__":
    unittest.main()
