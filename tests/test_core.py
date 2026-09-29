"""Small, CPU-only checks on real SQLite behavior and training input contracts."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from sql_agent_rl.config import load_recipe, load_tasks, make_config, pad_training_rows, require_complete_actions
from sql_agent_rl.sql_safety import execute_observation, read_only_connection, resolve_database


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "school.sqlite"
        con = sqlite3.connect(self.db)
        con.executescript("CREATE TABLE scores (name TEXT, score INTEGER); INSERT INTO scores VALUES ('a',1),('b',2),('c',3);")
        con.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_select_and_preview_are_bounded(self):
        result = execute_observation(self.db, "SELECT * FROM scores ORDER BY score", preview_rows=2)
        self.assertEqual(result.status, "success")
        self.assertEqual(json.loads(result.result_preview), [["a", 1], ["b", 2]])
        self.assertTrue(result.result_truncated)

    def test_write_and_attach_are_denied_without_altering_data(self):
        for sql in ("DELETE FROM scores", "DROP TABLE scores", "ATTACH ':memory:' AS another"):
            with self.subTest(sql=sql):
                self.assertEqual(execute_observation(self.db, sql).status, "policy_violation")
        con = read_only_connection(self.db)
        try:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM scores").fetchone()[0], 3)
        finally:
            con.close()

    def test_actual_sqlite_deadline_interrupts_recursive_query(self):
        result = execute_observation(self.db, "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT SUM(n) FROM x", timeout=0.01)
        self.assertEqual(result.status, "execution_timeout")

    def test_bad_sql_is_an_observation(self):
        result = execute_observation(self.db, "SELECT missing_column FROM scores")
        self.assertEqual(result.status, "sql_error")
        self.assertIn("missing_column", result.raw_message)

    def test_database_cannot_escape_task_root(self):
        inner = self.root / "data"
        inner.mkdir()
        with self.assertRaises(ValueError):
            resolve_database(inner, {"db_id": "school", "db_relative_path": "../school.sqlite"})

    def test_padding_preserves_every_original_question(self):
        rows = [{"source_id": f"task:{i:05d}"} for i in range(6563)]
        padded, ids = pad_training_rows(rows)
        self.assertEqual(len(padded), 6592)
        self.assertEqual(padded[:6563], rows)
        self.assertEqual(ids, [r["source_id"] for r in rows[:29]])
        self.assertEqual(len(rows), 6563)

    def test_exact_joint_action_selection(self):
        require_complete_actions(1, ["write_query"])
        require_complete_actions(3, ["write_query", "rewrite_query", "rewrite_query"])
        for names in (["rewrite_query"], ["write_query", "check_query"], ["write_query"]):
            with self.assertRaises(RuntimeError):
                require_complete_actions(2, names)

    def test_task_payload_does_not_accept_reference_sql(self):
        (self.root / "splits").mkdir()
        row = {"source_id": "x", "question": "Count scores", "db_id": "school", "split": "train", "db_relative_path": "school.sqlite"}
        path = self.root / "splits/train.jsonl"
        path.write_text(json.dumps(row), encoding="utf-8")
        self.assertEqual(load_tasks(self.root, "train"), [row])
        row["query"] = "SELECT COUNT(*) FROM scores"
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_tasks(self.root, "train")

    def test_split_identity_and_duplicate_ids_are_rejected(self):
        (self.root / "splits").mkdir()
        row = {"source_id": "x", "question": "Count", "db_id": "school", "split": "heldout", "db_relative_path": "school.sqlite"}
        path = self.root / "splits/train.jsonl"
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_tasks(self.root, "train")
        row["split"] = "train"
        path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_tasks(self.root, "train")

    def test_recipe_paths_and_frequencies(self):
        recipe = load_recipe()
        config = make_config(recipe, self.root / "data", self.root / "model", self.root / "run")
        self.assertEqual(config["trainer"]["test_freq"], 52)
        self.assertEqual(config["trainer"]["save_freq"], 52)
        self.assertEqual(config["data"]["train_files"], str(self.root / "data/splits/train.parquet"))
        self.assertEqual(recipe["verl"]["actor_rollout_ref"]["model"]["path"], "models/qwen2.5-coder-1.5b")

    def test_dry_run_imports_no_gpu_stack_and_creates_no_run(self):
        result = subprocess.run([sys.executable, "-m", "sql_agent_rl.train", "--dry-run", "--run-dir", str(self.root / "not-created")], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["mode"], "configuration_preview_only")
        self.assertFalse((self.root / "not-created").exists())


if __name__ == "__main__":
    unittest.main()
