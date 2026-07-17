"""Tests for DPO training pipeline."""

import json
import sqlite3
from pathlib import Path

import pytest
from src.training.dpo import DPOTrainer


def _make_db(db_path, pairs):
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, user_message TEXT, assistant_response TEXT, rating INTEGER, model TEXT, created_at REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS preferences (id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT, chosen TEXT, rejected TEXT, model TEXT, created_at REAL)")
    for msg, resp, rating in pairs:
        conn.execute("INSERT INTO feedback (user_message, assistant_response, rating, model, created_at) VALUES (?, ?, ?, ?, ?)",
                     (msg, resp, rating, "llama", 1000.0))
    conn.commit()
    conn.close()


@pytest.fixture
def trainer(tmp_path):
    db_path = tmp_path / "test_feedback.db"
    _make_db(db_path, [
        ("hello", "hi there!", 1),
        ("hello", "hey", -1),
        ("help", "no", -1),
        ("help", "here is help", 1),
    ])
    return DPOTrainer(feedback_db=str(db_path), output_dir=str(tmp_path / "dpo"))


def test_export_preference_pairs(trainer):
    pairs = trainer.export_preference_pairs()
    assert len(pairs) >= 2
    for p in pairs:
        assert "prompt" in p
        assert p["chosen"] and p["rejected"]


def test_format_for_trl(trainer, tmp_path):
    pairs = [{"prompt": "hi", "chosen": "hello", "rejected": "bye"}]
    path = trainer.format_for_trl(pairs)
    assert Path(path).exists()
    lines = Path(path).read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["prompt"] == "hi"


def test_format_for_modelfile(trainer, tmp_path):
    pairs = [{"prompt": "hi", "chosen": "hello", "rejected": "bye"}]
    path = trainer.format_for_modelfile(pairs)
    assert Path(path).exists()


def test_get_stats(trainer):
    stats = trainer.get_stats()
    assert stats["total_feedback"] >= 4
    assert stats["thumbs_up"] >= 2
