"""Tests for OpenTelemetry tracing and event logging."""

import json
import os
from pathlib import Path

import pytest
from src.monitor.tracing import EventLogger, get_tracer, get_event_logger


def test_event_logger(tmp_path):
    log_file = tmp_path / "test_traces.jsonl"
    logger = EventLogger(log_path=str(log_file))
    logger.log("test_event", {"key": "value"})
    lines = Path(log_file).read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["event"] == "test_event"
    assert data["properties"]["key"] == "value"


def test_log_request(tmp_path):
    log_file = tmp_path / "req.jsonl"
    logger = EventLogger(log_path=str(log_file))
    logger.log_request("POST", "/api/chat", 200, 150.5)
    data = json.loads(Path(log_file).read_text().strip())
    assert data["properties"]["method"] == "POST"
    assert data["properties"]["status"] == 200


def test_log_llm_call(tmp_path):
    log_file = tmp_path / "llm.jsonl"
    logger = EventLogger(log_path=str(log_file))
    logger.log_llm_call("llama3.2:1b", 100, 50, 500.0)
    data = json.loads(Path(log_file).read_text().strip())
    assert data["properties"]["total_tokens"] == 150


def test_get_tracer():
    tracer = get_tracer()
    assert tracer is not None


def test_event_logger_no_path():
    logger = EventLogger(log_path="")
    logger.log("test", {})  # Should not raise
