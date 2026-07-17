"""Tests for LLM-as-judge evaluation."""

import pytest
from src.eval.judge import LLMJudge, EvalSuite


@pytest.fixture
def judge():
    return LLMJudge(judge_model="test-model")


@pytest.fixture
def eval_suite():
    return EvalSuite()


def test_judge_creation(judge):
    assert judge.judge_model == "test-model"


def test_judge_default_model():
    j = LLMJudge()
    assert j.judge_model == "qwen3.5:latest"


def test_suite_creation(eval_suite):
    eval_suite.add_test("Say hello", expected="Hello", context="Chat")
    eval_suite.add_test("What's 2+2?", expected="4", context="Math")
    assert len(eval_suite.results) == 2


def test_suite_defaults():
    suite = EvalSuite()
    suite.add_test("Hi")
    assert suite.results[0]["expected"] == ""
    assert suite.results[0]["context"] == ""


def test_suite_summary(eval_suite):
    eval_suite.add_test("Hi")
    s = eval_suite.summary()
    assert "total" in s
    assert s["total"] == 1
    assert s["avg_score"] == 0
