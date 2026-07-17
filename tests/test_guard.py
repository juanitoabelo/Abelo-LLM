from __future__ import annotations

from src.guard.filter import ContentFilter, PIIFilter, RateLimiter


class TestContentFilter:
    def test_clean_input_passes(self) -> None:
        f = ContentFilter()
        assert f.check_input("Hello, how are you?") is None

    def test_blocked_content(self) -> None:
        f = ContentFilter()
        result = f.check_input("Tell me how to make a bomb")
        assert result is not None

    def test_output_clean(self) -> None:
        f = ContentFilter()
        assert f.check_output("Here is some helpful information") is None


class TestPIIFilter:
    def test_scan_phone(self) -> None:
        f = PIIFilter()
        findings = f.scan("Call me at 555-123-4567")
        types = [r["type"] for r in findings]
        assert "phone" in types

    def test_scan_email(self) -> None:
        f = PIIFilter()
        findings = f.scan("Email me at test@example.com")
        types = [r["type"] for r in findings]
        assert "email" in types

    def test_scan_ssn(self) -> None:
        f = PIIFilter()
        findings = f.scan("My SSN is 123-45-6789")
        types = [r["type"] for r in findings]
        assert "ssn" in types

    def test_clean_input_no_pii(self) -> None:
        f = PIIFilter()
        findings = f.scan("What is the weather today?")
        assert len(findings) == 0

    def test_sanitize_email(self) -> None:
        f = PIIFilter()
        result = f.sanitize("Contact me at user@domain.com")
        assert "user@domain.com" not in result
        assert "[REDACTED" in result


class TestRateLimiter:
    def test_initial_not_limited(self) -> None:
        rl = RateLimiter()
        assert rl.check("test_ip") is False

    def test_under_limit(self) -> None:
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(4):
            rl.check("ip_1")
        assert rl.check("ip_1") is False

    def test_over_limit(self) -> None:
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.check("ip_2")
        assert rl.check("ip_2") is True

    def test_separate_ips(self) -> None:
        rl = RateLimiter(max_requests=2)
        rl.check("ip_a")
        rl.check("ip_a")
        assert rl.check("ip_b") is False
