from __future__ import annotations

import tempfile
from pathlib import Path

from src.auth.jwt import UserStore, create_token, verify_token


class TestAuth:
    def test_create_and_verify_token(self) -> None:
        token = create_token(1, "test_user")
        assert token is not None
        assert token.count(".") == 2
        payload = verify_token(token)
        assert payload is not None
        assert payload["sub"] == 1
        assert payload["username"] == "test_user"

    def test_invalid_token(self) -> None:
        payload = verify_token("invalid.token.here")
        assert payload is None

    def test_create_and_authenticate_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserStore(str(Path(tmpdir) / "test_users.db"))
            user = store.create_user("alice", "secret123")
            assert user.username == "alice"
            assert user.id > 0

            authed = store.authenticate("alice", "secret123")
            assert authed is not None
            assert authed.username == "alice"

            failed = store.authenticate("alice", "wrong_password")
            assert failed is None

    def test_duplicate_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserStore(str(Path(tmpdir) / "test_dup.db"))
            store.create_user("bob", "pass")
            try:
                store.create_user("bob", "pass2")
                assert False, "Should have raised ValueError"
            except ValueError:
                pass

    def test_get_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserStore(str(Path(tmpdir) / "test_get.db"))
            user = store.create_user("charlie", "pass")
            found = store.get_user(user.id)
            assert found is not None
            assert found.username == "charlie"

    def test_get_nonexistent_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserStore(str(Path(tmpdir) / "test_nonexist.db"))
            found = store.get_user(999)
            assert found is None

    def test_user_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserStore(str(Path(tmpdir) / "test_exists.db"))
            store.create_user("dave", "pass")
            assert store.user_exists("dave") is True
            assert store.user_exists("nobody") is False
