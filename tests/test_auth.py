from __future__ import annotations

from src.auth.jwt import JWTAuth, create_access_token, decode_token, hash_password, verify_password


class TestJWTAuth:
    def test_hash_and_verify(self) -> None:
        password = "my_secret_password"
        hashed = hash_password(password)
        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrong_password", hashed)

    def test_token_round_trip(self) -> None:
        token = create_access_token({"sub": "test_user", "role": "admin"})
        assert token is not None
        payload = decode_token(token)
        assert payload is not None
        assert payload.get("sub") == "test_user"
        assert payload.get("role") == "admin"

    def test_invalid_token(self) -> None:
        payload = decode_token("invalid.token.here")
        assert payload is None

    def test_register_and_login(self) -> None:
        auth = JWTAuth()
        user = auth.register_user("test_user", "test_password")
        assert user is not None
        assert user.get("username") == "test_user"

        token = auth.login("test_user", "test_password")
        assert token is not None

    def test_login_wrong_password(self) -> None:
        auth = JWTAuth()
        auth.register_user("user2", "correct_password")
        token = auth.login("user2", "wrong_password")
        assert token is None

    def test_get_user(self) -> None:
        auth = JWTAuth()
        auth.register_user("alice", "pass123")
        user = auth.get_user("alice")
        assert user is not None
        assert user["username"] == "alice"

    def test_get_nonexistent_user(self) -> None:
        auth = JWTAuth()
        user = auth.get_user("nobody")
        assert user is None
