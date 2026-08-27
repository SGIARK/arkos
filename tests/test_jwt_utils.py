"""Verifying somebody else's token, and minting our own cookie."""

import time

import jwt
import pytest

from harness_module import jwt_utils
from harness_module.jwt_utils import assert_secure_secrets, mint_session, read_session, verify_supabase

SUPABASE_SECRET = "test-supabase-secret-at-least-32-chars"


def _supabase(secret: str | None = None, **claims) -> str:
    """Mint a token shaped like the ones Supabase issues."""
    payload = {
        "sub": "8f1d4a02-0000-4000-8000-000000000001",
        "email": "a@example.com",
        "aud": "authenticated",
        "exp": int(time.time()) + 300,
        **claims,
    }
    return jwt.encode(payload, secret or SUPABASE_SECRET, algorithm="HS256")


class TestVerifySupabase:
    def test_round_trips_a_valid_token(self):
        claims = verify_supabase(_supabase())

        assert claims["sub"] == "8f1d4a02-0000-4000-8000-000000000001"
        assert claims["email"] == "a@example.com"

    def test_rejects_a_token_signed_with_another_secret(self):
        with pytest.raises(jwt.PyJWTError):
            verify_supabase(_supabase(secret="a-different-secret-entirely"))

    def test_rejects_an_expired_token(self):
        with pytest.raises(jwt.ExpiredSignatureError):
            verify_supabase(_supabase(exp=int(time.time()) - 1))

    def test_rejects_a_token_minted_for_another_audience(self):
        with pytest.raises(jwt.InvalidAudienceError):
            verify_supabase(_supabase(aud="someone-elses-app"))


class TestSessionCookie:
    def test_round_trips_what_we_signed(self):
        cookie, jti, _expires = mint_session("u-1", "a@example.com")
        claims = read_session(cookie)

        assert claims["sub"] == "u-1"
        assert claims["email"] == "a@example.com"
        # The handle the server revokes by; without it the cookie is refused.
        assert claims["jti"] == jti

    def test_a_cookie_we_did_not_sign_is_refused(self):
        forged = jwt.encode({"sub": "u-1", "iss": "buddy"}, "not-our-secret", algorithm="HS256")

        with pytest.raises(jwt.PyJWTError):
            read_session(forged)

    def test_a_supabase_token_is_not_a_session_cookie(self):
        """A Supabase token is refused where a session cookie is expected."""
        with pytest.raises(jwt.PyJWTError):
            read_session(_supabase())

    def test_an_expired_cookie_is_refused(self, monkeypatch):
        monkeypatch.setattr(jwt_utils.config, "get", lambda key, default=None: -1)

        with pytest.raises(jwt.ExpiredSignatureError):
            read_session(mint_session("u-1")[0])


class TestKeyCachePersistence:
    """A restart must not be cold: the endpoint goes away for minutes at a time,
    and an empty cache refuses every token."""

    JWKS = {
        "keys": [
            {
                "kid": "k1",
                "kty": "EC",
                "crv": "P-256",
                "alg": "ES256",
                "use": "sig",
                "x": "f83OJ3D2xF1Bg8vub9tLe1gHMzV76e8Tus9uPHvRVEU",
                "y": "x_FEzRu9m36HLN_tue659LNpXW6pCyStikYjKIWI5a0",
            }
        ]
    }

    def _client(self, monkeypatch, tmp_path, fetched):
        import jwt as pyjwt

        monkeypatch.setattr(jwt_utils, "jwks_url", lambda: "https://example.test/jwks.json")
        monkeypatch.setattr(jwt_utils, "_jwks_file", lambda: tmp_path / "jwks.json")
        jwt_utils.reset_jwks()
        client = jwt_utils._jwks()
        monkeypatch.setattr(client, "fetch_data", lambda: fetched.append(1) or self.JWKS)
        return client, pyjwt

    def test_a_refresh_writes_the_set_to_disk(self, monkeypatch, tmp_path):
        fetched = []
        self._client(monkeypatch, tmp_path, fetched)

        assert jwt_utils.refresh_jwks() is True
        assert (tmp_path / "jwks.json").exists()
        assert fetched == [1]

    def test_a_restart_primes_from_disk_without_touching_the_network(self, monkeypatch, tmp_path):
        fetched = []
        self._client(monkeypatch, tmp_path, fetched)
        jwt_utils.refresh_jwks()

        # The restart: a brand-new client that must never call out.
        jwt_utils.reset_jwks()
        client = jwt_utils._jwks()

        def forbidden():
            raise AssertionError("primed from disk must not fetch")

        monkeypatch.setattr(client, "fetch_data", forbidden)

        assert jwt_utils.prime_jwks_from_disk() is True
        token = jwt.encode({"sub": "u"}, "x" * 32, algorithm="HS256", headers={"kid": "k1"})
        assert jwt_utils._signing_key(token) is not None

    def test_a_corrupt_cache_file_is_not_fatal(self, monkeypatch, tmp_path):
        self._client(monkeypatch, tmp_path, [])
        (tmp_path / "jwks.json").write_text("{not json")

        assert jwt_utils.prime_jwks_from_disk() is False

    def test_no_cache_file_is_simply_a_cold_start(self, monkeypatch, tmp_path):
        self._client(monkeypatch, tmp_path, [])

        assert jwt_utils.prime_jwks_from_disk() is False


class TestAssertSecureSecrets:
    def test_raises_without_a_way_to_sign_sessions(self, monkeypatch):
        monkeypatch.delenv("BUDDY_SESSION_SECRET", raising=False)

        with pytest.raises(RuntimeError, match="BUDDY_SESSION_SECRET"):
            assert_secure_secrets()

    def test_raises_without_any_way_to_verify_a_token(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
        monkeypatch.setattr(jwt_utils, "jwks_url", lambda: None)

        with pytest.raises(RuntimeError, match="verify a Supabase token"):
            assert_secure_secrets()

    def test_a_project_url_alone_is_enough_to_verify(self, monkeypatch):
        """Asymmetric signing needs no shared secret, only somewhere to fetch keys."""
        monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
        monkeypatch.setattr(jwt_utils, "jwks_url", lambda: "https://ref.supabase.co/auth/v1/.well-known/jwks.json")

        assert assert_secure_secrets() is None

    def test_passes_when_both_are_set(self):
        assert assert_secure_secrets() is None

    def test_no_demo_bypass_exists(self, monkeypatch):
        """No env var excuses a missing session secret.

        There was once a demo mode that did. It is gone, and this pins that
        nothing takes its place: setting a plausible bypass flag changes nothing.
        """
        monkeypatch.delenv("BUDDY_SESSION_SECRET", raising=False)
        monkeypatch.setenv("DEMO_MODE", "1")

        with pytest.raises(RuntimeError):
            assert_secure_secrets()


def test_extract_bearer_takes_only_a_bearer_scheme():
    assert jwt_utils.extract_bearer("Bearer abc") == "abc"
    assert jwt_utils.extract_bearer("Basic dXNlcjpwYXNz") is None
    assert jwt_utils.extract_bearer(None) is None
    assert jwt_utils.extract_bearer("Bearer ") is None


def _publishing(signing, kid="test-kid"):
    """A stand-in JWKS client publishing one key, the way the real set does.

    Since 12.2.5 the lookup is by `kid` against a cached set: an unknown kid is
    refused rather than triggering a fetch, so the fake has to publish a SET
    rather than answer per token.
    """

    class Key:
        key_id = kid

        @property
        def key(self):
            return signing().key

    class Set:
        keys = [Key()]

    class Client:
        def get_jwk_set(self, refresh=False):
            return Set()

    return Client()


class TestAsymmetricTokens:
    """Supabase signs with a project key published at its JWKS endpoint."""

    @staticmethod
    def _es256_key():
        cryptography = pytest.importorskip("cryptography", reason="ES256 needs the cryptography package")
        from cryptography.hazmat.primitives.asymmetric import ec

        assert cryptography
        return ec.generate_private_key(ec.SECP256R1())

    def test_a_project_signed_token_verifies_against_the_published_key(self, monkeypatch):
        private_key = self._es256_key()
        token = jwt.encode(
            {"sub": "u-1", "aud": "authenticated", "exp": int(time.time()) + 300},
            private_key,
            algorithm="ES256",
            headers={"kid": "k1"},
        )

        class Signing:
            key = private_key.public_key()

        monkeypatch.setattr(jwt_utils, "_jwks", lambda: _publishing(Signing, kid="k1"))

        assert verify_supabase(token)["sub"] == "u-1"

    def test_a_token_signed_by_someone_else_is_refused(self, monkeypatch):
        mine = self._es256_key()
        theirs = self._es256_key()
        token = jwt.encode(
            {"sub": "u-1", "aud": "authenticated", "exp": int(time.time()) + 300},
            theirs,
            algorithm="ES256",
            headers={"kid": "k1"},
        )

        class Signing:
            key = mine.public_key()

        monkeypatch.setattr(jwt_utils, "_jwks", lambda: _publishing(Signing, kid="k1"))

        with pytest.raises(jwt.InvalidSignatureError):
            verify_supabase(token)

    def test_an_unknown_kid_is_refused_without_a_fetch(self, monkeypatch):
        """12.2.5: `POST /auth/session` is public and the header is the caller's,
        so a miss must cost a dictionary lookup, not a blocking JWKS fetch on the
        pool shared with blob IO and every sandbox call. Verification NEVER
        fetches — the background tick is the only thing that does."""
        private_key = self._es256_key()
        token = jwt.encode(
            {"sub": "u-1", "aud": "authenticated", "exp": int(time.time()) + 300},
            private_key,
            algorithm="ES256",
            headers={"kid": "a-kid-nobody-published"},
        )

        class Signing:
            key = private_key.public_key()

        fetches = []
        client = _publishing(Signing, kid="k1")
        original = client.get_jwk_set

        def counting(refresh=False):
            if refresh:
                fetches.append(1)
            return original(refresh)

        client.get_jwk_set = counting
        monkeypatch.setattr(jwt_utils, "_jwks", lambda: client)

        with pytest.raises(jwt.InvalidKeyError):
            verify_supabase(token)
        with pytest.raises(jwt.InvalidKeyError):
            verify_supabase(token)

        assert fetches == [], "verification must never fetch; the tick owns that"

    def test_an_asymmetric_token_with_nowhere_to_fetch_keys_is_refused(self, monkeypatch):
        token = jwt.encode(
            {"sub": "u-1", "aud": "authenticated", "exp": int(time.time()) + 300},
            self._es256_key(),
            algorithm="ES256",
        )
        monkeypatch.setattr(jwt_utils, "_jwks", lambda: None)

        with pytest.raises(jwt.InvalidKeyError):
            verify_supabase(token)

    def test_an_algorithm_we_do_not_accept_is_refused(self):
        """`none` is the classic forged token, and anything unlisted is treated the same."""
        unsigned = jwt.encode({"sub": "u-1", "aud": "authenticated"}, key=None, algorithm=None)

        with pytest.raises(jwt.InvalidAlgorithmError):
            verify_supabase(unsigned)

    def test_the_jwks_url_is_derived_from_the_project(self):
        url = jwt_utils.jwks_url()

        assert url is None or url.endswith("/auth/v1/.well-known/jwks.json")
