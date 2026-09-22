from tools.netutil import validate_public_url


def test_private_urls_are_blocked():
    for url in ("http://127.0.0.1:11434", "http://10.0.0.1", "http://192.168.1.1"):
        try:
            validate_public_url(url)
        except ValueError:
            pass
        else:
            raise AssertionError(url)


def test_public_literal_ip_is_accepted():
    assert validate_public_url("https://93.184.216.34").startswith("https://")


def test_fetch_bytes_enforces_total_body_deadline(monkeypatch):
    import tools.netutil as netutil

    class FakeResponse:
        is_redirect = False
        is_permanent_redirect = False
        headers = {"Content-Type": "text/plain", "Content-Length": "2"}
        encoding = "utf-8"

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=65536):
            yield b"a"
            yield b"b"

        def close(self):
            return None

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

        def close(self):
            return None

    ticks = iter([0.0, 0.1, 0.5, 1.2])
    monkeypatch.setattr(netutil.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(netutil, "validate_public_url", lambda url, allow_private=False: url)
    monkeypatch.setattr(netutil.requests, "Session", FakeSession)

    try:
        netutil.fetch_bytes("https://example.test/", timeout=1.0, allowed_types={"text/plain"})
    except TimeoutError as exc:
        assert "total timeout" in str(exc)
    else:
        raise AssertionError("expected total body deadline to fire")
