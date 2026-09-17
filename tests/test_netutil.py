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
