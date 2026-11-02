import pytest

from kalshicombos import http


class _Resp:
    def __init__(self, code, body=None, headers=None):
        self.status_code, self._body, self.headers = code, body or {}, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_retries_through_rate_limits(monkeypatch):
    replies = iter([_Resp(429, headers={"Retry-After": "0"}), _Resp(429), _Resp(200, {"ok": 1})])
    monkeypatch.setattr(http._SESSION, "get", lambda *a, **k: next(replies))
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    assert http.get_json("https://x")[0] == {"ok": 1}


def test_gives_up_eventually(monkeypatch):
    monkeypatch.setattr(http._SESSION, "get", lambda *a, **k: _Resp(429))
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="429"):
        http.get_json("https://x", retries=3)


def test_backoff_grows_and_is_capped():
    assert http._backoff(0, None) < http._backoff(4, None) <= http.MAX_WAIT
    assert http._backoff(10, None) <= http.MAX_WAIT
    assert http._backoff(0, "5") == 5
