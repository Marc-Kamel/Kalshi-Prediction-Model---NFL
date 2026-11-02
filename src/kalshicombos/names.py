"""Player-name normalisation so 'A.J. Barner', 'AJ Barner' and 'Luther Burden III' / 'Luther Burden' match."""
from __future__ import annotations

import re
import unicodedata

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("’", "")
    s = s.replace("-", " ")
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()
