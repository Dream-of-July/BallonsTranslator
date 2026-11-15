"""Protocol framing; this module intentionally has no Qt/application imports."""
from __future__ import annotations

import json
import os
import sys
import threading
from typing import TextIO


class ProtocolError(Exception):
    def __init__(self, code: str, message: str, data: dict | None = None) -> None:
        super().__init__(message)
        self.code, self.data = code, data


def isolate_stdout() -> TextIO:
    """Reserve the original pipe before imports can write diagnostics to fd 1."""
    sys.stdout.flush()
    stream = os.fdopen(os.dup(1), 'w', encoding='utf-8', buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return stream


class Writer:
    """Serialize writes from the reader, Qt thread and connection-test workers.

    >>> import io
    >>> sink = io.StringIO()
    >>> Writer(sink).send({'id': 1, 'result': True})
    >>> json.loads(sink.getvalue())['result']
    True
    """
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.lock = threading.Lock()

    def send(self, value: dict) -> None:
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False)
        with self.lock:
            self.stream.write(encoded + '\n')
            self.stream.flush()


def decode(line: str) -> dict:
    try:
        value = json.loads(line, parse_constant=_invalid_constant)
    except (ValueError, TypeError):
        raise ProtocolError('bad_request', 'Malformed JSON.') from None
    if not isinstance(value, dict):
        raise ProtocolError('bad_request', 'Request must be an object.')
    return value


def _invalid_constant(value: str) -> None:
    raise ValueError('Non-finite number')
