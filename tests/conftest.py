import socket
import threading
import time

import pytest
import uvicorn

from bench.mock_server import make_app


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_url():
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(make_app(slots=4, step_s=0.001), host="127.0.0.1", port=port,
                                           log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=5)
