import io
import os

import pytest
from PIL import Image

# Never load the heavy model during tests.
os.environ.setdefault("WARMUP", "0")


@pytest.fixture
def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (12, 12), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as c:
        yield c
