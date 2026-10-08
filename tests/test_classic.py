"""classic engine tests.

Engine-level tests (palette/resize/OPQ) need faiss+skimage; they skip when
those are absent (CI). API-shape tests run everywhere by importing the route
source only.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

try:
    import faiss  # noqa: F401
    import skimage  # noqa: F401

    _HAS_ENGINE_DEPS = True
except ImportError:
    _HAS_ENGINE_DEPS = False


def test_tasks_registry_shape():
    path = ROOT / "classic" / "engine.py"
    spec = importlib.util.spec_from_file_location("classic_engine_src", path)
    mod = importlib.util.module_from_spec(spec)
    # py3.11 dataclasses resolve cls.__module__ via sys.modules at class creation
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError:
        pytest.skip("engine deps missing")
    assert set(mod.TASKS) == {"color", "colorPalette"}
    assert mod.MAX_BATCH == 64


@pytest.mark.skipif(not _HAS_ENGINE_DEPS, reason="faiss/skimage not installed")
def test_rgb2hex_variants():
    from classic.color import rgb2hex, rgb2hex1

    # np.round uses banker's rounding at .5 (127.5 -> 128): tools-identical
    assert rgb2hex([1.0, 0.0, 0.5]) == "#ff0080"
    assert rgb2hex1([255.0, 0.0, 127.4]) == "#ff007f"


@pytest.mark.skipif(not _HAS_ENGINE_DEPS, reason="faiss/skimage not installed")
def test_resize_image_bytes_long_short_side_and_small_keep():
    import io

    import numpy as np
    from PIL import Image

    from classic.color import resize_image_bytes

    def png_bytes(w, h):
        buf = io.BytesIO()
        Image.fromarray((np.random.rand(h, w, 3) * 255).astype("uint8")).save(buf, "PNG")
        return buf.getvalue()

    out = resize_image_bytes(png_bytes(200, 100), 112)  # short side target
    assert Image.open(io.BytesIO(out)).size == (224, 112)
    out = resize_image_bytes(png_bytes(200, 100), 400, inverse=True)  # long side target
    assert Image.open(io.BytesIO(out)).size == (400, 200)
    small = resize_image_bytes(png_bytes(50, 30), 112, if_smaller_no_resize=True)
    assert Image.open(io.BytesIO(small)).size == (50, 30)  # kept original size


@pytest.mark.skipif(not _HAS_ENGINE_DEPS, reason="faiss/skimage not installed")
def test_color_engine_end_to_end():
    import io

    import numpy as np
    from PIL import Image

    from classic.engine import classify_one

    buf = io.BytesIO()
    rng = np.random.RandomState(7)
    Image.fromarray((rng.rand(64, 80, 3) * 255).astype("uint8")).save(buf, "PNG")
    b = buf.getvalue()

    color = classify_one("color", b)
    assert len(color["features"]) == 81
    # tools opqCode format: space-joined 'code_index' string (ES colorCodes verbatim)
    parts = color["opqCode"].split()
    assert len(parts) == 3
    assert all(p.split("_")[1] == str(i) for i, p in enumerate(parts))
    assert all(int(p.split("_")[0]) >= 0 for p in parts)

    palette = classify_one("colorPalette", b)
    assert len(palette["hexColors"]) == 6
    hex_str, count = palette["hexColors"][0]
    assert hex_str.startswith("#") and len(hex_str) == 7
    assert isinstance(count, int)


@pytest.mark.skipif(not _HAS_ENGINE_DEPS, reason="faiss/skimage not installed")
def test_quantizer_uses_shipped_codebooks():
    import numpy as np

    from classic.color import OpqQuantizer

    q = OpqQuantizer(str(ROOT / "models" / "color"))
    features = np.zeros((1, 81), dtype="float32")
    code = q.quantize(features)
    parts = code.split()
    assert len(parts) == 3
    assert all(p.split("_")[1] == str(i) for i, p in enumerate(parts))
