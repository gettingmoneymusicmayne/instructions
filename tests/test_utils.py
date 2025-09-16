import os
import types
import numpy as np


def import_from_workspace(module_name: str, filename: str) -> types.ModuleType:
    """Import a Python file from the workspace by path to access its functions.

    We avoid installing the package and instead import by file location.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_hex_to_bgr_basic():
    utils = import_from_workspace("utils_module", os.path.abspath("utils.py"))
    assert utils.hex_to_bgr("#00FF00") == "0,255,0"
    assert utils.hex_to_bgr("00ff00") == "0,255,0"
    assert utils.hex_to_bgr("#ff0000") == "0,0,255"
    assert utils.hex_to_bgr("#0000ff") == "255,0,0"


def test_model_selection_scoring_prefers_engine(tmp_path):
    utils = import_from_workspace("utils_module2", os.path.abspath("utils.py"))
    # Create fake models
    (tmp_path / "yolov11n.engine").write_text("engine")
    (tmp_path / "yolov11s.pt").write_text("pt")
    (tmp_path / "yolov11x.pt").write_text("pt")

    candidates = utils.find_candidate_models([str(tmp_path)])
    assert any(p.endswith("yolov11n.engine") for p in candidates)

    best = utils.pick_best_model(candidates, default_dir=str(tmp_path))
    # Should prefer .engine even if a larger size .pt exists
    assert best.endswith("yolov11n.engine")


def test_pick_best_model_when_empty_uses_default(tmp_path):
    utils = import_from_workspace("utils_module3", os.path.abspath("utils.py"))
    expected = os.path.join(str(tmp_path), "yolov11n.pt")
    assert utils.pick_best_model([], default_dir=str(tmp_path)) == expected


def test_alpha_blend_center_basic():
    utils = import_from_workspace("utils_module4", os.path.abspath("utils.py"))
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    overlay = np.zeros((10, 10, 4), dtype=np.uint8)
    overlay[:, :, 1] = 255  # green
    overlay[:, :, 3] = 128  # 50% alpha
    out = utils.alpha_blend_center(frame, overlay)
    assert out.sum() > 0


def test_alpha_blend_center_clip_edges():
    utils = import_from_workspace("utils_module5", os.path.abspath("utils.py"))
    frame = np.zeros((5, 5, 3), dtype=np.uint8)
    overlay = np.zeros((10, 10, 4), dtype=np.uint8)
    overlay[:, :, :] = 0
    overlay[:, :, 2] = 255  # blue
    overlay[:, :, 3] = 255  # opaque
    out = utils.alpha_blend_center(frame, overlay)
    # Entire frame becomes blue
    assert np.all(out[:, :, 2] == 255)
    assert np.all(out[:, :, :2] == 0)

