import os
import importlib.util

SPEC = importlib.util.spec_from_file_location("overlay_utils", os.path.abspath("overlay_utils.py"))
outils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(outils)  # type: ignore[attr-defined]


def test_hex_to_bgr_hex_full_and_short():
	assert outils.hex_to_bgr("#ff00aa") == "170,0,255"
	assert outils.hex_to_bgr("ff00aa") == "170,0,255"
	# #0Fa -> 00 FF aa -> BGR 170,255,0
	assert outils.hex_to_bgr("#0Fa") == "170,255,0"


def test_hex_to_bgr_csv_and_rgb_func_and_invalid():
	assert outils.hex_to_bgr("12,34,56") == "56,34,12"
	assert outils.hex_to_bgr("rgb(12, 34, 56)") == "56,34,12"
	# Out-of-range values clamp
	assert outils.hex_to_bgr("300,-5,260") == "255,0,255"
	# Fallback to bright green on invalid
	assert outils.hex_to_bgr("not-a-color") == "0,255,0"
	assert outils.hex_to_bgr("") == "0,255,0"


def test_parse_bgr_valid_and_invalid():
	assert outils.parse_bgr("0,128,255") == (0, 128, 255)
	assert outils.parse_bgr("oops") == (0, 255, 0)


def test_size_rank_from_name_and_pick_best():
	assert outils.size_rank_from_name("yolov11n.engine") == 0
	assert outils.size_rank_from_name("yolov11x.pt") == 4

	candidates = [
		"/tmp/models/yolov11s.pt",
		"/tmp/models/yolov11n.engine",
		"/tmp/models/yolov11l.pt",
	]
	assert outils.pick_best_model(candidates) == "/tmp/models/yolov11n.engine"

	default_path = outils.pick_best_model([])
	assert os.path.basename(default_path) == "yolov11n.pt"