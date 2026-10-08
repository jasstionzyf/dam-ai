"""Classic (non-NN) image algorithms ported from the tools codebase.

Source of truth: /data/projects/model-infer-api/vcgImageAI/subProjects/
  - mcsearch/colorModel.py   -> ColorModelV2, ColorPaletteModel
  - featuresRetrive/quantizers.py -> OPQFeaturesQuantizer
  - souJpg/comm helpers      -> resize/rgb2hex/l2-norm behaviour (container build)

Numeric behaviour is kept byte-compatible with the tools container
(faiss-cpu 1.9.0.post1 / numpy 1.26.4 / scikit-image 0.24.0): same resize
target/JPEG re-encode, same LAB palette, same smoothing, same OPQ codebooks
(models/color/*.model copied verbatim from the running tools cache, never
retrained).
"""

from __future__ import annotations

import io
import logging
import time

import faiss
import numpy as np
from PIL import Image
from skimage.color import hsv2rgb, rgb2lab
from sklearn import preprocessing
from sklearn.metrics import euclidean_distances

logger = logging.getLogger("damai.classic")

PALETTE_NUM_HUES = 8
PALETTE_SAT_RANGE = 3
PALETTE_LIGHT_RANGE = 3
PALETTE_SIGMA = 20
FEATURE_DIM = 81  # 8 hues x 9 rows + 9 grays
FEATURE_RESIZE_TARGET = 112
PALETTE_RESIZE_TARGET = 400

DEFAULT_PQ_SUBSPACES = 3
DEFAULT_PQ_BITS = 10


def rgb2hex(rgb_number) -> str:
    """0..1 float rgb -> '#rrggbb' (tools souJpg.comm.imageUtils.rgb2hex)."""
    return "#%02x%02x%02x" % tuple(int(np.round(val * 255)) for val in rgb_number)


def rgb2hex1(rgb_number) -> str:
    """0..255 rgb -> '#rrggbb' (tools mcsearch util.rgb2hex1)."""
    return "#%02x%02x%02x" % tuple(int(np.round(val)) for val in rgb_number)


def l2_norm_and_round(feature_array: np.ndarray, norm: str | None = "l2") -> np.ndarray:
    """tools souJpg.comm.mathUtils.l2NormAndRound (sklearn normalize + around3)."""
    assert norm in ("l1", "l2", "hellinger", None)
    eps = 1e-7
    if norm is not None:
        if norm == "hellinger":
            feature_array = feature_array / (feature_array.sum(axis=1, keepdims=True) + eps)
            feature_array = np.sqrt(feature_array)
        else:
            feature_array = preprocessing.normalize(feature_array, norm=norm)
    return np.around(feature_array, 3)


def resize_image_bytes(
    image_bytes: bytes,
    target_size: int,
    inverse: bool = False,
    if_smaller_no_resize: bool = False,
) -> bytes:
    """tools container build of souJpg.comm.imageUtils.resizeImageBytes.

    JPEG re-encode with PIL defaults (no quality kwarg in the tools container
    version -> 75), long side = target_size when inverse else short side.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        raw_w, raw_h = img.size
        new_w, new_h = raw_w, raw_h
        if if_smaller_no_resize and max(raw_h, raw_w) < target_size:
            pass  # keep original size
        else:
            if inverse:
                if raw_h > raw_w:
                    new_h = target_size
                    new_w = (new_h / raw_h) * raw_w
                else:
                    new_w = target_size
                    new_h = (new_w / raw_w) * raw_h
            else:
                if raw_h > raw_w:
                    new_w = target_size
                    new_h = (new_w / raw_w) * raw_h
                else:
                    new_h = target_size
                    new_w = (new_h / raw_h) * raw_w
            img = img.resize((int(new_w), int(new_h)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()


def _create_palette(num_hues: int, sat_range: int, light_range: int) -> np.ndarray:
    """Verbatim palette construction from tools ColorModelV2.createPalette."""
    height = 1 + sat_range + (2 * light_range - 1)
    hues = np.tile(np.linspace(0, 1, num_hues + 1)[:-1], (height, 1))
    if num_hues == 8:
        hues = np.tile(
            np.array([0.0, 0.10, 0.15, 0.28, 0.51, 0.58, 0.77, 0.85]), (height, 1)
        )
    if num_hues == 9:
        hues = np.tile(
            np.array([0.0, 0.10, 0.15, 0.28, 0.49, 0.54, 0.60, 0.7, 0.87]), (height, 1)
        )
    if num_hues == 10:
        hues = np.tile(
            np.array([0.0, 0.10, 0.15, 0.28, 0.49, 0.54, 0.60, 0.66, 0.76, 0.87]),
            (height, 1),
        )
    elif num_hues == 11:
        hues = np.tile(
            np.array(
                [0.0, 0.0833, 0.166, 0.25, 0.333, 0.5, 0.56333, 0.666, 0.73, 0.803, 0.916]
            ),
            (height, 1),
        )

    sats = np.hstack(
        (
            np.linspace(0, 1, sat_range + 2)[1:-1],
            1,
            [1] * light_range,
            [0.4] * (light_range - 1),
        )
    )
    lights = np.hstack(
        (
            [1] * sat_range,
            1,
            np.linspace(1, 0.2, light_range + 2)[1:-1],
            np.linspace(1, 0.2, light_range + 2)[1:-2],
        )
    )

    sats = np.tile(np.atleast_2d(sats).T, (1, num_hues))
    lights = np.tile(np.atleast_2d(lights).T, (1, num_hues))
    colors = hsv2rgb(np.dstack((hues, sats, lights)))
    grays = np.tile(np.linspace(1, 0, height)[:, np.newaxis, np.newaxis], (1, 1, 3))

    h, w, _ = colors.shape
    color_array = colors.T.reshape((3, w * h)).T
    h, w, _ = grays.shape
    gray_array = grays.T.reshape((3, w * h)).T
    rgb_array = np.vstack((color_array, gray_array))
    return rgb2lab(rgb_array[None, :, :]).squeeze()


class ColorModelV2:
    """81-dim smoothed color histogram (tools mcsearch ColorModelV2)."""

    def __init__(
        self,
        num_hues: int = PALETTE_NUM_HUES,
        sat_range: int = PALETTE_SAT_RANGE,
        light_range: int = PALETTE_LIGHT_RANGE,
        sigma: int = PALETTE_SIGMA,
    ):
        self.num_hues = num_hues
        self.sat_range = sat_range
        self.light_range = light_range
        self.sigma = sigma
        self.palette_lab_array = _create_palette(num_hues, sat_range, light_range)
        self.palette_lab_colors_distances = euclidean_distances(
            self.palette_lab_array, squared=True
        )

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        image_bytes = resize_image_bytes(
            image_bytes, target_size=FEATURE_RESIZE_TARGET, if_smaller_no_resize=True
        )
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_lab = rgb2lab(np.array(img))
        h, w, d = img_lab.shape
        return img_lab.reshape((h * w, d))

    def _histogram_colors_smoothed(self, lab_array: np.ndarray) -> np.ndarray:
        dist = euclidean_distances(self.palette_lab_array, lab_array, squared=True).T
        min_ind = np.argmin(dist, axis=1)
        num_colors = self.palette_lab_array.shape[0]
        num_pixels = lab_array.shape[0]
        color_hist = 1.0 * np.bincount(min_ind, minlength=num_colors) / num_pixels
        n = 2.0 * self.sigma**2
        weights = np.exp(-self.palette_lab_colors_distances / n)
        norm_weights = weights / weights.sum(1)[:, np.newaxis]
        color_hist_smooth = (norm_weights * color_hist).sum(1)
        color_hist_smooth[color_hist_smooth < 1e-5] = 0
        return color_hist_smooth

    def infer(self, image_bytes_list: list[bytes]) -> list[dict | None]:
        """Batch of {features: [float]} per image, None for failed items
        (same contract as tools infer: caller decides per-item status)."""
        tic = time.perf_counter()
        results: list[dict | None] = []
        try:
            img_labs = [self._preprocess(b) for b in image_bytes_list]
            features_array = []
            for img_lab in img_labs:
                features = np.around(self._histogram_colors_smoothed(img_lab), 3)
                features_array.append(features)
            features_array = l2_norm_and_round(feature_array=np.asarray(features_array))
            features_array = features_array.astype("float32")
            for features in features_array:
                results.append({"features": features.tolist()})
        except Exception as e:  # noqa: BLE001 - tools swallows per-batch too
            logger.error("ColorModelV2 batch failed: %s", e)
            results = [None] * len(image_bytes_list)
        toc = time.perf_counter()
        logger.debug(
            "ColorModelV2 batch=%d spent %.1f ms", len(image_bytes_list), 1000 * (toc - tic)
        )
        return results


class ColorPaletteModel:
    """Dominant colors via faiss.Kmeans on RGB (tools ColorPaletteModel).

    Byte-compat notes vs the RUNNING gpu7 tools container (commit 289c2b0,
    not the stale local copy in /data/projects/model-infer-api):
      - niter = 10 (was reduced from 50; 3D pixel clustering converges early)
      - assignment is passed to np.unique AS THE FULL (D, I) TUPLE returned by
        kmeans.assign(), exactly like the production code. np.unique flattens
        it, so distances mix into the value set: some centroid indices get no
        entry in hex_count_map (count falls back to 1) and counts can exceed
        100. This is intentional-to-parity: reproducing the quirk is what
        keeps hexColors byte-identical with production ES colorPalette data.
    """

    def __init__(self, colors_num: int = 6, niter: int = 10):
        self.colors_num = colors_num
        self.niter = niter

    def infer(self, image_bytes: bytes, colors_num: int | None = None) -> dict:
        k = int(colors_num or self.colors_num)
        image_bytes = resize_image_bytes(
            image_bytes, target_size=PALETTE_RESIZE_TARGET, inverse=True
        )
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            pixels = np.array(img)
        size = pixels.shape
        total_num = int(size[0] * size[1])
        pixels = pixels.astype(np.float32).reshape((total_num, 3))

        kmeans = faiss.Kmeans(3, k, niter=self.niter, verbose=False)
        kmeans.train(pixels)
        centroids = kmeans.centroids
        assignment = kmeans.assign(pixels)  # (D, I) tuple, passed whole (see docstring)

        values, counts = np.unique(assignment, return_counts=True)
        # keys stay RAW (mix of centroid indices and distance floats) exactly
        # like tools: hexCountMap[key] lookup below uses the same raw ints, and
        # only some of them hit (others fall back to 1) — int() truncation here
        # would merge distance keys onto index keys and corrupt counts.
        hex_count_map = {
            key: int(value) / total_num * 100 + 1 for key, value in zip(values, counts)
        }

        hex_colors = []
        for index, center in enumerate(centroids):
            center = np.around(center, decimals=2)
            count = hex_count_map.get(index, 1)
            hex_colors.append([rgb2hex1(center.tolist()), int(count)])
        return {"hexColors": hex_colors}


class OpqQuantizer:
    """OPQ rotate + PQ encode, reusing the shipped tools codebooks (never retrained).

    Codebook files: models/color/{opq,pq}_<subspaces>_<bits>.model — copied
    byte-identical from the running tools container cache
    /root/.cache/soujpg/models/ytuHighEnd-color_81-es_opq-10_3/ (see card
    t_62a5c1dd comment: md5 2d515c43... / 3428f147...).
    """

    def __init__(
        self,
        model_dir: str,
        sub_space_num: int = DEFAULT_PQ_SUBSPACES,
        sub_space_bits: int = DEFAULT_PQ_BITS,
    ):
        self.sub_space_num = sub_space_num
        self.sub_space_bits = sub_space_bits
        opq_file = f"{model_dir.rstrip('/')}/opq_{sub_space_num}_{sub_space_bits}.model"
        pq_file = f"{model_dir.rstrip('/')}/pq_{sub_space_num}_{sub_space_bits}.model"
        self.vt_model = faiss.read_VectorTransform(opq_file)
        self.pq_model = faiss.read_ProductQuantizer(pq_file)

    def quantize(self, features: np.ndarray) -> str:
        """(1, 81) float32 -> tools opqCode string 'code_0 code_1 code_2'.

        tools OPQFeaturesQuantizer.quantize builds per-vector strings
        'code_index' joined by spaces (BitstringReader per subspace); the
        string lands verbatim in ES colorCodes, so format is part of the
        byte-compat contract.
        """
        vectors = self.vt_model.apply_py(np.ascontiguousarray(features, dtype="float32"))
        codes = self.pq_model.compute_codes(vectors)
        reader = faiss.BitstringReader(faiss.swig_ptr(codes[0]), codes.shape[1])
        parts = [reader.read(self.sub_space_bits) for _ in range(self.sub_space_num)]
        return " ".join(f"{code}_{i}" for i, code in enumerate(parts))
