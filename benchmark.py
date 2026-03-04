import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_Menon2007

from tip_ri import demosaicing

RESULTS_DIR = Path("results")


def mosaicing_cfa_bayer(img_bgr: np.ndarray, pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"] | str = "RGGB") -> tuple[np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    dtype = img_bgr.dtype

    # Construct masks for each color channel.
    c2i = {c: i for i, c in enumerate("BGR")}
    img_bgr_mask = np.zeros((h, w, 3), dtype=bool)
    for c, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        img_bgr_mask[y::2, x::2, c2i[c]] = True

    # Create the Bayer CFA mosaic image (pseudo raw image).
    img_cfa = np.zeros((h, w), dtype=dtype)
    for c in range(3):
        img_cfa[img_bgr_mask[..., c]] = img_bgr[..., c][img_bgr_mask[..., c]]

    return img_cfa, img_bgr_mask


def _crop_and_zoom(img_bgr: np.ndarray, x0: int, y0: int, w: int, h: int, scale: int = 4) -> np.ndarray:
    crop = img_bgr[y0 : y0 + h, x0 : x0 + w]
    return cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def _to_bgr_uint(img_rgb_or_bgr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        vmax = np.iinfo(dtype).max
        return np.clip(np.round(img_rgb_or_bgr * vmax), 0, vmax).astype(dtype)
    return img_rgb_or_bgr.astype(dtype)


def _cfa_to_bgr_colored(img_cfa: np.ndarray, img_bgr_mask: np.ndarray) -> np.ndarray:
    img_cfa_bgr = np.zeros((*img_cfa.shape, 3), dtype=img_cfa.dtype)
    for channel in range(3):
        img_cfa_bgr[..., channel][img_bgr_mask[..., channel]] = img_cfa[img_bgr_mask[..., channel]]
    return img_cfa_bgr


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    filename = "tshirts.jpg"
    pattern = "RGGB"
    img_bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError("Image file not found. Please check the path and filename.")

    stem = Path(filename).stem
    cv2.imwrite(RESULTS_DIR / f"{stem}_input.png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    img_cfa, img_bgr_mask = mosaicing_cfa_bayer(img_bgr, pattern=pattern)
    cv2.imwrite(RESULTS_DIR / f"{stem}_cfa.png", img_cfa, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    img_cfa_bgr = _cfa_to_bgr_colored(img_cfa, img_bgr_mask)
    cv2.imwrite(RESULTS_DIR / f"{stem}_cfa_rgb.png", img_cfa_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    h_img, w_img = img_bgr.shape[:2]
    crop_w = crop_h = 100
    crop_specs = {
        "crop1": (530, 318),
        "crop2": (382, 555),
    }
    crop_specs = {crop_name: (min(max(0, x0), w_img - crop_w), min(max(0, y0), h_img - crop_h)) for crop_name, (x0, y0) in crop_specs.items()}

    for crop_name, (x0, y0) in crop_specs.items():
        input_crop = _crop_and_zoom(img_bgr, x0, y0, crop_w, crop_h, scale=4)
        cv2.imwrite(RESULTS_DIR / f"{stem}_input_{crop_name}.png", input_crop)
        cfa_crop = _crop_and_zoom(img_cfa_bgr, x0, y0, crop_w, crop_h, scale=4)
        cv2.imwrite(RESULTS_DIR / f"{stem}_cfa_rgb_{crop_name}.png", cfa_crop)

    code = cv2.COLOR_BAYER_RGGB2BGR
    code_ea = cv2.COLOR_BAYER_RGGB2BGR_EA

    cfa_float = img_cfa.astype(np.float32)
    if np.issubdtype(img_cfa.dtype, np.integer):
        cfa_float /= float(np.iinfo(img_cfa.dtype).max)

    methods = {
        "tip_ri_mlri": lambda cfa: demosaicing(cfa, code),
        "opencv_bilinear": lambda cfa: cv2.demosaicing(cfa, code),
        "opencv_ea": lambda cfa: cv2.demosaicing(cfa, code_ea),
        "colour_malvar2004": lambda _cfa: cv2.cvtColor(
            _to_bgr_uint(demosaicing_CFA_Bayer_Malvar2004(cfa_float, pattern=pattern), img_bgr.dtype),
            cv2.COLOR_RGB2BGR,
        ),
        "colour_menon2007": lambda _cfa: cv2.cvtColor(
            _to_bgr_uint(demosaicing_CFA_Bayer_Menon2007(cfa_float, pattern=pattern), img_bgr.dtype),
            cv2.COLOR_RGB2BGR,
        ),
    }

    results: list[dict[str, float | str]] = []

    # Number of runs for timing measurement
    num_runs = 20

    for method_name, fn in methods.items():
        print(f"Processing {method_name}...")

        # Run multiple times to collect timing data
        times = []
        for run in range(num_runs):
            time_start = time.time()
            img_bgr_demosaiced = fn(img_cfa)
            elapsed = time.time() - time_start
            times.append(elapsed)
            print(f"  Run {run + 1}/{num_runs}: {elapsed:.4f} s")

        mean_time = np.mean(times)
        std_time = np.std(times)

        filename_full = RESULTS_DIR / f"{stem}_demosaiced_{method_name}.png"
        cv2.imwrite(filename_full, img_bgr_demosaiced, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        for crop_name, (x0, y0) in crop_specs.items():
            crop_zoom = _crop_and_zoom(img_bgr_demosaiced, x0, y0, crop_w, crop_h, scale=4)
            filename_crop = RESULTS_DIR / f"{stem}_demosaiced_{method_name}_{crop_name}.png"
            cv2.imwrite(filename_crop, crop_zoom, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            if crop_name == "crop1":
                cv2.imwrite(RESULTS_DIR / f"{stem}_demosaiced_{method_name}_crop.png", crop_zoom, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        psnr = cv2.PSNR(img_bgr, img_bgr_demosaiced)
        results.append(
            {
                "method": method_name,
                "psnr": psnr,
                "time_mean": mean_time,
                "time_std": std_time,
                "full": filename_full,
            }
        )

        print(f"{method_name:20s} | PSNR: {psnr:8.4f} dB | Time: {mean_time:7.4f}±{std_time:6.4f} s")

    print("\nSummary:")
    for result in sorted(results, key=lambda x: float(x["psnr"]), reverse=True):
        print(f"- {result['method']:20s}: " f"PSNR={float(result['psnr']):8.4f} dB, " f"Time={float(result['time_mean']):7.4f}±{float(result['time_std']):6.4f} s")

    return results


if __name__ == "__main__":
    results = main()
