"""MLRI Bayer demosaicing implementation.

This module provides `demosaicing`, an OpenCV-style API for Bayer-to-BGR
demosaicing based on the Residual Interpolation / MLRI method from:
"Beyond Color Difference: Residual Interpolation for Color Image
Demosaicking" (IEEE Transactions on Image Processing, 2016).
"""

import cv2
import numpy as np

FLOATING_DTYPE = np.float32
BORDER = cv2.BORDER_REPLICATE
_CODE_TO_PATTERN = {
    cv2.COLOR_BAYER_RGGB2BGR: "RGGB",
    cv2.COLOR_BAYER_GRBG2BGR: "GRBG",
    cv2.COLOR_BAYER_BGGR2BGR: "BGGR",
    cv2.COLOR_BAYER_GBRG2BGR: "GBRG",
}


def demosaicing(
    src: np.ndarray,
    code: int | str = cv2.COLOR_BAYER_RGGB2BGR,
    *,
    sigma: float = 1.0,
    eps: float = 1e-16,
) -> np.ndarray:
    """Demosaic a 2D Bayer CFA image into a 3-channel BGR image.


    Parameters
    ----------
    src : np.ndarray
        Input 2D Bayer CFA image.
    code : int, optional
        OpenCV color conversion code indicating the Bayer pattern (default: `cv2.COLOR_BAYER_RGGB2BGR`).
    sigma : float, optional
        Standard deviation for the Gaussian weighting in directional color difference combination (default: 1.0).
    eps : float, optional
        Small constant to prevent division by zero in guided filtering (default: 1e-16).

    Returns
    -------
    np.ndarray
        Demosaiced 3-channel BGR image.

    References
    ----------
    - D. Kiku, Y. Monno, M. Tanaka, and M. Okutomi, "Beyond color difference: Residual interpolation for color image demosaicking," IEEE Transactions on Image Processing, vol. 25, no. 3, pp. 1288–1300, 2016.
    """
    if src.ndim != 2:
        raise ValueError(f"Input image must be a 2D Bayer CFA image, got shape {src.shape}.")

    # Normalize input to [0, 1] range for processing,
    # and remember the scale to convert back to original range later.
    input_scale = float(np.max(src))
    src_norm = src.astype(FLOATING_DTYPE) / input_scale

    # Create mosaic and mask from the CFA image.
    if isinstance(code, int):
        pattern = _CODE_TO_PATTERN[code]
    elif isinstance(code, str):
        pattern = code.upper()
        if pattern not in _CODE_TO_PATTERN.values():
            raise ValueError(f"Unsupported Bayer pattern {pattern!r}.")
    else:
        raise ValueError(f"Unsupported code type {type(code)}. Must be int or str.")
    mosaic, mask = _mosaic_and_mask_from_cfa(src_norm, pattern)

    # Perform green channel interpolation first,
    # then chromatic interpolation for red and blue channels.
    green = _green_interpolation(mosaic, mask, pattern, sigma, eps)
    red = _chromatic_interpolation(green, mosaic[:, :, 2], mask[:, :, 2], eps)
    blue = _chromatic_interpolation(green, mosaic[:, :, 0], mask[:, :, 0], eps)

    # Stack the channels back into a BGR image
    # and scale back to original range.
    img_bgr = np.stack([blue, green, red], axis=-1) * input_scale

    # Clip to valid range and convert back to original dtype if it's an integer type.
    if np.issubdtype(src.dtype, np.integer):
        vmax = np.iinfo(src.dtype).max
        img_bgr = np.clip(img_bgr, 0, vmax).astype(src.dtype)

    return img_bgr


def _mosaic_and_mask_from_cfa(cfa: np.ndarray, pattern: str) -> tuple[np.ndarray, np.ndarray]:
    channel_map = {"R": 2, "G": 1, "B": 0}
    h, w = cfa.shape
    mask = np.zeros((h, w, 3), dtype=FLOATING_DTYPE)

    for color, (y, x) in zip(pattern, ((0, 0), (0, 1), (1, 0), (1, 1))):
        mask[y::2, x::2, channel_map[color]] = 1.0

    mosaic = cfa[:, :, None].astype(FLOATING_DTYPE) * mask
    return mosaic, mask


def _build_green_split_masks(shape: tuple[int, int], pattern: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    mask_gr = np.zeros((h, w), dtype=FLOATING_DTYPE)
    mask_gb = np.zeros((h, w), dtype=FLOATING_DTYPE)
    pattern = pattern.upper()

    if pattern == "GRBG":
        mask_gr[0::2, 0::2] = 1.0
        mask_gb[1::2, 1::2] = 1.0
    elif pattern == "RGGB":
        mask_gr[0::2, 1::2] = 1.0
        mask_gb[1::2, 0::2] = 1.0
    elif pattern == "GBRG":
        mask_gb[0::2, 0::2] = 1.0
        mask_gr[1::2, 1::2] = 1.0
    elif pattern == "BGGR":
        mask_gb[0::2, 1::2] = 1.0
        mask_gr[1::2, 0::2] = 1.0
    else:
        raise ValueError(f"Unsupported Bayer pattern {pattern!r}.")

    return mask_gr, mask_gb


def _guidedfilter_mlri(
    guide: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
    I: np.ndarray,
    p: np.ndarray,
    M: np.ndarray,
    radius_h: int,
    radius_v: int,
    eps: float,
) -> np.ndarray:
    ksize = (2 * radius_v + 1, 2 * radius_h + 1)
    N = cv2.boxFilter(M, ddepth=-1, ksize=ksize, normalize=False, borderType=BORDER)
    N = np.where(N == 0, 1.0, N)
    mean_Ip = cv2.boxFilter(I * p * M, -1, ksize, normalize=False, borderType=BORDER) / N
    mean_II = cv2.boxFilter(I**2 * M, -1, ksize, normalize=False, borderType=BORDER) / N
    a = mean_Ip / (mean_II + eps)

    N3 = cv2.boxFilter(mask, -1, ksize, normalize=False, borderType=BORDER)
    N3 = np.where(N3 == 0, 1.0, N3)
    mean_G = cv2.boxFilter(guide * mask, -1, ksize, normalize=False, borderType=BORDER) / N3
    mean_R = cv2.boxFilter(ref * mask, -1, ksize, normalize=False, borderType=BORDER) / N3
    b = mean_R - a * mean_G

    gg = cv2.boxFilter(guide**2 * mask, -1, ksize, normalize=False, borderType=BORDER)
    rr = cv2.boxFilter(ref**2 * mask, -1, ksize, normalize=False, borderType=BORDER)
    g = cv2.boxFilter(guide * mask, -1, ksize, normalize=False, borderType=BORDER)
    r = cv2.boxFilter(ref * mask, -1, ksize, normalize=False, borderType=BORDER)
    rg = cv2.boxFilter(ref * guide * mask, -1, ksize, normalize=False, borderType=BORDER)
    dif = gg * a**2 + b**2 * N3 + rr + 2 * a * b * g - 2 * b * r - 2 * a * rg
    dif = dif / N3
    dif = np.maximum(dif, eps)
    dif = 1.0 / dif
    wdif = cv2.boxFilter(dif, -1, ksize, normalize=False, borderType=BORDER)
    wdif = np.maximum(wdif, eps)
    mean_a = cv2.boxFilter(a * dif, -1, ksize, normalize=False, borderType=BORDER) / wdif
    mean_b = cv2.boxFilter(b * dif, -1, ksize, normalize=False, borderType=BORDER) / wdif
    return mean_a * guide + mean_b


def _green_interpolation(
    mosaic: np.ndarray,
    mask: np.ndarray,
    pattern: str,
    sigma: float,
    eps: float,
) -> np.ndarray:
    h, w = mosaic.shape[:2]
    mask_gr, mask_gb = _build_green_split_masks((h, w), pattern)

    rawq = mosaic.sum(axis=2)
    mosaic_r = mosaic[:, :, 2]
    mosaic_g = mosaic[:, :, 1]
    mosaic_b = mosaic[:, :, 0]
    mask_r = mask[:, :, 2]
    mask_g = mask[:, :, 1]
    mask_b = mask[:, :, 0]

    # Guide image
    Kh = np.array([[0.5, 0.0, 0.5]], dtype=FLOATING_DTYPE)
    Kv = Kh.T
    rawh = cv2.filter2D(rawq, -1, Kh, borderType=BORDER)
    rawv = cv2.filter2D(rawq, -1, Kv, borderType=BORDER)
    guide_gh = mosaic_g + rawh * (mask_r + mask_b)
    guide_rh = mosaic_r + rawh * mask_gr
    guide_bh = mosaic_b + rawh * mask_gb
    guide_gv = mosaic_g + rawv * (mask_r + mask_b)
    guide_rv = mosaic_r + rawv * mask_gb
    guide_bv = mosaic_b + rawv * mask_gr

    # Tentative image
    h_rad = 3
    v_rad = 3
    Fh = np.array([[-1, 0, 2, 0, -1]], dtype=FLOATING_DTYPE)
    Fv = Fh.T

    # Horizontal
    dif_r = cv2.filter2D(mosaic_r, -1, Fh, borderType=BORDER)
    dif_gr = cv2.filter2D(guide_gh * mask_r, -1, Fh, borderType=BORDER)
    tentative_rh = _guidedfilter_mlri(guide_gh, mosaic_r, mask_r, dif_gr, dif_r, mask_r, h_rad, v_rad, eps)

    dif_gr = cv2.filter2D(mosaic_g * mask_gr, -1, Fh, borderType=BORDER)
    dif_r = cv2.filter2D(guide_rh * mask_gr, -1, Fh, borderType=BORDER)
    tentative_grh = _guidedfilter_mlri(guide_rh, mosaic_g * mask_gr, mask_gr, dif_r, dif_gr, mask_gr, h_rad, v_rad, eps)

    dif_b = cv2.filter2D(mosaic_b, -1, Fh, borderType=BORDER)
    dif_gb = cv2.filter2D(guide_gh * mask_b, -1, Fh, borderType=BORDER)
    tentative_bh = _guidedfilter_mlri(guide_gh, mosaic_b, mask_b, dif_gb, dif_b, mask_b, h_rad, v_rad, eps)

    dif_gb = cv2.filter2D(mosaic_g * mask_gb, -1, Fh, borderType=BORDER)
    dif_b = cv2.filter2D(guide_bh * mask_gb, -1, Fh, borderType=BORDER)
    tentative_gbh = _guidedfilter_mlri(guide_bh, mosaic_g * mask_gb, mask_gb, dif_b, dif_gb, mask_gb, h_rad, v_rad, eps)

    # Vertical
    dif_r = cv2.filter2D(mosaic_r, -1, Fv, borderType=BORDER)
    dif_gr = cv2.filter2D(guide_gv * mask_r, -1, Fv, borderType=BORDER)
    tentative_rv = _guidedfilter_mlri(guide_gv, mosaic_r, mask_r, dif_gr, dif_r, mask_r, v_rad, h_rad, eps)

    dif_gr = cv2.filter2D(mosaic_g * mask_gb, -1, Fv, borderType=BORDER)
    dif_r = cv2.filter2D(guide_rv * mask_gb, -1, Fv, borderType=BORDER)
    tentative_grv = _guidedfilter_mlri(guide_rv, mosaic_g * mask_gb, mask_gb, dif_r, dif_gr, mask_gb, v_rad, h_rad, eps)

    dif_b = cv2.filter2D(mosaic_b, -1, Fv, borderType=BORDER)
    dif_gb = cv2.filter2D(guide_gv * mask_b, -1, Fv, borderType=BORDER)
    tentative_bv = _guidedfilter_mlri(guide_gv, mosaic_b, mask_b, dif_gb, dif_b, mask_b, v_rad, h_rad, eps)

    dif_gb = cv2.filter2D(mosaic_g * mask_gr, -1, Fv, borderType=BORDER)
    dif_b = cv2.filter2D(guide_bv * mask_gr, -1, Fv, borderType=BORDER)
    tentative_gbv = _guidedfilter_mlri(guide_bv, mosaic_g * mask_gr, mask_gr, dif_b, dif_gb, mask_gr, v_rad, h_rad, eps)

    # Residual
    Kh_lin = np.array([[0.5, 0.0, 0.5]], dtype=FLOATING_DTYPE)
    Kv_lin = Kh_lin.T
    residual_grh = cv2.filter2D((mosaic_g - tentative_grh) * mask_gr, -1, Kh_lin, borderType=BORDER)
    residual_gbh = cv2.filter2D((mosaic_g - tentative_gbh) * mask_gb, -1, Kh_lin, borderType=BORDER)
    residual_rh = cv2.filter2D((mosaic_r - tentative_rh) * mask_r, -1, Kh_lin, borderType=BORDER)
    residual_bh = cv2.filter2D((mosaic_b - tentative_bh) * mask_b, -1, Kh_lin, borderType=BORDER)
    residual_grv = cv2.filter2D((mosaic_g - tentative_grv) * mask_gb, -1, Kv_lin, borderType=BORDER)
    residual_gbv = cv2.filter2D((mosaic_g - tentative_gbv) * mask_gr, -1, Kv_lin, borderType=BORDER)
    residual_rv = cv2.filter2D((mosaic_r - tentative_rv) * mask_r, -1, Kv_lin, borderType=BORDER)
    residual_bv = cv2.filter2D((mosaic_b - tentative_bv) * mask_b, -1, Kv_lin, borderType=BORDER)

    # Add tentative and residual
    grh = (tentative_grh + residual_grh) * mask_r
    gbh = (tentative_gbh + residual_gbh) * mask_b
    rh = (tentative_rh + residual_rh) * mask_gr
    bh = (tentative_bh + residual_bh) * mask_gb
    grv = (tentative_grv + residual_grv) * mask_r
    gbv = (tentative_gbv + residual_gbv) * mask_b
    rv = (tentative_rv + residual_rv) * mask_gb
    bv = (tentative_bv + residual_bv) * mask_gr

    # Vertical-Horizontal color difference
    mosaic_g_r_b = mosaic_g - mosaic_r - mosaic_b
    difh = grh + gbh - rh - bh + mosaic_g_r_b
    difv = grv + gbv - rv - bv + mosaic_g_r_b

    # Combine vertical and horizontal color differences
    # color difference gradient
    grad_kh = np.array([[1, 0, -1]], dtype=FLOATING_DTYPE)
    grad_kv = grad_kh.T
    difh2 = np.abs(cv2.filter2D(difh, -1, grad_kh, borderType=BORDER))
    difv2 = np.abs(cv2.filter2D(difv, -1, grad_kv, borderType=BORDER))

    # directional weight
    K = np.ones((3, 3), dtype=FLOATING_DTYPE)
    wh = cv2.filter2D(difh2, -1, K, borderType=BORDER)
    wv = cv2.filter2D(difv2, -1, K, borderType=BORDER)
    Kw = np.array([[1, 0, 0]], dtype=FLOATING_DTYPE)
    Ke = np.array([[0, 0, 1]], dtype=FLOATING_DTYPE)
    Ks = Ke.T
    Kn = Kw.T
    Ww = cv2.filter2D(wh, -1, Kw, borderType=BORDER)
    We = cv2.filter2D(wh, -1, Ke, borderType=BORDER)
    Wn = cv2.filter2D(wv, -1, Kn, borderType=BORDER)
    Ws = cv2.filter2D(wv, -1, Ks, borderType=BORDER)
    Ww = 1.0 / (Ww**2 + eps)
    We = 1.0 / (We**2 + eps)
    Ws = 1.0 / (Ws**2 + eps)
    Wn = 1.0 / (Wn**2 + eps)

    # Combine directional color difference
    gauss = cv2.getGaussianKernel(9, sigma).T.astype(FLOATING_DTYPE)
    ke_vec = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=FLOATING_DTYPE) * gauss
    kw_vec = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=FLOATING_DTYPE) * gauss
    ke_vec /= np.sum(ke_vec)
    kw_vec /= np.sum(kw_vec)
    Ke_full = ke_vec.reshape(1, -1)
    Kw_full = kw_vec.reshape(1, -1)
    Ks_full = Ke_full.T
    Kn_full = Kw_full.T
    difn = cv2.filter2D(difv, -1, Kn_full, borderType=BORDER)
    difs = cv2.filter2D(difv, -1, Ks_full, borderType=BORDER)
    difw = cv2.filter2D(difh, -1, Kw_full, borderType=BORDER)
    dife = cv2.filter2D(difh, -1, Ke_full, borderType=BORDER)
    Wt = Ww + We + Wn + Ws
    dif = (Wn * difn + Ws * difs + Ww * difw + We * dife) / Wt
    # green = (dif + rawq) * (1.0 - mask_g) + rawq * mask_g
    green = dif * (1.0 - mask_g) + rawq
    return green


def _chromatic_interpolation(
    green: np.ndarray,
    chroma_sample: np.ndarray,
    chroma_mask: np.ndarray,
    eps: float,
) -> np.ndarray:
    h_rad = 5
    v_rad = 5
    lap_kernel = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [-1, 0, 4, 0, -1],
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=FLOATING_DTYPE,
    )
    lap_chroma = cv2.filter2D(chroma_sample, -1, lap_kernel, borderType=BORDER)
    lap_green = cv2.filter2D(green * chroma_mask, -1, lap_kernel, borderType=BORDER)
    tentative = _guidedfilter_mlri(green, chroma_sample, chroma_mask, lap_green, lap_chroma, chroma_mask, h_rad, v_rad, eps)
    residual = chroma_mask * (chroma_sample - tentative)
    kernel = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], dtype=FLOATING_DTYPE)
    residual = cv2.filter2D(residual, -1, kernel, borderType=BORDER)
    return residual + tentative
