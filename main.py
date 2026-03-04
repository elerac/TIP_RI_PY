import cv2
import numpy as np

from tip_ri import demosaicing


def create_bayer_cfa(img_bgr: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Create a Bayer CFA mosaic from a BGR image (simulates raw sensor data)."""
    h, w = img_bgr.shape[:2]
    dtype = img_bgr.dtype

    # Map color channels: B=0, G=1, R=2
    c2i = {"B": 0, "G": 1, "R": 2}

    # Create CFA mosaic
    img_cfa = np.zeros((h, w), dtype=dtype)
    for c, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        img_cfa[y::2, x::2] = img_bgr[y::2, x::2, c2i[c]]

    return img_cfa


def main():
    # Load an image
    img_bgr = cv2.imread("tshirts.jpg", cv2.IMREAD_COLOR)
    if img_bgr is None:
        print("Error: Could not load image. Please ensure 'tshirts.jpg' exists.")
        return

    print(f"Input image shape: {img_bgr.shape}")

    # Create Bayer CFA (simulating raw sensor data)
    img_cfa = create_bayer_cfa(img_bgr, pattern="RGGB")
    print(f"CFA shape: {img_cfa.shape}")

    # Demosaic using TIP_RI method
    img_demosaiced = demosaicing(img_cfa, cv2.COLOR_BAYER_RGGB2BGR)
    print(f"Demosaiced image shape: {img_demosaiced.shape}")

    # Calculate PSNR against original
    psnr = cv2.PSNR(img_bgr, img_demosaiced)
    print(f"PSNR: {psnr:.2f} dB")

    # Save results
    cv2.imwrite("output_cfa.png", img_cfa)
    cv2.imwrite("output_demosaiced.png", img_demosaiced)
    print("\nResults saved:")
    print("  - output_cfa.png (Bayer CFA)")
    print("  - output_demosaiced.png (Demosaiced image)")


if __name__ == "__main__":
    main()
