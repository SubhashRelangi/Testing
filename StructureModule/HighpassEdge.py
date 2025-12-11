from typing import Any, Dict, Optional, Tuple
import cv2 as cv
import numpy as np


# =====================================================
# 1. FREQUENCY MASK GENERATOR (MODULAR & FLEXIBLE)
# =====================================================
def create_frequency_mask(
    D: np.ndarray,
    mode: str = "highpass",
    **params: Any
) -> np.ndarray:
    """
    Create a frequency-domain mask.

    Args:
        D (np.ndarray): distance matrix from FFT center (H×W).
        mode (str): "lowpass","highpass","bandpass",
                    "gaussian_lp","gaussian_hp",
                    "butterworth_lp","butterworth_hp".
        params: mode-specific parameters:
            - radius (int)
            - inner (int), outer (int)
            - sigma (float)
            - order (int)

    Returns:
        mask (np.ndarray[float32]): mask of shape (H, W)
    """
    h, w = D.shape
    mask = np.zeros((h, w), np.float32)

    if mode == "lowpass":
        R: int = int(params.get("radius", 50))
        mask[D <= R] = 1.0

    elif mode == "highpass":
        R: int = int(params.get("radius", 50))
        mask[D > R] = 1.0

    elif mode == "bandpass":
        r1: int = int(params.get("inner", 20))
        r2: int = int(params.get("outer", 60))
        mask[(D >= r1) & (D <= r2)] = 1.0

    elif mode == "gaussian_lp":
        sigma: float = float(params.get("sigma", 20.0))
        mask = np.exp(-(D**2) / (2.0 * sigma**2))

    elif mode == "gaussian_hp":
        sigma: float = float(params.get("sigma", 20.0))
        mask = 1.0 - np.exp(-(D**2) / (2.0 * sigma**2))

    elif mode == "butterworth_lp":
        R: float = float(params.get("radius", 50.0))
        order: int = int(params.get("order", 2))
        mask = 1.0 / (1.0 + (D / R) ** (2.0 * order))

    elif mode == "butterworth_hp":
        R: float = float(params.get("radius", 50.0))
        order: int = int(params.get("order", 2))
        mask = 1.0 - (1.0 / (1.0 + (D / R) ** (2.0 * order)))

    else:
        raise ValueError(f"Unknown mask mode: {mode}")

    return mask.astype(np.float32)


# =====================================================
# 2. ENHANCEMENT FUNCTION (COMPLETE IMPROVED VERSION)
# =====================================================
def HighPassEdge(
    image: np.ndarray,
    *,
    filter_mode: str = "highpass",
    radius: int = 50,
    gain: float = 1.5,
    sigma: float = 20.0,
    order: int = 2,
    fft_flag: int = cv.DFT_COMPLEX_OUTPUT,
    norm_dst: Optional[np.ndarray] = None,
    norm_alpha: float = 0.0,
    norm_beta: float = 255.0,
    norm_type: int = cv.NORM_MINMAX,
) -> np.ndarray:
    """
    Fourier-based high-frequency enhancement.

    Args (types & meaning):
      image (np.ndarray): input single-channel image (uint8/uint16/float32/float64).
      filter_mode (str): mask type (see create_frequency_mask).
      radius (int): cutoff radius in pixels (for LP/HP).
      gain (float): HF amplification multiplier.
      sigma (float): Gaussian sigma (for gaussian masks).
      order (int): Butterworth order (for butterworth masks).
      fft_flag (int): cv.dft flags (e.g., cv.DFT_COMPLEX_OUTPUT).
      norm_dst (Optional[np.ndarray]): destination array for cv.normalize or None.
      norm_alpha (float): lower bound for normalization.
      norm_beta (float): upper bound for normalization.
      norm_type (int): cv.normalize type (cv.NORM_MINMAX etc.).

    Returns:
      final_img (np.ndarray): uint8 enhanced image (H×W).
    """
    # validation
    if image is None:
        raise ValueError("Input image is None.")
    if image.ndim != 2:
        raise ValueError("Input must be single-channel (grayscale).")
    if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ValueError("Unsupported image dtype. Use uint8/uint16/float32/float64.")
    if gain < 1.0:
        raise ValueError("gain must be >= 1.0")
    if radius <= 0:
        raise ValueError("radius must be > 0")

    try:
        # convert to float32 for FFT
        float_image: np.ndarray = np.float32(image)

        # DFT
        F: np.ndarray = cv.dft(float_image, flags=fft_flag)
        F_shift: np.ndarray = np.fft.fftshift(F)

        # distance matrix
        h, w = float_image.shape
        cx, cy = w // 2, h // 2
        u = np.arange(w)
        v = np.arange(h)
        U, V = np.meshgrid(u, v)
        D = np.sqrt((U - cx) ** 2 + (V - cy) ** 2)

        # mask
        mask = create_frequency_mask(
            D,
            mode=filter_mode,
            radius=radius,
            sigma=sigma,
            order=order,
            inner=max(1, radius // 2),
            outer=radius,
        )
        mask2 = cv.merge([mask, mask])

        # apply and enhance
        HF = F_shift * mask2
        HF_enhanced = HF * float(gain)
        LF = F_shift * (1.0 - mask2)
        merged = LF + HF_enhanced

        # inverse
        merged_shift = np.fft.fftshift(merged)
        complex_image = cv.idft(merged_shift)
        magnitude_image = cv.magnitude(complex_image[:, :, 0], complex_image[:, :, 1])

        # normalize
        final_img = cv.normalize(magnitude_image, norm_dst, norm_alpha, norm_beta, norm_type)
        return final_img.astype(np.uint8)

    except Exception as exc:
        raise RuntimeError(f"HighPassEdge failed: {exc}")


# =====================================================
# 3. EXECUTION EXAMPLE
# =====================================================
if __name__ == "__main__":
    src_path = "StructureModule/Inputs/Input.jpg"
    image = cv.imread(src_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read {src_path}")

    enhanced = HighPassEdge(
        image,
        filter_mode="gaussian_hp",   # str
        radius=40,                   # int
        gain=2.0,                    # float
        sigma=25.0,                  # float
        order=3,                     # int
        fft_flag=cv.DFT_COMPLEX_OUTPUT,  # int (OpenCV constant)
        norm_alpha=0.0,              # float
        norm_beta=255.0,             # float
        norm_type=cv.NORM_MINMAX     # int (OpenCV constant)
    )

    cv.imshow("Original", image)
    cv.imshow("Enhanced", enhanced)
    result = np.hstack([image, enhanced])
    cv.imwrite("/home/user1/learning/Testing/StructureModule/Outputs/HighPassEdgeEnhancement.png", result)
    cv.waitKey(0)
    cv.destroyAllWindows()
