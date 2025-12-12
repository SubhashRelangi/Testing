import cv2
import numpy as np
import time 
from typing import Optional


def thermal_nlm_denoise(
    src: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 15,
    preserve_depth: bool = True
) -> np.ndarray:
    """
    Apply Non-Local Means (NLM) denoising on a THERMAL single-channel image.

    Parameters:
        src (ndarray): Thermal image (must be single-channel). *
        h (float): Filter strength (higher = more smoothing). Default=10.0
        template_window (int): Size in pixels of the template patch (must be odd). Default=7
        search_window (int): Size in pixels of the window used to compute weighted average (must be odd). Default=15
        preserve_depth (bool): If True, attempt to return image in the same dtype/range as input.
                               For >8-bit input, the algorithm internally normalizes to 8-bit, denoises,
                               then remaps back to original range if possible. Default=True

    Returns:
        ndarray: Denoised thermal image (same shape as input).

    Raises:
        ValueError: If invalid arguments or multi-channel image provided.
    """

    start_time = time.time()

    # --- Input checks ---
    if src is None:
        raise ValueError("Source image is NONE")

    if not isinstance(src, np.ndarray):
        raise ValueError("Input must be a numpy ndarray")

    # Strict thermal rule: single-channel only
    if src.ndim == 2:
        img = src.copy()
    elif src.ndim == 3 and src.shape[2] == 1:
        img = src[..., 0].copy()
    else:
        raise ValueError(f"NLM requires SINGLE-CHANNEL thermal image. Got shape={src.shape}")

    # Parameter validation
    if not (isinstance(template_window, int) and template_window > 0 and template_window % 2 == 1):
        raise ValueError("template_window must be a positive odd integer")

    if not (isinstance(search_window, int) and search_window > 0 and search_window % 2 == 1):
        raise ValueError("search_window must be a positive odd integer")

    if h <= 0:
        raise ValueError("h (filter strength) must be > 0")

    # --- Handle bit depth: OpenCV's fastNlMeansDenoising expects 8-bit single-channel input ---
    original_dtype = img.dtype
    original_min = None
    original_max = None

    # If input is floating, convert to a reasonable integer scale first
    if np.issubdtype(original_dtype, np.floating):
        # assume range [0,1] or arbitrary; normalize to 0-255
        original_min = float(np.min(img))
        original_max = float(np.max(img))
        if original_max == original_min:
            # flat image, nothing to denoise
            return img.copy()
        img_norm = ((img - original_min) / (original_max - original_min) * 255.0).astype(np.uint8)
    elif original_dtype == np.uint8:
        img_norm = img.copy()
    elif np.issubdtype(original_dtype, np.integer):
        # e.g., uint16 or uint12 packed into uint16 - normalize min..max -> 0..255
        original_min = int(np.min(img))
        original_max = int(np.max(img))
        if original_max == original_min:
            return img.copy()
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image dtype: {original_dtype}")

    # --- Apply OpenCV Non-Local Means (single-channel) ---
    denoised8 = cv2.fastNlMeansDenoising(
        img_norm,
        None,
        h=float(h),
        templateWindowSize=int(template_window),
        searchWindowSize=int(search_window)
    )

    

    end_time = time.time()
    print(f"{'gaussian_filter execution time:':<36}{end_time - start_time:.4f} seconds")

    # always return 8-bit denoised image
    return denoised8
    


if __name__ == "__main__":
    # Example usage similar to your bilateral example.
    input_path = "/home/user1/learning/Testing/NoiseReduction/Inputs/SingleChannel.png"
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Tune these for your sensor; thermal imagery is smoother than visible-light imagery.
    denoised = thermal_nlm_denoise(
        img,
        h=8.0,
        template_window=7,
        search_window=15,
        preserve_depth=True
    )

    cv2.imshow("Thermal NLM Denoise", denoised)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
