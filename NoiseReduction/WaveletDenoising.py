import cv2
import numpy as np
import time
from skimage.restoration import denoise_wavelet

def wavelet_denoise_thermal(
    image_path: str,
    *,
    method: str = "BayesShrink",
    mode: str = "soft",
    wavelet: str = "db4",
    wavelet_levels: int = 1,
    rescale_sigma: bool = True,
    preserve_dtype: bool = True,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> np.ndarray | None:
    """
    Wavelet-based denoising for thermal images (8-bit or 16-bit).

    RETURNS:
        denoised_image (np.ndarray) or None if error.

    PARAMETERS (each entry: Description / Min & Max / Units / Default / Best-case)
    ------------------------------------------------------------------------------
    image_path : str
        Description: Filesystem path to the input thermal image. Image is read in
                     grayscale mode and must be a single-channel image.
        Min & Max: Must be an existing, readable file path. (Min: non-empty string,
                   Max: path length limited by OS.)
        Units: filesystem path (string)
        Default: required (no default)
        Best case: Local SSD path to a valid single-channel 8-bit or 16-bit image.

    method : str
        Description: Threshold estimation method forwarded to skimage.restoration.denoise_wavelet.
                     Common choices: "BayesShrink", "VisuShrink".
        Min & Max: Must be a valid method name supported by denoise_wavelet.
        Units: string
        Default: "BayesShrink"
        Best case: "BayesShrink" (adaptive, generally good for thermal noise)

    mode : str
        Description: Thresholding mode applied to wavelet coefficients ("soft" or "hard").
        Min & Max: "soft" or "hard"
        Units: string
        Default: "soft"
        Best case: "soft" (produces smoother, visually pleasing results)

    wavelet : str
        Description: Wavelet family name used for transform (e.g., "db4", "sym8").
        Min & Max: Valid PyWavelets wavelet name strings.
        Units: string
        Default: "db4"
        Best case: "db4" or "symX" families for thermal images (good balance of localization/smoothness)

    wavelet_levels : int
        Description: Number of decomposition levels (depth of the wavelet transform).
        Min & Max: 1 .. floor(log2(min(H, W))) (practical upper bound depends on image size)
        Units: integer (levels)
        Default: 1
        Best case: 1–3 for typical thermal frames; avoid too deep when image is small.

    rescale_sigma : bool
        Description: If True, let denoise_wavelet rescale sigma (noise estimate) automatically.
        Min & Max: Boolean
        Units: flag
        Default: True
        Best case: True (robust automatic estimation for varied noise levels)

    preserve_dtype : bool
        Description: If True, the returned image is scaled back and cast to the original
                     integer dtype (uint8 or uint16). If False, the function returns a
                     float32 image in normalized [0..1] range.
        Min & Max: Boolean
        Units: flag
        Default: True
        Best case: True to preserve original bit-depth for downstream processing/storage.

    clip_min : float | None
        Description: Optional lower clipping bound applied after conversion back to dtype.
                     If None, no explicit clipping beyond dtype conversion is performed.
        Min & Max: None or numeric <= clip_max
        Units: pixel intensity (same units as output dtype)
        Default: None
        Best case: None (leave natural result) or 0 for strict non-negative output.

    clip_max : float | None
        Description: Optional upper clipping bound applied after conversion back to dtype.
                     If None, no explicit clipping beyond dtype conversion is performed.
        Min & Max: None or numeric >= clip_min
        Units: pixel intensity (same units as output dtype)
        Default: None
        Best case: None (leave natural result) or 255/65535 for uint8/uint16 workflows.

    NOTES:
    - The function reads the image using cv2.IMREAD_GRAYSCALE so multi-channel inputs will be
      converted by OpenCV to single-channel; the function returns None on load failure or if
      the image cannot be processed.
    - wavelet_levels is automatically clipped between 1 and floor(log2(min(H,W))) to avoid
      invalid decomposition depths.
    - If preserve_dtype is True, the floating result (in [0..1]) is scaled back to the
      appropriate integer range (0..255 or 0..65535) and cast; small numerical over/undershoots
      are clamped by .clip(...) when casting in the existing code.
    - The function returns None on any internal error (loading, parameter issues, or denoising failure).

    Example:
        denoised = wavelet_denoise_thermal("/path/to/frame.png", wavelet_levels=2, preserve_dtype=True)
    """

    start_time = time.perf_counter()

    # Load image
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None

    if img.ndim == 3:  # BGR image
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

    # If still not 2D, it's invalid
    if img.ndim != 2:
        return None

    # Determine normalization scale
    try:
        if img.dtype == np.uint16 or img.max() > 255:
            img_norm = img.astype(np.float32) / 65535.0
            out_dtype = np.uint16
        else:
            img_norm = img.astype(np.float32) / 255.0
            out_dtype = np.uint8
    except Exception:
        return None

    # Validate wavelet levels
    H, W = img.shape
    max_levels = int(np.floor(np.log2(min(H, W)))) if min(H, W) > 1 else 1
    wavelet_levels = max(1, min(wavelet_levels, max_levels))

    # Apply wavelet denoising
    try:
        den = denoise_wavelet(
            img_norm,
            method=method,
            mode=mode,
            wavelet=wavelet,
            wavelet_levels=wavelet_levels,
            rescale_sigma=rescale_sigma,
            channel_axis=None
        )
    except Exception:
        return None

    # Restore original dtype
    try:
        if preserve_dtype:
            if out_dtype == np.uint16:
                out_image = (den * 65535.0).clip(0, 65535).astype(np.uint16)
            else:
                out_image = (den * 255.0).clip(0, 255).astype(np.uint8)
        else:
            out_image = den.astype(np.float32)
    except Exception:
        return None
    
    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(f"[Wavelet Denoise] Time Taken: {total_time:.6f} seconds")

    return out_image

if __name__ == "__main__":

    denoise = wavelet_denoise_thermal("/home/user1/learning/Testing/NoiseReduction/Inputs/frame_000000.jpg", method="BayesShrink", mode="hard", wavelet="db6", wavelet_levels=4)

    cv2.imshow("Denoised Image", denoise)
    cv2.waitKey(0)
    cv2.destroyAllWindows()