import cv2 as cv
import numpy as np
import time
from typing import Tuple, Union, Optional
from skimage.metrics import structural_similarity as ssim


ImageArray = np.ndarray


def sharpness_score(image: np.ndarray) -> float:
    """
    Compute sharpness score using Variance of Laplacian.

    Higher value => sharper image
    """
    if image.ndim != 2:
        raise ValueError("sharpness_score expects a single-channel image")

    lap = cv.Laplacian(image, cv.CV_64F)
    return float(lap.var())


def ssim_score(
    reference: np.ndarray,
    processed: np.ndarray
) -> float:
    """
    Compute SSIM between input and output images.

    Range: 0.0 – 1.0 (higher is better)
    """
    if reference.shape != processed.shape:
        raise ValueError("Images must have the same shape for SSIM")

    return float(
        ssim(
            reference,
            processed,
            data_range=processed.max() - processed.min()
        )
    )




def compare_images(
    input_image: np.ndarray,
    output_image: np.ndarray
) -> dict:
    """
    Compare input and output images and return quality scores.
    """

    sharp_in = sharpness_score(input_image)
    sharp_out = sharpness_score(output_image)

    sharpness_gain = sharp_out / sharp_in if sharp_in != 0 else 0.0
    structure_similarity = ssim_score(input_image, output_image)

    return {
        "sharpness_input": sharp_in,
        "sharpness_output": sharp_out,
        "sharpness_gain": sharpness_gain,
        "ssim": structure_similarity,
    }




def _ensure_odd_positive_int_pair(ksize: Tuple[int, int]) -> Tuple[int, int]:
    if not (isinstance(ksize, tuple) and len(ksize) == 2):
        raise TypeError("blur_ksize must be a tuple of two ints (odd, positive).")
    kx, ky = int(ksize[0]), int(ksize[1])
    if kx <= 0 or ky <= 0:
        raise ValueError("blur_ksize elements must be positive.")

    # Force odd values
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1

    return (kx, ky)


def apply_unsharp_masking(
    image: Union[str, ImageArray],
    *,
    blur_ksize: Tuple[int, int] = (5, 5),
    blur_sigma_x: float = 0.0,
    blur_sigma_y: float = 0.0,
    mask_scale: float = 1.0,
    sharp_alpha: float = 3.0,
    out_min: float = 0.0,
    out_max: float = 255.0,
    output_dtype: Optional[Union[str, np.dtype]] = None,
) -> ImageArray:
    """
    Apply Unsharp Masking (USM) to enhance image sharpness.

    OVERVIEW
    --------
    The unsharp masking logic performs:
        1) Gaussian blur (low-pass)
        2) Detail extraction: Mask = Original - Blurred
        3) Sharpening: Sharpened = Original + (Mask * sharp_alpha)
        4) Mask visualization: normalized for display
        5) Output stacking: [Original | Mask | Sharpened]

    PARAMETERS
    ----------
    image : np.ndarray (H×W, uint8 recommended)
        Input grayscale image.

    blur_ksize : tuple(int, int)
        Size of Gaussian kernel.
        Min/Max: odd, positive integers
        Default: (5, 5)

    blur_sigma_x : float
        Gaussian sigma in X direction.
        Default: 0.0 → auto-calculated by OpenCV.

    blur_sigma_y : float
        Gaussian sigma in Y direction.
        Default: 0.0 → automatically same as X if zero.

    mask_scale : float
        Multiplier applied to the high-frequency mask before sharpening.
        Default: 1.0

    sharp_alpha : float
        Sharpening strength applied to the scaled mask.
        Typical range: 0–5  
        Default: 3.0

    mask_add_bias : float
        Bias added to mask for visualization.
        Moves zero-detail level to mid-gray.
        Default: 128.0

    mask_clip_min, mask_clip_max : float
        Clip range for mask visualization.
        Default: 0 → 255

    out_min, out_max : float
        Output clipping range.
        Default: 0 → 255

    RETURNS
    -------
     out_image : np.ndarray (H × (W*3) × 1, uint8)
    

    NOTES
    -----
    - All parameters fully control internal computations.
    - No hidden constants remain.
    """

    start_time = time.perf_counter()

    # -------------------------
    # Load / convert input
    # -------------------------
    if image is None:
        raise ValueError("Input image is None.")

    if isinstance(image, str):
        img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image at path: {image}")
    else:
        img = np.asarray(image)

    if img.size == 0:
        raise ValueError("Input image is empty.")

    # Use only first channel if multi-channel
    if img.ndim > 2:
        img = img[..., 0]

    if img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got {img.shape}")

    orig_dtype = img.dtype

    # -------------------------
    # Convert to float32
    # -------------------------
    if np.issubdtype(orig_dtype, np.floating):
        img_f = np.clip(img, 0.0, 1.0).astype(np.float32) * 255.0

    elif orig_dtype == np.uint8:
        img_f = img.astype(np.float32)

    elif orig_dtype == np.uint16:
        img_f = (img.astype(np.float32) / 65535.0) * 255.0

    else:
        raise TypeError("Unsupported dtype. Allowed: uint8, uint16, float32, float64.")

    # -------------------------
    # Gaussian Blur
    # -------------------------
    blur_ksize = _ensure_odd_positive_int_pair(blur_ksize)

    blurred_f = cv.GaussianBlur(
        img_f,
        ksize=blur_ksize,
        sigmaX=float(blur_sigma_x),
        sigmaY=float(blur_sigma_y),
    )

    # -------------------------
    # High-pass mask & Sharpening
    # -------------------------
    mask_f = (img_f - blurred_f) * float(mask_scale)

    sharpened_f = img_f + (mask_f * float(sharp_alpha))

    # -------------------------
    # Clip output
    # -------------------------
    sharpened_clipped = np.clip(sharpened_f, out_min, out_max)

    # -------------------------
    # Convert to output dtype
    # -------------------------
    if output_dtype is None:
        out_dtype = np.uint8
    else:
        out_dtype = np.dtype(output_dtype)

    if out_dtype == np.uint8:
        out_image = sharpened_clipped.astype(np.uint8)

    elif out_dtype == np.uint16:
        out_image = (sharpened_clipped.astype(np.uint16) * 257)

    else:
        out_image = sharpened_clipped.astype(out_dtype)

    end_time = time.perf_counter()
    print(f"{'apply_unsharp_masking execution time:':<36}{end_time - start_time:.4f} seconds")

    return out_image


if __name__ == "__main__":
    image = "/home/user1/learning/Testing/StructureModule/Inputs/Input.jpg"

    result = apply_unsharp_masking(
            image,
            blur_ksize=(5, 5),
            blur_sigma_x=0.0,
            blur_sigma_y=0.0,
            mask_scale=0.6,
            sharp_alpha=1.5,
            out_min=0.0,
            out_max=255.0,
            output_dtype=np.uint8
        )

    original = cv.imread(image, cv.IMREAD_GRAYSCALE)

    scores = compare_images(original, result)

    print(f"Sharpness gain : {scores['sharpness_gain']:.2f}x")
    print(f"SSIM           : {scores['ssim']:.4f}")

    # For display
    original = cv.imread(image)
    cv.imshow("Original Image", original)
    cv.imshow("USM Result", result)

    cv.waitKey(0)
    cv.destroyAllWindows()
