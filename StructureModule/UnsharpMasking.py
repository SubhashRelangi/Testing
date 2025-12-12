import cv2 as cv
import numpy as np
import time
from typing import Tuple, Union, Optional

ImageArray = np.ndarray


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
    mask_add_bias: float = 128.0,      # kept for internal consistency
    mask_clip_min: float = 0.0,
    mask_clip_max: float = 255.0,
    out_min: float = 0.0,
    out_max: float = 255.0,
    output_dtype: Optional[Union[str, np.dtype]] = None,
) -> ImageArray:
    """
    Apply Unsharp Masking (USM) to an image.
    Returns ONLY the sharpened output image.
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
        output = sharpened_clipped.astype(np.uint8)

    elif out_dtype == np.uint16:
        output = (sharpened_clipped.astype(np.uint16) * 257)

    else:
        output = sharpened_clipped.astype(out_dtype)

    end_time = time.perf_counter()
    print(f"{'apply_unsharp_masking execution time:':<36}{end_time - start_time:.4f} seconds")

    return output


if __name__ == "__main__":
    image = "/home/user1/learning/Testing/StructureModule/Inputs/Input.jpg"

    result = apply_unsharp_masking(image)

    # For display
    original = cv.imread(image)
    cv.imshow("Original Image", original)
    cv.imshow("USM Result", result)

    cv.waitKey(0)
    cv.destroyAllWindows()
