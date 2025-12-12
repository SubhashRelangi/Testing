import numpy as np
import cv2
from typing import Optional, Union, Tuple
import time

ImageArray = np.ndarray


def _ensure_odd_positive_int_pair(ksize: Tuple[int, int]) -> Tuple[int, int]:
    """Ensure Gaussian kernel size is positive & odd."""
    if not (isinstance(ksize, tuple) and len(ksize) == 2):
        raise TypeError("ksize must be a tuple of two positive odd integers.")
    
    kx, ky = int(ksize[0]), int(ksize[1])
    if kx <= 0 or ky <= 0:
        raise ValueError("Kernel sizes must be > 0.")

    # enforce odd sizes
    if kx % 2 == 0: kx += 1
    if ky % 2 == 0: ky += 1

    return (kx, ky)


def gaussian_filter(
    image: Union[str, ImageArray],
    *,
    ksize: Tuple[int, int] = (5, 5),
    sigma_x: float = 0.0,
    sigma_y: float = 0.0,
    output_dtype: Optional[Union[str, np.dtype]] = None,
) -> ImageArray:
    """
    Apply Gaussian Blur to thermal images.

    STRICT RULE:
    - If user provides a 3-channel image → raise ValueError
    - If image is 1-channel (HxW or HxWx1) → execute normally
    """

    start_time = time.perf_counter()

    # -------------------------
    # Load image
    # -------------------------
    if image is None:
        raise ValueError("Input image is None.")

    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image at path: {image}")
    else:
        img = np.asarray(image)

    if img.size == 0:
        raise ValueError("Input image is empty.")

    # -------------------------
    # CHANNEL VALIDATION
    # -------------------------
    if img.ndim == 3:
        if img.shape[2] == 3:
            raise ValueError(
                "Gaussian filter expects a single-channel thermal image, "
                "but a 3-channel image was provided."
            )
        elif img.shape[2] == 1:
            img = img[..., 0]   # safe squeeze
        else:
            raise ValueError(f"Unsupported channel count: {img.shape[2]}")

    elif img.ndim != 2:
        raise ValueError(f"Expected 1-channel grayscale image, got shape {img.shape}")

    # At this point: img is guaranteed to be 2D (H×W)
    orig_dtype = img.dtype

    # -------------------------
    # Normalize to float32
    # -------------------------
    if np.issubdtype(orig_dtype, np.floating):
        img_f = np.clip(img, 0.0, 1.0).astype(np.float32) * 255.0

    elif orig_dtype == np.uint8:
        img_f = img.astype(np.float32)

    elif orig_dtype == np.uint16:
        img_f = (img.astype(np.float32) / 65535.0) * 255.0

    else:
        raise TypeError(
            f"Unsupported dtype {orig_dtype}. "
            "Allowed: uint8, uint16, float32, float64."
        )

    # -------------------------
    # Ensure valid kernel size
    # -------------------------
    ksize = _ensure_odd_positive_int_pair(ksize)

    # -------------------------
    # Apply Gaussian Blur
    # -------------------------
    filtered_f = cv2.GaussianBlur(
        img_f,
        ksize=ksize,
        sigmaX=float(sigma_x),
        sigmaY=float(sigma_y),
    )

    # -------------------------
    # Output dtype handling
    # -------------------------
    if output_dtype is None:
        out_dtype = np.uint8
    else:
        out_dtype = np.dtype(output_dtype)

    if out_dtype == np.uint8:
        output = np.clip(filtered_f, 0, 255).astype(np.uint8)

    elif out_dtype == np.uint16:
        output = (np.clip(filtered_f, 0, 255).astype(np.uint16) * 257)

    else:
        output = filtered_f.astype(out_dtype)

    end_time = time.perf_counter()
    print(f"{'gaussian_filter execution time:':<36}{end_time - start_time:.4f} seconds")

    return output


if __name__ == "__main__":
    
    image = "/home/user1/learning/Testing/NoiseReduction/Inputs/SingleChannel.png"

    gaussian = gaussian_filter(image)

    original = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    stackedOutput = np.hstack([original, gaussian])

    cv2.imshow("Original | Gaussian", stackedOutput)

    cv2.imwrite("/home/user1/learning/Testing/NoiseReduction/Outputs/GaussiamFilter.png", stackedOutput)

    cv2.waitKey(0)
    cv2.destroyAllWindows()