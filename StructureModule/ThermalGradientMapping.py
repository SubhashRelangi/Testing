"""
heatmap_gradientmap.py

Thermal Image Visualization Utilities

Exports:
    - heat_map(gray_image, ...)
    - gradient_map(gray_image, ...)

This module:
    ✔ Performs strict validation
    ✔ Produces clean 3-channel BGR heatmaps
    ✔ No metadata is returned anymore
    ✔ Prints processing start/end time and total duration
"""

import cv2 as cv
import numpy as np
import time
from typing import Optional


# ---------------------------------
# Exceptions
# ---------------------------------
class GradientMapError(Exception):
    """Base exception for heatmap operations."""
    pass


class InvalidParameterErrorHM(GradientMapError):
    """Raised when invalid input or parameters are provided."""
    pass


# ---------------------------------
# Helpers
# ---------------------------------
def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Safely convert numeric image → uint8 with clipping."""
    arr = np.asarray(img, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0, posinf=255, neginf=0)
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
    """Ensures input is valid single-channel image."""
    if img is None:
        raise InvalidParameterErrorHM("Input image is None")
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise InvalidParameterErrorHM("Image must be single-channel (H×W)")
    if not np.issubdtype(arr.dtype, np.number):
        raise InvalidParameterErrorHM("Image dtype must be numeric")
    return arr


# ============================================================
# GRADIENT MAP — RETURNS ONLY THE IMAGE
# ============================================================
def gradient_map(
    gray_image: np.ndarray,
    *,
    dx: int = 1,
    dy: int = 1,
    ddepth: int = cv.CV_64F,
    ksize: int = 3,
    sobel_scale: float = 1.0,
    sobel_delta: float = 0.0,
    sobel_border: int = cv.BORDER_DEFAULT,
    normalize_output: bool = True,
    norm_alpha: float = 0,
    norm_beta: float = 255,
    norm_type: int = cv.NORM_MINMAX,
    colormap: int = cv.COLORMAP_JET,
) -> Optional[np.ndarray]:
    
    """
    Compute Sobel gradient magnitude → normalize → colorize (3-channel).

    PARAMETERS
    ----------
    gray_image : ndarray
        Description: Input thermal / grayscale data
        Units: pixel intensity
        Min/Max: float or uint types
        Best-case: float32

    dx, dy : int
        Description: Sobel derivative order in x and y
        Min/Max: 0–2
        Default: 1,1
        Best-case: dx=1, dy=1 for full gradient

    ddepth : int
        Description: Output depth type
        Default: CV_64F (best precision)

    ksize : int
        Description: Sobel kernel size
        Units: pixels
        Min/Max: 1,3,5,7
        Default: 3
        Best-case: 3 (balanced detail)

    sobel_scale : float
        Min/Max: >0
        Default: 1.0
        Units: multiplier

    sobel_delta : float
        Min/Max: any float
        Default: 0.0
        Units: offset

    sobel_border : int
        Default: cv.BORDER_DEFAULT

    normalize_output : bool
        Description: Normalize gradient magnitude before colormap
        Default: True

    RETURNS
    -------
    grad_heatmap : ndarray | None
        3-channel BGR gradient heatmap (H×W×3), uint8

    metadata : dict
    """

    start_time = time.time()

    try:
        img = _ensure_grayscale(gray_image)

        # Parameter validation
        ksize = int(ksize)
        if ksize not in (1, 3, 5, 7):
            raise InvalidParameterErrorHM("ksize must be 1, 3, 5, 7")

        dx = int(dx)
        dy = int(dy)
        ddepth = int(ddepth)

        src = img.astype(np.float32)

        # Sobel
        grad_x = cv.Sobel(
            src, ddepth, dx, 0,
            ksize=ksize,
            scale=float(sobel_scale),
            delta=float(sobel_delta),
            borderType=sobel_border
        )

        grad_y = cv.Sobel(
            src, ddepth, 0, dy,
            ksize=ksize,
            scale=float(sobel_scale),
            delta=float(sobel_delta),
            borderType=sobel_border
        )

        magnitude = cv.magnitude(grad_x, grad_y)

        if normalize_output:
            mag_norm = cv.normalize(magnitude, None, norm_alpha, norm_beta, norm_type)
            mag_u8 = _to_uint8(mag_norm)
        else:
            mag_u8 = _to_uint8(magnitude)

        grad_heat = cv.applyColorMap(mag_u8, colormap)


        end_time = time.time()
        print(f"[gradient_map] Start: {start_time:.6f}  End: {end_time:.6f}  Duration: {end_time - start_time:.6f} sec")

        return grad_heat

    except Exception as ex:
        print(f"[gradient_map] ERROR: {ex}")
        return None


# ============================================================
# DEMO
# ============================================================
if __name__ == "__main__":
    import os
    import cv2 as cv

    sample_path = "StructureModule/Inputs/Input.jpg"

    if not os.path.exists(sample_path):
        print("Image not found:", sample_path)
    else:
        img = cv.imread(sample_path, cv.IMREAD_GRAYSCALE)

        if img is None:
            print("Failed to load image")
        else:
            result = gradient_map(img)

            if result is not None:
                cv.imshow("Gradient Map", result)
                cv.waitKey(0)
                cv.destroyAllWindows()
            else:
                print("Gradient map generation failed.")
