"""
heatmap_gradientmap.py

Thermal Image Visualization Utilities

Exports:
    - heat_map(gray_image, ...)
    - gradient_map(gray_image, ...)

Both return:
    (output_image | None, metadata_dict)

This module:
    ✔ Provides full parameter documentation (description, min/max, units, defaults, best-case)
    ✔ Performs strict validation & exception handling
    ✔ Produces clean 3-channel BGR heatmaps
    ✔ Does NOT perform any stacking or concatenation
    ✔ Does NOT show images or print anything (library-safe)
"""

import cv2 as cv
import numpy as np
from typing import Tuple, Dict, Any, Optional

# ---------------------------------
# Exceptions
# ---------------------------------
class HeatMapError(Exception):
    """Base exception for heatmap operations."""
    pass


class InvalidParameterErrorHM(HeatMapError):
    """Raised when invalid input or parameters are provided."""
    pass


# ---------------------------------
# Helpers
# ---------------------------------
def _to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Safely convert any numeric image → uint8

    Behavior:
        - NaN → 0
        - +INF → 255
        - -INF → 0
        - Clip range [0,255]
    """
    arr = np.asarray(img, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0, posinf=255, neginf=0)
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
    """Ensures input is a valid single-channel numeric image."""
    if img is None:
        raise InvalidParameterErrorHM("Input image is None")

    arr = np.asarray(img)

    if arr.ndim != 2:
        raise InvalidParameterErrorHM("Image must be single-channel (H×W), not multi-channel")

    if not np.issubdtype(arr.dtype, np.number):
        raise InvalidParameterErrorHM("Image dtype must be numeric")

    return arr


# ============================================================
# 1. HEAT MAP
# ============================================================
def heat_map(
    gray_image: np.ndarray,
    *,
    colormap: int = cv.COLORMAP_JET,
    normalize: bool = False,
    norm_alpha: float = 0.0,   # units: intensity
    norm_beta: float = 255.0,  # units: intensity
    norm_type: int = cv.NORM_MINMAX,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Generate a 3-channel BGR heatmap from a 1-channel grayscale image.

    PARAMETERS
    ----------
    gray_image : ndarray
        Description: Input thermal / grayscale image (single-channel)
        Units: pixel intensity
        Min/Max: float or uint types accepted
        Best case: float32 or uint16 raw sensor data

    colormap : int
        Description: OpenCV colormap constant
        Min/Max: must match cv.COLORMAP_* constants
        Default: cv.COLORMAP_JET
        Best-case: JET or TURBO for thermal contrast

    normalize : bool
        Description: Whether to apply cv.normalize before colormap
        Units: boolean
        Min/Max: True/False
        Default: False
        Best-case: True for raw thermal sensor floats

    norm_alpha / norm_beta : float
        Description: Output normalization intensity range
        Units: intensity values
        Min/Max: typically 0 → 255
        Default: 0, 255
        Best-case: 0–255 for heatmaps

    norm_type : int
        Description: OpenCV normalization type
        Default: cv.NORM_MINMAX

    RETURNS
    -------
    heatmap : ndarray | None
        3-channel BGR heatmap (H×W×3), uint8

    metadata : dict
        Contains:
            - method
            - shape
            - dtype
            - colormap used
            - normalized flag
            - error message (None if success)
    """
    meta = {"method": "heat_map", "error": None}

    try:
        img = _ensure_grayscale(gray_image)

        if normalize:
            img_norm = cv.normalize(img, None, norm_alpha, norm_beta, norm_type)
            img_u8 = _to_uint8(img_norm)
        else:
            img_u8 = _to_uint8(img)

        heat = cv.applyColorMap(img_u8, int(colormap))

        meta.update({
            "shape": heat.shape,
            "dtype": heat.dtype.name,
            "normalized": normalize,
            "colormap": colormap,
        })

        return heat, meta

    except HeatMapError as he:
        meta["error"] = str(he)
        return None, meta

    except Exception as ex:
        meta["error"] = f"Runtime error: {ex}"
        return None, meta


# ============================================================
# 2. GRADIENT MAP
# ============================================================
def gradient_map(
    gray_image: np.ndarray,
    *,
    dx: int = 1,   # units: sobel order
    dy: int = 1,   # units: sobel order
    ddepth: int = cv.CV_64F,
    ksize: int = 3,  # units: pixels
    sobel_scale: float = 1.0,
    sobel_delta: float = 0.0,
    sobel_border: int = cv.BORDER_DEFAULT,
    normalize_output: bool = True,
    norm_alpha: float = 0,
    norm_beta: float = 255,
    norm_type: int = cv.NORM_MINMAX,
    colormap: int = cv.COLORMAP_JET,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
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
    meta = {"method": "gradient_map", "error": None}

    try:
        img = _ensure_grayscale(gray_image)

        # Validate parameters
        ksize = int(ksize)
        if ksize not in (1, 3, 5, 7):
            raise InvalidParameterErrorHM("ksize must be 1, 3, 5, 7")

        dx = int(dx)
        dy = int(dy)
        ddepth = int(ddepth)

        # Sobel conversion for stability
        src = img.astype(np.float32)

        grad_x = cv.Sobel(
            src=src, ddepth=ddepth,
            dx=dx, dy=0,
            ksize=ksize,
            scale=float(sobel_scale),
            delta=float(sobel_delta),
            borderType=sobel_border
        )

        grad_y = cv.Sobel(
            src=src, ddepth=ddepth,
            dx=0, dy=dy,
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

        meta.update({
            "dx": dx,
            "dy": dy,
            "ksize": ksize,
            "shape": grad_heat.shape,
            "dtype": grad_heat.dtype.name,
            "normalized": normalize_output,
            "colormap": colormap,
        })

        return grad_heat, meta

    except HeatMapError as he:
        meta["error"] = str(he)
        return None, meta

    except Exception as ex:
        meta["error"] = f"Runtime error: {ex}"
        return None, meta



# ============================================================
# DEMO (executable)
# ============================================================
if __name__ == "__main__":
    import cv2 as cv
    import os

    # -------- SIMPLE FIXED INPUT PATH --------
    sample_path = "StructureModule/Inputs/Input.jpg"

    # -------- LOAD IMAGE --------
    if not os.path.exists(sample_path):
        print("Image not found:", sample_path)
    else:
        img = cv.imread(sample_path, cv.IMREAD_GRAYSCALE)
        print("Loaded image:", img is not None)
        if img is None:
            print("Failed to read image as grayscale")
        else:
            # -------- COMPUTE MAPS --------
            heat, meta1 = heat_map(img)
            grad, meta2 = gradient_map(img)

            print("\nHeat Map Metadata:", meta1)
            print("Gradient Map Metadata:", meta2)

            # -------- DISPLAY OUTPUT --------
            if heat is not None:
                cv.imshow("Heat Map", heat)
            else:
                print("Heat map generation failed:", meta1["error"])

            if grad is not None:
                cv.imshow("Gradient Map", grad)
            else:
                print("Gradient map generation failed:", meta2["error"])


            print("Input shape:", img.shape)
            if heat is not None:
                print("Heat shape:", heat.shape)
            if grad is not None:
                print("Grad shape:", grad.shape)


            cv.waitKey(0)
            cv.destroyAllWindows()