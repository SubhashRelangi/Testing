"""
heatmap_gradientmap.py

Both functions use the same standardized structure:
 - heat_map(...)
 - gradient_map(...)

Both return:
    (result_image | None, metadata_dict)

All internal constants removed — fully parameterized.
Exception handling included.
"""

import cv2 as cv
import numpy as np
from typing import Optional, Tuple, Dict, Any

Image = np.ndarray


# -------------------------------
# Exceptions
# -------------------------------
class HeatMapError(Exception):
    pass


class InvalidParameterErrorHM(HeatMapError):
    pass


# -------------------------------
# Helper
# -------------------------------
def _to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)


# ============================================================
# 1. HEAT MAP (applies colormap to raw grayscale)
# ============================================================
def heat_map(
    gray_image: Image,
    *,
    colormap: int = cv.COLORMAP_JET,
    normalize: bool = False,
    norm_alpha: float = 0.0,
    norm_beta: float = 255.0,
    norm_type: int = cv.NORM_MINMAX,
) -> Tuple[Optional[Image], Dict[str, Any]]:
    """
    Create a heatmap from a grayscale image using a configurable colormap.

    Args:
      gray_image (Image):
          Input thermal/grayscale image.

      colormap (int):
          OpenCV colormap constant.
          Min/Max: must be valid cv.COLORMAP_*
          Default: cv.COLORMAP_JET
          Best case: JET for thermal visualization.

      normalize (bool):
          Whether to normalize grayscale before mapping.
          Default: False

      norm_alpha, norm_beta, norm_type:
          Normalization parameters if normalize=True.

    Returns:
      heatmap (Image | None)
      metadata (dict)
    """
    meta: Dict[str, Any] = {"method": "heat_map", "error": None}

    try:
        if gray_image is None:
            raise InvalidParameterErrorHM("gray_image is None")
        if gray_image.ndim != 2:
            raise InvalidParameterErrorHM("gray_image must be single-channel")

        if normalize:
            gray_norm = cv.normalize(gray_image, None, norm_alpha, norm_beta, norm_type)
            gray_u8 = _to_uint8(gray_norm)
        else:
            gray_u8 = _to_uint8(gray_image)

        heatmap = cv.applyColorMap(gray_u8, colormap)

        meta.update({
            "shape": heatmap.shape,
            "dtype": heatmap.dtype.name,
            "normalized": normalize,
            "colormap": colormap,
        })

        return heatmap, meta

    except HeatMapError as he:
        meta["error"] = str(he)
        return None, meta

    except Exception as ex:
        meta["error"] = f"runtime: {ex}"
        return None, meta


# ============================================================
# 2. GRADIENT MAP (Sobel → magnitude → heatmap)
# ============================================================
def gradient_map(
    gray_image: Image,
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
) -> Tuple[Optional[Image], Dict[str, Any]]:
    """
    Compute gradient heatmap using Sobel → magnitude → colormap.

    Args:
      dx, dy:
          Gradient direction orders.
          Min/Max: 0–2
          Default: 1,1

      ddepth (int):
          Sobel output depth.
          Default: cv.CV_64F

      ksize (int):
          Kernel size: 1,3,5,7
          Default: 3

      sobel_scale (float):
          Multiply gradient output.
          Default: 1.0

      sobel_delta (float):
          Added to gradient result.
          Default: 0.0

      sobel_border (int):
          Border handling type.
          Default: cv.BORDER_DEFAULT

      normalize_output (bool):
          Normalize gradient magnitude.
          Default: True

      colormap (int):
          Heatmap style.
          Default: cv.COLORMAP_JET

    Returns:
       heatmap (Image | None)
       metadata (dict)
    """
    meta: Dict[str, Any] = {"method": "gradient_map", "error": None}

    try:
        if gray_image is None:
            raise InvalidParameterErrorHM("gray_image is None")
        if gray_image.ndim != 2:
            raise InvalidParameterErrorHM("gray_image must be single-channel")

        if ksize not in (1, 3, 5, 7):
            raise InvalidParameterErrorHM("ksize must be 1,3,5,7")

        grad_x = cv.Sobel(gray_image, ddepth, dx, 0, ksize, sobel_scale, sobel_delta, sobel_border)
        grad_y = cv.Sobel(gray_image, ddepth, 0, dy, ksize, sobel_scale, sobel_delta, sobel_border)

        magnitude = cv.magnitude(grad_x, grad_y)

        if normalize_output:
            mag_norm = cv.normalize(magnitude, None, norm_alpha, norm_beta, norm_type)
            mag_u8 = _to_uint8(mag_norm)
        else:
            mag_u8 = _to_uint8(magnitude)

        heatmap = cv.applyColorMap(mag_u8, colormap)

        meta.update({
            "dx": dx, "dy": dy,
            "ksize": ksize,
            "ddepth": ddepth,
            "scale": sobel_scale,
            "delta": sobel_delta,
            "border": sobel_border,
            "normalized": normalize_output,
            "shape": heatmap.shape,
            "dtype": heatmap.dtype.name,
        })

        return heatmap, meta

    except HeatMapError as he:
        meta["error"] = str(he)
        return None, meta

    except Exception as ex:
        meta["error"] = f"runtime: {ex}"
        return None, meta


# ============================================================
# DEMO (outside module)
# ============================================================
if __name__ == "__main__":
    img = cv.imread("/home/user1/learning/Testing/StructureModule/Inputs/Input.jpg", cv.IMREAD_GRAYSCALE)

    heat, meta1 = heat_map(img)
    grad, meta2 = gradient_map(img)

    print(meta1)
    print(meta2)

    if heat is not None:
        cv.imshow("Heat Map", heat)
    if grad is not None:
        cv.imshow("Gradient Heat Map", grad)

    cv.waitKey(0)
    cv.destroyAllWindows()
