import cv2 as cv
import numpy as np
import sys
import time
from typing import Optional

Image = np.ndarray


def apply_scharr_operator(
    image: Image,
    *,
    # Pre-blur
    blur_ksize: int = 3,
    # Scharr operator parameters (fully exposed)
    scharr_ddepth: int = cv.CV_32F,
    dx_scharr_x: int = 1,
    dy_scharr_x: int = 0,
    dx_scharr_y: int = 0,
    dy_scharr_y: int = 1,
    # Normalization parameters (for cv.normalize)
    norm_dst: Optional[np.ndarray] = None,
    norm_alpha: float = 0.0,
    norm_beta: float = 255.0,
    norm_type: int = cv.NORM_MINMAX,
    # Thresholding parameters (for cv.threshold)
    threshold_val: float = 50.0,
    threshold_max_value: float = 255.0,
    threshold_type: int = cv.THRESH_BINARY,
) -> Image:
    """
    Apply the Scharr operator to compute gradient magnitude and return a binary edge map.
    All numerical and algorithmic choices are exposed as parameters so callers control
    each internal operation.

    -------------------------------------------------------------------------
    PARAMETER SPECIFICATIONS (Description / Min & Max / Units / Default / Best-case)
    -------------------------------------------------------------------------

    image : np.ndarray
        Description: Input single-channel grayscale/thermal image.
        Min & Max: 2D array with H,W >= 1.
        Units: Pixel intensity.
        Default: Required.
        Best case: Clean uint8 thermal or camera grayscale frame.

    blur_ksize : int
        Description: Gaussian blur kernel size applied before Scharr.
        Min & Max: Odd integers >= 1.
        Units: Pixels (kernel size).
        Default: 3.
        Best case: 3–7 to reduce noise without losing edges.

    scharr_ddepth : int
        Description: OpenCV depth for Scharr output (e.g., cv.CV_32F).
        Min & Max: Any valid cv depth enum (cv.CV_16S, cv.CV_32F, cv.CV_64F, ...).
        Units: OpenCV enum.
        Default: cv.CV_32F.
        Best case: cv.CV_32F to preserve sign and magnitude.

    dx_scharr_x, dy_scharr_x : int
        Description: Derivative orders passed to Scharr for the X call (dx, dy).
        Min & Max: 0–2.
        Units: Derivative order.
        Default: (1, 0).
        Best case: (1,0) for standard X-gradient.

    dx_scharr_y, dy_scharr_y : int
        Description: Derivative orders passed to Scharr for the Y call (dx, dy).
        Min & Max: 0–2.
        Units: Derivative order.
        Default: (0, 1).
        Best case: (0,1) for standard Y-gradient.

    norm_dst : Optional[np.ndarray]
        Description: Destination array for cv.normalize (or None to let OpenCV allocate).
        Min & Max: None or an array matching image shape.
        Units: ndarray or None.
        Default: None.
        Best case: None for convenience.

    norm_alpha : float
        Description: Lower bound after normalization (alpha in cv.normalize).
        Min & Max: 0.0 → 65535.0 (practical).
        Units: Pixel intensity.
        Default: 0.0.
        Best case: 0.0 for full-range scaling.

    norm_beta : float
        Description: Upper bound after normalization (beta in cv.normalize).
        Min & Max: 1.0 → 65535.0 (practical).
        Units: Pixel intensity.
        Default: 255.0.
        Best case: 255.0 for uint8 visualization.

    norm_type : int
        Description: Normalization type (cv.NORM_MINMAX, cv.NORM_INF, ...).
        Min & Max: Valid cv.normalize enums.
        Units: OpenCV enum.
        Default: cv.NORM_MINMAX.
        Best case: cv.NORM_MINMAX for per-frame contrast scaling.

    threshold_val : float
        Description: Threshold level applied to the normalized magnitude.
        Min & Max: 0.0 → norm_beta (practical).
        Units: Pixel intensity (uint8 scale).
        Default: 50.0.
        Best case: 30–70 for typical thermal edge maps.

    threshold_max_value : float
        Description: Value assigned to pixels exceeding the threshold (maxVal).
        Min & Max: >0.0 → 65535.0 (practical).
        Units: Pixel intensity.
        Default: 255.0.
        Best case: 255.0 for crisp binary maps.

    threshold_type : int
        Description: cv threshold type (cv.THRESH_BINARY, cv.THRESH_BINARY_INV, ...).
        Min & Max: Valid cv threshold enums.
        Units: OpenCV enum.
        Default: cv.THRESH_BINARY.
        Best case: cv.THRESH_BINARY for standard edges.

    RETURNS
    -------
    edge_map : np.ndarray (uint8, H×W)
        Binary edge map with values {0, threshold_max_value} (clipped to uint8 if <=255).
    -------------------------------------------------------------------------
    """

    start_time = time.time()
   
    # -----------------------
    # Validation
    # -----------------------
    if image is None:
        raise ValueError("Input `image` is None.")
    if image.ndim != 2:
        raise ValueError("Input must be a single-channel (grayscale) image.")
    if blur_ksize < 1 or blur_ksize % 2 == 0:
        raise ValueError("`blur_ksize` must be an odd integer >= 1.")
    if not (0.0 <= norm_alpha <= norm_beta):
        raise ValueError("`norm_alpha` must be <= `norm_beta` and both non-negative.")
    if not (0.0 <= threshold_val <= max(255.0, norm_beta)):
        raise ValueError("`threshold_val` must be in a reasonable range relative to norm_beta.")
    if not (threshold_max_value > 0.0):
        raise ValueError("`threshold_max_value` must be > 0.")
    for v in (dx_scharr_x, dy_scharr_x, dx_scharr_y, dy_scharr_y):
        if not (0 <= v <= 2):
            raise ValueError("Scharr derivative orders must be integers in [0, 2].")

    # -----------------------
    # Processing pipeline
    # -----------------------

    # 1) Pre-blur
    image_blur = cv.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

    # 2) Scharr gradients using external parameters
    scharr_x = cv.Scharr(image_blur, scharr_ddepth, dx_scharr_x, dy_scharr_x)
    scharr_y = cv.Scharr(image_blur, scharr_ddepth, dx_scharr_y, dy_scharr_y)

    # 3) Gradient magnitude
    gradient_magnitude = cv.magnitude(scharr_x, scharr_y)

    # 4) Normalize using parameters from outside
    # cv.normalize returns the normalized dst (or a new array if dst is None)
    magnitude_norm = cv.normalize(
        gradient_magnitude,
        norm_dst,
        norm_alpha,
        norm_beta,
        norm_type
    )

    # 5) Convert to 8-bit (clamp) if thresholding expects 0..255; otherwise clip to threshold_max_value range
    # We choose to produce a uint8 binary map when threshold_max_value <= 255, else produce uint16.
    if threshold_max_value <= 255.0:
        magnitude_for_thresh = np.clip(np.rint(magnitude_norm), 0, 255).astype(np.uint8)
        thresh_max_cast = float(threshold_max_value)
    else:
        # keep higher range in uint16 to preserve dynamic range
        magnitude_for_thresh = np.clip(np.rint(magnitude_norm), 0, threshold_max_value).astype(np.uint16)
        thresh_max_cast = float(threshold_max_value)

    # 6) Threshold using externally provided params
    _, out_image = cv.threshold(magnitude_for_thresh, threshold_val, thresh_max_cast, threshold_type)

    # 7) If edge_map dtype is > uint8 but user likely wants uint8, optionally downcast if max <=255
    if out_image.dtype != np.uint8 and threshold_max_value <= 255.0:
        out_image = out_image.astype(np.uint8)

    end_time = time.time()
    print(f"[apply_scharr_operator] Start: {start_time:.6f}  End: {end_time:.6f}  Duration: {end_time - start_time:.6f} sec")

    return out_image


if __name__ == "__main__":
    # Path to image (change to your path)
    image_path = "/home/user1/learning/Testing/StructureModule/Inputs/Input.jpg"

    # Read as grayscale (function expects a single-channel image)
    src = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if src is None:
        print(f"ERROR: Failed to load image at: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Call your operator (keeps texture/logic unchanged)
    edge_map = apply_scharr_operator(image=src)

    # Show results
    cv.imshow("input_gray", src)
    cv.imshow("apply_scharr_edge_map", edge_map)
    cv.waitKey(0)
    cv.destroyAllWindows()
