from typing import Optional, Tuple
import cv2 as cv
import time
import numpy as np

Image = np.ndarray


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float/large arrays into uint8 safely."""
    if img.dtype.kind in np.typecodes['AllFloat']:
        img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _ensure_grayscale(image: Image) -> Optional[Image]:
    """
    Validate input and ensure single-channel grayscale output.

    - If input is None -> returns None
    - If input is 2D -> assumed grayscale, returned as-is
    - If input is 3D (BGR/RGB) -> converted to grayscale via cvtColor and returned
    - Otherwise -> returns None

    This is a small helper so every processing function first normalizes the image shape.
    """
    if image is None:
        return None

    # numpy arrays: check dims
    if not isinstance(image, np.ndarray):
        return None

    if image.ndim == 2:
        return image
    if image.ndim == 3:
        # assume color in BGR (OpenCV default); convert to grayscale
        try:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            return gray
        except Exception:
            # try converting from RGB if BGR conversion fails
            try:
                gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                return gray
            except Exception:
                return None
    # unsupported dimension
    return None


def subtract_low_pass(
    image: Image,
    *,
    gblur_ksize: Tuple[int, int] = (0, 0),
    sigma: float = 3.0,
    offset: float = 127.0,
    min_clip: int = 0,
    max_clip: int = 255
) -> Optional[Image]:
    """
    Applies a high-pass filter by subtracting a Gaussian blurred image (low-pass) 
    from the original. 

    PARAMETERS
    ----------
    image : ndarray
        Description: The input grayscale image (single-channel).
        Units: pixel intensity
        Min/Max: uint8 accepted

    gblur_ksize : Tuple[int, int]
        Description: Kernel size (width, height) for the Gaussian blur.
        Units: pixels (must be odd)
        Default: (0, 0) - size is calculated from sigma.
        Best case: (5, 5) or similar odd size.

    sigma : float
        Description: Standard deviation for the Gaussian blur.
        Units: pixels
        Min/Max: 0.1 to 10.0 typically
        Default: 3.0

    offset : float
        Description: Offset added to re-center high-pass result around mid-gray.
        Units: intensity value
        Min/Max: 0 to 255
        Default: 127.0

    min_clip / max_clip : int
        Description: Output clipping range.
        Units: intensity values
        Default: 0, 255

    RETURNS
    -------
    result : ndarray | None
        The high-pass filtered image (uint8).
"""
    start_time = time.perf_counter()

    # validation & grayscale ensure
    img = _ensure_grayscale(image)
    if img is None:
        return None

    try:
        img_f = img.astype(np.float32)
        blurred = cv.GaussianBlur(img_f, gblur_ksize, sigma)
        high_pass = img_f - blurred + offset
        out_image = np.clip(high_pass, min_clip, max_clip).astype(np.uint8)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"[subtract_low_pass] Time Taken: {total_time:.6f} seconds")


        return out_image

    except Exception:
        return None

def convolve_with_kernel(
    image: Image,
    *,
    kernel: np.ndarray = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    ddepth: int = -1
) -> Optional[Image]:
    """
    Applies 2D convolution using a predefined kernel (e.g., sharpening).
    

    PARAMETERS
    ----------
    image : ndarray
        Description: The input grayscale image (single-channel).
        Units: pixel intensity

    kernel : ndarray
        Description: The 2D convolution kernel (e.g., Laplacian, Sharpening).
        Units: float array
        Best case: Standard, validated kernels (e.g., 3x3 array of floats).

    ddepth : int
        Description: Desired depth of the destination image.
        Units: OpenCV constant (e.g., CV_8U, CV_32F)
        Default: -1 (same as source depth, which is uint8).

    RETURNS
    -------
    result : ndarray | None
        The convolved image (uint8).
    """
    start_time = time.perf_counter()

    img = _ensure_grayscale(image)
    if img is None:
        return None

    try:
        # filter2D preserves shape; ddepth=-1 keeps dtype
        filtered = cv.filter2D(img, ddepth=ddepth, kernel=kernel)
        # normalize/clip to uint8 for safe downstream use
        out_image = _to_uint8(filtered)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"[convolve_with_kernel] Time Taken: {total_time:.6f} seconds")

        return out_image

    except Exception:
        return None


def apply_laplacian_detector(
    image: Image,
    *,
    ksize: int = 3,
    ddepth: int = cv.CV_64F
) -> Optional[Image]:
    """
    Applies the Laplacian edge detection operator (second-order derivative).
    

    PARAMETERS
    ----------
    image : ndarray
        Description: The input grayscale image (single-channel).

    ksize : int
        Description: Aperture size for the Laplacian operator.
        Units: pixels
        Min/Max: 1 to 31 (must be odd)
        Default: 3

    ddepth : int
        Description: Output depth for the derivative calculation.
        Units: OpenCV constant
        Default: cv.CV_64F (ensures no clipping before conversion to uint8).

    RETURNS
    -------
    result : ndarray | None
        The Laplacian edge map (uint8).
    """
    start_time = time.perf_counter()

    img = _ensure_grayscale(image)
    if img is None:
        return None

    try:
        lap = cv.Laplacian(img, ddepth=ddepth, ksize=ksize)
        out_image = _to_uint8(np.abs(lap))

        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"[apply_laplacian_detector] Time Taken: {total_time:.6f} seconds")

        return out_image

    except Exception:
        return None




def apply_sobel_xy_detectors(
    image: Image,
    *,
    ksize: int = 3,
    ddepth: int = cv.CV_64F,
    dx_sobel_x: int = 1,
    dy_sobel_x: int = 0,
    dx_sobel_y: int = 0,
    dy_sobel_y: int = 1,
    # ---- NEW NORMALIZATION PARAMETERS ----
    norm_min: int = 0,
    norm_max: int = 255,
    norm_type: int = cv.NORM_MINMAX,
    norm_dtype: int = cv.CV_8U
) -> Optional[Image]:
    """
    Applies Sobel X and Y, computes gradient magnitude, normalizes by user-specified
    parameters, and returns a uint8 magnitude image. Returns None on failure.

    -------------------------------------------------------------------------
    PARAMETER SPECIFICATIONS 
    -------------------------------------------------------------------------

    image : np.ndarray  
        Description: Input thermal or grayscale image.  
        Min & Max: ≥ 1×1 pixels.  
        Units: Pixel intensity.  
        Default: Required.  
        Best case: Clean uint8 thermal frame.

    ksize : int  
        Description: Sobel aperture size (odd).  
        Min & Max: 1 → 31 (odd only).  
        Units: Pixels.  
        Default: 3.  
        Best case: 3 for thermal edge enhancement.

    ddepth : int  
        Description: Depth of Sobel output.  
        Min & Max: OpenCV depth enums.  
        Units: cv.CV_*  
        Default: cv.CV_64F  
        Best case: CV_64F for high precision.

    dx_sobel_x / dy_sobel_x : int  
        Description: Derivative orders for Sobel-X.  
        Min & Max: 0–2.  
        Units: Order.  
        Default: 1, 0.  
        Best case: (1,0) for pure X-gradient.

    dx_sobel_y / dy_sobel_y : int  
        Description: Derivative orders for Sobel-Y.  
        Min & Max: 0–2.  
        Units: Order.  
        Default: (0,1).  
        Best case: (0,1) for pure Y-gradient.

    ---------------- NORMALIZATION PARAMETERS ----------------

    norm_min : int  
        Description: Minimum output intensity after normalization.  
        Min & Max: 0 → 255  
        Units: Pixel intensity.  
        Default: 0  
        Best case: 0 (typical for uint8 visualization).

    norm_max : int  
        Description: Maximum output intensity after normalization.  
        Min & Max: 1 → 255  
        Units: Pixel intensity.  
        Default: 255  
        Best case: 255 for full contrast.

    norm_type : int  
        Description: OpenCV normalization type.  
        Min & Max: Valid cv normalization modes.  
        Units: cv.NORM_*  
        Default: cv.NORM_MINMAX  
        Best case: cv.NORM_MINMAX for edge maps.

    norm_dtype : int  
        Description: Output datatype of normalized image.  
        Min & Max: OpenCV depth enums  
        Units: cv.CV_*  
        Default: cv.CV_8U  
        Best case: CV_8U for display and further processing.

    -------------------------------------------------------------------------
    NOTES:
      - Gradient magnitude: sqrt(SobelX² + SobelY²)
      - Fully customizable normalization
      - Returns uint8 edge map unless user changes norm_dtype
    -------------------------------------------------------------------------

    RETURNS
    -------
    result : ndarray | None
        The Sobel magnitude image.
    """
    start_time = time.perf_counter()

    img = _ensure_grayscale(image)
    if img is None:
        return None

    try:
        sobelx = cv.Sobel(img, ddepth=ddepth, dx=dx_sobel_x, dy=dy_sobel_x, ksize=ksize)
        sobely = cv.Sobel(img, ddepth=ddepth, dx=dx_sobel_y, dy=dy_sobel_y, ksize=ksize)

        magnitude = np.sqrt(sobelx**2 + sobely**2)

        magnitude_norm = cv.normalize(
            magnitude,
            None,
            alpha=norm_min,
            beta=norm_max,
            norm_type=norm_type,
            dtype=norm_dtype
        )

        end_time = time.perf_counter() 
        total_time = end_time - start_time 
        print(f"[apply_sobel_xy_detectors] Time Taken: {total_time:.6f} seconds")

        return magnitude_norm

    except Exception:
        return None

       
 

if __name__ == "__main__":

    image = cv.imread("/home/user1/learning/Testing/NoiseReduction/Inputs/frame_000000.jpg")

    if image is None:
        print("Error: Image not loaded. Check file path.")
        # Exit or handle error

    low_pass = subtract_low_pass(image=image)

    fil = convolve_with_kernel(image=image)

    apply_lap = apply_laplacian_detector(image=image)

    apply_sob = apply_sobel_xy_detectors(image=image)

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    low_pass_stack = np.hstack([gray_image, low_pass])
    fil_stack = np.hstack([gray_image, fil])
    apply_lap_stack = np.hstack([gray_image, apply_lap])
    apply_sob_stack = np.hstack([gray_image, apply_sob])

    cv.imwrite("/home/user1/learning/Testing/NoiseReduction/Outputs/low_pass_stack.png", low_pass_stack)
    cv.imwrite("/home/user1/learning/Testing/NoiseReduction/Outputs/fil_stack.png", fil_stack)
    cv.imwrite("/home/user1/learning/Testing/NoiseReduction/Outputs/apply_lap_stack.png", apply_lap_stack)
    cv.imwrite("/home/user1/learning/Testing/NoiseReduction/Outputs/apply_sob_stack.png", apply_sob_stack)

    cv.waitKey(0)
    cv.destroyAllWindows()
