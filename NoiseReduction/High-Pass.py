from typing import Any, Dict, Optional, Tuple, Union
import cv2 as cv
import numpy as np
import sys
import os

# Type hint alias for image arrays
Image = np.ndarray

# --- Placeholder utility functions (Error class, type converter) ---

class FilterError(Exception):
    """Custom exception for filtering errors."""
    pass

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Clips and converts an image array to uint8."""
    # Assuming standard 0-255 scaling for visualization if input is float
    if img.dtype.kind in np.typecodes['AllFloat']:
        img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

# =====================================================
# 2. CORE PROCESSING FUNCTIONS (MAXIMUM SIGNATURE DETAIL)
# =====================================================

def subtract_low_pass(
    image: Image,
    *,
    gblur_ksize: Tuple[int, int] = (0, 0),
    sigma: float = 3.0,
    offset: float = 127.0,
    min_clip: int = 0,
    max_clip: int = 255
) -> Tuple[Optional[Image], Dict[str, Any]]:
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

    metadata : dict
        Contains method, shape, dtype, and error message.
    """
    meta = {"method": "subtract_low_pass", "error": None}
    if image is None:
        meta["error"] = "Input image is None."
        return None, meta
        
    try:
        image_float = image.astype(np.float32)
        
        blurred_img = cv.GaussianBlur(image_float, gblur_ksize, sigma)
        high_pass_img = image_float - blurred_img + offset
        
        # Clamping based on parameters
        high_pass_result = np.clip(high_pass_img, min_clip, max_clip).astype(np.uint8)
        
        meta.update({"shape": high_pass_result.shape, "dtype": high_pass_result.dtype.name})
        return high_pass_result, meta
    except Exception as exc:
        meta["error"] = f"Runtime error: {exc}"
        return None, meta


def convolve_with_kernel(
    image: Image,
    *,
    kernel: np.ndarray,
    ddepth: int = -1
) -> Tuple[Optional[Image], Dict[str, Any]]:
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

    metadata : dict
        Contains method, shape, dtype, and error message.
    """
    meta = {"method": "convolve_with_kernel", "error": None}
    if image is None:
        meta["error"] = "Input image is None."
        return None, meta
        
    try:
        filtered_image = cv.filter2D(image, ddepth=ddepth, kernel=kernel)
        
        meta.update({"shape": filtered_image.shape, "dtype": filtered_image.dtype.name})
        return filtered_image, meta
    except Exception as exc:
        meta["error"] = f"Runtime error: {exc}"
        return None, meta


def apply_laplacian_detector(
    image: Image,
    *,
    ksize: int = 3,
    ddepth: int = cv.CV_64F
) -> Tuple[Optional[Image], Dict[str, Any]]:
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

    metadata : dict
        Contains method, shape, dtype, and error message.
    """
    meta = {"method": "apply_laplacian_detector", "error": None}
    if image is None:
        meta["error"] = "Input image is None."
        return None, meta

    try:
        # Laplacian Operator (Second-order derivative)
        lap = cv.Laplacian(image, ddepth=ddepth, ksize=ksize)
        
        # Take absolute value and convert back to 8-bit for visualization
        laplacian_8bit = _to_uint8(np.absolute(lap))
        
        meta.update({"shape": laplacian_8bit.shape, "dtype": laplacian_8bit.dtype.name})
        return laplacian_8bit, meta
    except Exception as exc:
        meta["error"] = f"Runtime error: {exc}"
        return None, meta


# ... (rest of the file content before the function definitions) ...

def apply_sobel_xy_detectors(
    image: Image,
    *,
    ksize: int = 3,
    ddepth: int = cv.CV_64F,
    dx_sobel_x: int = 1,
    dy_sobel_x: int = 0,
    dx_sobel_y: int = 0,
    dy_sobel_y: int = 1
) -> Tuple[Optional[Image], Optional[Image], Optional[Image], Dict[str, Any]]:
    """
    Applies Sobel operators for X and Y gradient magnitude calculation.
    (Includes explicit normalization for visualization).
    

    ... (PARAMETERS and RETURNS documentation remains the same) ...
    """
    meta = {"method": "apply_sobel_xy_detectors", "error": None, "x_meta": {}, "y_meta": {}, "mag_meta": {}}
    if image is None:
        meta["error"] = "Input image is None."
        return None, None, None, meta

    try:
        # 1. Sobel Operator (X-direction) - Intermediate float result
        sobelx_float = cv.Sobel(image, ddepth=ddepth, dx=dx_sobel_x, dy=dy_sobel_x, ksize=ksize)

        # 2. Sobel Operator (Y-direction) - Intermediate float result
        sobely_float = cv.Sobel(image, ddepth=ddepth, dx=dx_sobel_y, dy=dy_sobel_y, ksize=ksize)

        # --- A. Visualize X and Y separately (using np.absolute and simple clip/convert) ---
        # Note: np.absolute() handles the negative values, _to_uint8() handles values > 255
        sobelx_8bit = _to_uint8(np.absolute(sobelx_float))
        sobely_8bit = _to_uint8(np.absolute(sobely_float))
        
        meta["x_meta"].update({"shape": sobelx_8bit.shape, "dtype": sobelx_8bit.dtype.name})
        meta["y_meta"].update({"shape": sobely_8bit.shape, "dtype": sobely_8bit.dtype.name})

        # --- B. Sobel Magnitude Calculation: G = sqrt(Gx^2 + Gy^2) ---
        magnitude_float = np.sqrt(np.square(sobelx_float) + np.square(sobely_float))
        
        # --- CRITICAL FIX: Explicit Normalization for Visualization ---
        # Normalize the high-range float magnitude data to fit the 0-255 visualization range.
        magnitude_norm = cv.normalize(
            magnitude_float, 
            None, 
            0,            # alpha (min value for output)
            255,          # beta (max value for output)
            cv.NORM_MINMAX,
            cv.CV_8U      # Explicitly specify 8-bit unsigned output
        )
        
        magnitude_8bit = magnitude_norm
        meta["mag_meta"].update({"shape": magnitude_8bit.shape, "dtype": magnitude_8bit.dtype.name})

        return sobelx_8bit, sobely_8bit, magnitude_8bit, meta
    except Exception as exc:
        meta["error"] = f"Runtime error: {exc}"
        return None, None, None, meta
    

# =====================================================
# 3. EXECUTION AND I/O EXAMPLE (USING DEFAULTS)
# =====================================================

def ensure_output_directory(output_dir: str):
    """
    Checks if the specified directory exists, and creates it if it does not.
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"FATAL I/O Error creating directory {output_dir}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    # --- I/O Setup ---
    # NOTE: Set SRC_PATH to a valid image file path for execution
    SRC_PATH = "/home/user1/learning/Testing/NoiseReduction/Inputs/frame_000000.jpg" 
    OUT_DIR = "/home/user1/learning/Testing/NoiseReduction/Outputs"
    
    ensure_output_directory(OUT_DIR)
    
    # 1. Load Image
    try:
        if not os.path.exists(SRC_PATH):
            print(f"WARNING: Image file not found at '{SRC_PATH}'. Using dummy image.")
            image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        else:
            image = cv.imread(SRC_PATH, cv.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError("OpenCV could not decode the image file.")
            print(f"Image loaded successfully: {image.shape}")

    except Exception as e:
        print(f"Critical Error during file loading: {e}", file=sys.stderr)
        sys.exit(1)


    # 2. Execute Filters and Save Results
    
    # --- 2.1. Subtracting Low Pass (Using all default parameters) ---
    try:
        result_subtraction, meta = subtract_low_pass(
            image,
            gblur_ksize=(5, 5), # Example of overriding a default parameter
            sigma=2.5
            # Remaining parameters (offset, min_clip, max_clip) use defaults
        )
        if result_subtraction is not None:
            stacked_subtraction = np.hstack([image, result_subtraction])
            cv.imwrite(os.path.join(OUT_DIR, "1_HighPass_Subtraction.png"), stacked_subtraction)
            print(f"Processed and saved: HighPass Subtraction. Meta: {meta['shape']}, {meta['dtype']}")
        else:
            print(f"Error in HighPass Subtraction: {meta['error']}")
    except Exception as e:
        print(f"Execution Error in Subtraction: {e}", file=sys.stderr)


    # --- 2.2. Convolution with Kernel (Laplacian Sharpening) ---
    try:
        sharpening_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        result_convolution, meta = convolve_with_kernel(
            image,
            kernel=sharpening_kernel,
            ddepth=-1
        )
        if result_convolution is not None:
            stacked_convolution = np.hstack([image, result_convolution])
            cv.imwrite(os.path.join(OUT_DIR, "2_Convolution_Sharpening.png"), stacked_convolution)
            print(f"Processed and saved: Convolution Sharpening. Meta: {meta['shape']}, {meta['dtype']}")
        else:
            print(f"Error in Convolution Sharpening: {meta['error']}")
    except Exception as e:
        print(f"Execution Error in Convolution: {e}", file=sys.stderr)


    # --- 2.3. Laplacian Edge Detector ---
    try:
        result_laplacian, meta = apply_laplacian_detector(
            image,
            ksize=3,
            ddepth=cv.CV_64F
        )
        if result_laplacian is not None:
            cv.imwrite(os.path.join(OUT_DIR, "3a_Laplacian_Edges.png"), np.hstack([image, result_laplacian]))
            print(f"Processed and saved: Laplacian Edges. Meta: {meta['shape']}, {meta['dtype']}")
        else:
            print(f"Error in Laplacian Detector: {meta['error']}")
    except Exception as e:
        print(f"Execution Error in Laplacian Detector: {e}", file=sys.stderr)


    # --- 2.4. Sobel X and Y Edge Detectors ---
    try:
        result_sobelx, result_sobely, result_magnitude, meta = apply_sobel_xy_detectors(
            image,
            ksize=5, 
            ddepth=cv.CV_64F,
            dx_sobel_x=1,
            dy_sobel_x=0,
            dx_sobel_y=0,
            dy_sobel_y=1
        )
        if result_sobelx is not None and result_magnitude is not None:
                     
            # Save the new Sobel Magnitude image
            cv.imwrite(os.path.join(OUT_DIR, "3d_SobelMagnitude_Edges.png"), np.hstack([image, result_magnitude]))
            
            print(f"Processed and saved: Sobel X/Y/Magnitude. Meta: Mag={meta['mag_meta']['shape']}, {meta['mag_meta']['dtype']}")
        else:
            print(f"Error in Sobel Detectors: {meta['error']}")

    except Exception as e:
        print(f"Execution Error in Sobel Detectors: {e}", file=sys.stderr)

    
    print(f"\nAll processing finished. Results saved to the '{OUT_DIR}' directory.")