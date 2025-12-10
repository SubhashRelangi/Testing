import cv2 as cv
import numpy as np
import sys
import os
from typing import Optional, Union, Tuple

# Type hint alias for image arrays
Image = np.ndarray

def subtracting_low_pass(image: Image) -> Image:
    """
    Applies a high-pass filter by subtracting a blurred (low-pass) version of the image.

    This method emphasizes fine details and edges by removing the smooth, low-frequency
    components of the image. The constant 127 is added to re-center the resulting
    high-pass image around a medium gray, making it visible.

    Args:
        image: The input image (expected to be a grayscale NumPy array, uint8).

    Returns:
        The high-pass filtered image (uint8), with details enhanced.
    """
    # Convert to float32 for precise mathematical operations
    image_float = image.astype(np.float32)

    # Apply Gaussian blur (Low-pass filter). The kernel size (0, 0) and sigma=3
    # means the kernel size is calculated automatically based on sigma.
    # 
    blurred_img = cv.GaussianBlur(image_float, (0, 0), 3)

    # High-pass = Original - Low-pass (Blur) + Offset
    # The offset (127) shifts the mean of the result to the middle of the 0-255 range.
    high_pass_img = image_float - blurred_img + 127

    # Clip values to the valid 0-255 range and convert back to uint8
    high_pass_img = np.clip(high_pass_img, 0, 255).astype(np.uint8)

    return high_pass_img


def convolve_with_kernel(image: Image, kernel: Image) -> Image:
    """
    Applies a convolution using a predefined kernel (e.g., a high-pass kernel).

    This function uses cv2.filter2D for 2D convolution, which is the direct
    implementation of filtering with a specific kernel.

    Args:
        image: The input image (expected to be a grayscale NumPy array, uint8).
        kernel: The convolution kernel (e.g., a 3x3 array for high-pass filtering).

    Returns:
        The filtered image (uint8), the result of the convolution.
    """
    # The ddepth=-1 means the output image will have the same depth (type) as the source.
    # For a uint8 input, the result might need conversion if intermediate results
    # exceed 255/fall below 0, but for standard kernels, this often works.
    # 
    high_pass_image = cv.filter2D(image, ddepth=-1, kernel=kernel)
    return high_pass_image


def apply_edge_detectors(image: Image, ksize: int = 3) -> Tuple[Image, Image, Image]:
    """
    Applies common gradient-based edge detection operators (Sobel and Laplacian).

    Sobel finds the gradient magnitude in X and Y directions, while Laplacian
    finds the second-order derivative, useful for identifying sharp changes.

    Args:
        image: The input image (expected to be a grayscale NumPy array, uint8).
        ksize: The aperture size for the Sobel operator (e.g., 3, 5, 7).

    Returns:
        A tuple containing: (Laplacian image, Sobel X image, Sobel Y image), all uint8.
    """
    # 1. Laplacian Operator
    # cv.CV_64F is used to prevent overflow, as derivatives can be negative.
    lap = cv.Laplacian(image, cv.CV_64F, ksize=ksize)
    # The absolute value and conversion to uint8 are necessary for visualization.
    # 
    laplacian_8bit = np.uint8(np.absolute(lap))

    # 2. Sobel Operator (X-direction)
    # dx=1, dy=0 specifies the derivative order in X and Y.
    sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=ksize)
    sobelx_8bit = np.uint8(np.absolute(sobelx))

    # 3. Sobel Operator (Y-direction)
    # dx=0, dy=1 specifies the derivative order in X and Y.
    sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=ksize)
    sobely_8bit = np.uint8(np.absolute(sobely))
    # 

    return laplacian_8bit, sobelx_8bit, sobely_8bit


def main(image_path: str):
    """
    Main function to load an image, apply various high-pass filtering techniques,
    and display the results.

    Args:
        image_path: The file path to the input image.
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        sys.exit(1)

    # Load the image in grayscale (0 flag)
    thermal_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if thermal_image is None:
        print(f"Error: Could not load image from '{image_path}'. Check file integrity.")
        sys.exit(1)

    print(f"Image loaded successfully: {thermal_image.shape}")

    # --- 1. Subtracting a Blurred Image (High-Pass by Subtraction) ---
    high_pass_subtraction_img = subtracting_low_pass(image=thermal_image)

    # --- 2. Using Predefined High Pass Kernel (Convolution) ---
    # Standard high-pass/sharpening kernel (emphasizes central pixel relative to neighbors)
    high_pass_kernel = np.array([[-1, -1, -1],
                                 [-1,  8, -1],
                                 [-1, -1, -1]], dtype=np.float32)
    high_pass_convolution_img = convolve_with_kernel(
        image=thermal_image,
        kernel=high_pass_kernel
    )

    # --- 3. Using Sobel and Laplacian Operators (Edge Detection) ---
    laplacian_img, sobelx_img, sobely_img = apply_edge_detectors(
        image=thermal_image,
        ksize=3
    )

    # --- Display Results ---

    # Display images from method 1 and 2
    cv.imshow("1. Original Thermal", thermal_image)
    cv.imshow("2. High Pass Filtered (Subtraction)", high_pass_subtraction_img)
    cv.imshow("3. High Pass Filtered (Convolution)", high_pass_convolution_img)

    # Display images from method 3
    sobel_combined = np.hstack([sobelx_img, sobely_img])
    laplacian_comparison = np.hstack([thermal_image, laplacian_img])

    cv.imshow("4. Sobel X and Y Edges", sobel_combined)
    cv.imshow("5. Original vs. Laplacian Edges", laplacian_comparison)

    # Wait for a key press and close all windows
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("Program finished.")


if __name__ == "__main__":
    # Define the image path here. For robustness, you could take this as a command-line argument.
    IMAGE_FILE_PATH = "VideoFrames/frame_000000.jpg"
    # Execute the main function
    main(IMAGE_FILE_PATH)