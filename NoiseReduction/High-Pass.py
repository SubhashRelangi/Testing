import cv2 as cv
import numpy as np
import sys
import os
from typing import Optional, Union, Tuple

# Type hint alias for image arrays
Image = np.ndarray

def subtracting_low_pass(image: Image) -> Image:
    """
    Applies a high-pass filter by subtracting a blurred (low-pass) version of the image
    and returns the original image stacked horizontally with the high-pass result.

    This method emphasizes fine details and edges by removing the smooth, low-frequency
    components of the image. The constant 127 is added to re-center the resulting
    high-pass image around a medium gray, making it visible.

    Args:
        image: The input image (expected to be a grayscale NumPy array, uint8).

    Returns:
        The original image horizontally stacked with the high-pass filtered image (uint8).
    """
    # Convert to float32 for precise mathematical operations
    image_float = image.astype(np.float32)

    # Apply Gaussian blur (Low-pass filter). 
    blurred_img = cv.GaussianBlur(image_float, (0, 0), 3)
    # 

    # High-pass = Original - Low-pass (Blur) + Offset
    high_pass_img = image_float - blurred_img + 127

    # Clip values to the valid 0-255 range and convert back to uint8
    high_pass_result = np.clip(high_pass_img, 0, 255).astype(np.uint8)

    # Stack original and result horizontally
    stacked_image = np.hstack([image, high_pass_result])
    cv.imwrite("Outputs/subtracting_low_pass.png", stacked_image)
    return stacked_image


def convolve_with_kernel(image: Image, kernel: Image) -> Image:
    """
    Applies a convolution using a predefined kernel (e.g., a high-pass kernel)
    and returns the original image stacked horizontally with the convoluted result.

    This function uses cv2.filter2D for 2D convolution.

    Args:
        image: The input image (expected to be a grayscale NumPy array, uint8).
        kernel: The convolution kernel (e.g., a 3x3 array for high-pass filtering).

    Returns:
        The original image horizontally stacked with the filtered image (uint8).
    """
    # 
    # The ddepth=-1 means the output image will have the same depth (type) as the source.
    high_pass_result = cv.filter2D(image, ddepth=-1, kernel=kernel)

    # Stack original and result horizontally
    stacked_image = np.hstack([image, high_pass_result])
    cv.imwrite("Outputs/convolve_with_kernel.png", stacked_image)
    return stacked_image


def apply_edge_detectors(image: Image, ksize: int = 3) -> Tuple[Image, Image, Image]:
    """
    Applies common gradient-based edge detection operators (Sobel and Laplacian)
    and returns a tuple of horizontally stacked images:
    1. Original vs. Laplacian result.
    2. Original vs. Sobel X result.
    3. Original vs. Sobel Y result.

    Args:
        image: The input image (expected to be a grayscale NumPy array, uint8).
        ksize: The aperture size for the Sobel and Laplacian operators.

    Returns:
        A tuple containing three stacked images, all uint8.
        (Original|Laplacian, Original|SobelX, Original|SobelY)
    """
    # 1. Laplacian Operator
    # cv.CV_64F is used to prevent overflow
    lap = cv.Laplacian(image, cv.CV_64F, ksize=ksize)
    laplacian_8bit = np.uint8(np.absolute(lap))
    # 
    
    # 2. Sobel Operator (X-direction)
    sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=ksize)
    sobelx_8bit = np.uint8(np.absolute(sobelx))

    # 3. Sobel Operator (Y-direction)
    sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=ksize)
    sobely_8bit = np.uint8(np.absolute(sobely))
    # 

    # Stack results with the original image for visualization
    laplacian_stacked = np.hstack([image, laplacian_8bit])
    sobelx_stacked = np.hstack([image, sobelx_8bit])
    sobely_stacked = np.hstack([image, sobely_8bit])
    cv.imwrite("Outputs/laplacian.png", laplacian_stacked)
    cv.imwrite("Outputs/sobelx.png", sobelx_stacked)
    cv.imwrite("Outputs/sobely.png", sobely_stacked)


    return laplacian_stacked, sobelx_stacked, sobely_stacked


def main(image_path: str):
    """
    Main function to load an image, apply various high-pass filtering techniques,
    and display the results, showing the original and output side-by-side.

    Args:
        image_path: The file path to the input image.
    """
    # Input validation and image loading
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        sys.exit(1)

    thermal_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if thermal_image is None:
        print(f"Error: Could not load image from '{image_path}'. Check file integrity.")
        sys.exit(1)

    print(f"Image loaded successfully: {thermal_image.shape}")

    # --- 1. Subtracting a Blurred Image (High-Pass by Subtraction) ---
    stacked_subtraction = subtracting_low_pass(image=thermal_image)

    # --- 2. Using Predefined High Pass Kernel (Convolution) ---
    high_pass_kernel = np.array([[-1, -1, -1],
                                 [-1,  8, -1],
                                 [-1, -1, -1]], dtype=np.float32)
    stacked_convolution = convolve_with_kernel(
        image=thermal_image,
        kernel=high_pass_kernel
    )

    # --- 3. Using Sobel and Laplacian Operators (Edge Detection) ---
    # The function now returns stacks, not just the results
    stacked_laplacian, stacked_sobelx, stacked_sobely = apply_edge_detectors(
        image=thermal_image,
        ksize=3
    )

    # --- Display Results (All are already stacked with the original) ---

    cv.imshow("1. Original | High Pass (Subtraction)", stacked_subtraction)
    cv.imshow("2. Original | High Pass (Convolution Kernel)", stacked_convolution)
    cv.imshow("3. Original | Laplacian Edge Detection", stacked_laplacian)
    cv.imshow("4. Original | Sobel X Edge Detection", stacked_sobelx)
    cv.imshow("5. Original | Sobel Y Edge Detection", stacked_sobely)

    # Wait for a key press and close all windows
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("Program finished.")


if __name__ == "__main__":
    # Define the image path here. For robustness, you could take this as a command-line argument.
    IMAGE_FILE_PATH = "/home/user1/learning/Testing/NoiseReduction/Inputs/frame_000000.jpg"
    # Execute the main function
    main(IMAGE_FILE_PATH)