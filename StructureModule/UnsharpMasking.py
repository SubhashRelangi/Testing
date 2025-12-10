import cv2 as cv
import numpy as np
import sys
import os
from typing import Optional, Union, Tuple

# Type hint alias for image arrays
Image = np.ndarray

# Define the output directory at the module level for consistency
OUTPUT_DIR = "Sharpening/Outputs"


def ensure_output_directory(output_dir: str):
    """
    Checks if the specified directory exists, and creates it if it does not.

    Args:
        output_dir: The path to the directory to create.
    """
    if not os.path.exists(output_dir):
        try:
            # Create directory if it doesn't exist; exist_ok=True prevents error if already exists
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating directory {output_dir}: {e}")
            sys.exit(1)


def apply_unsharp_masking(
    image: Image,
    k_size: Tuple[int, int] = (5, 5),
    sigma: float = 0.0,
    alpha: float = 3.0
) -> Image:
    """
    Applies the Unsharp Masking (USM) technique to sharpen an image.

    The process enhances high-frequency details by:
    1. Creating a blurred (low-pass) version of the image.
    2. Subtracting the blurred image from the original to get a high-frequency mask (detail map).
    3. Adding the scaled mask back to the original image to enhance details. 

    Args:
        image: The input grayscale image (expected uint8).
        k_size: The kernel size for the Gaussian blur (low-pass filter). Must be odd and positive.
        sigma: The standard deviation (sigma) for the Gaussian blur. (0.0 means calculated from k_size).
        alpha: The sharpening strength (or 'Amount'). A higher value increases sharpening.

    Returns:
        The original image horizontally stacked with the mask and the sharpened result (uint8).
    """
    # 1. Convert to floating-point BEFORE arithmetic operations
    image_float = image.astype(np.float32)

    # 2. Create the Blurred Image (Low-Pass Filter)
    blurred_Image_float = cv.GaussianBlur(image_float, ksize=k_size, sigmaX=sigma, sigmaY=sigma)

    # 3. Create the Mask (High-Pass Filter/Detail Map)
    # Mask = Original - Blurred
    Mask_float = image_float - blurred_Image_float

    # 4. Sharpen the Image (Sharpened = Original + Mask * Alpha)
    sharpedImage_float = image_float + Mask_float * alpha

    # 5. Clip and Convert to uint8
    # Clip results to the valid range [0, 255]
    sharpedImage_float = np.clip(sharpedImage_float, 0, 255)
    sharpedImage_uint8 = sharpedImage_float.astype(np.uint8)

    # 6. Stack results for visualization: Original | Mask | Sharpened
    # NOTE: Add 128 to the float mask and clip for visible stacking (centers zero-detail at mid-gray)
    mask_for_display = np.clip(Mask_float + 128, 0, 255).astype(np.uint8)
    
    output_stacked = np.hstack([image, mask_for_display, sharpedImage_uint8])

    return output_stacked


def main(image_path: str):
    """
    Main function to load an image, apply Unsharp Masking with specific parameters,
    and display/save the visualization results.

    Args:
        image_path: The file path to the input image.
    """
    # 1. Setup and Validation
    ensure_output_directory(OUTPUT_DIR)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        sys.exit(1)

    # Load the image in grayscale (0 flag)
    image_uint8 = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if image_uint8 is None:
        print(f"Error: Could not load image from '{image_path}'. Check file integrity.")
        sys.exit(1)

    print(f"Image loaded successfully: {image_uint8.shape}")

    # 2. Configuration & Application (Using your chosen parameters)
    K_SIZE = (5, 5)
    SIGMA = 0.0
    ALPHA = 3.0
    
    output_stacked = apply_unsharp_masking(
        image=image_uint8,
        k_size=K_SIZE,
        sigma=SIGMA,
        alpha=ALPHA
    )

    # 3. Display and Save Results
    cv.imshow("Original | High-Frequency Mask | Sharpened Image", output_stacked)
    
    output_filename = os.path.join(OUTPUT_DIR, "UnsharpMaskingResult.png")
    cv.imwrite(output_filename, output_stacked)
    print(f"Result saved to {output_filename}")

    # 4. Cleanup
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("Program finished.")


if __name__ == "__main__":
    # NOTE: Update this path to point to your actual image file.
    IMAGE_FILE_PATH = "StructureModule/Inputs/Input.jpg"
    
    # Execute the main function
    main(IMAGE_FILE_PATH)