import cv2 as cv
import numpy as np
import sys
import os
from typing import Optional, Union

# Type hint alias for image arrays
Image = np.ndarray

def apply_scharr_operator(image: Image, blur_ksize: int = 3, threshold_val: float = 50.0) -> Image:
    """
    Applies the Scharr operator to find the gradient magnitude (edges) in an image.

    The Scharr operator is used to calculate the first derivatives (gradients)
    in the X and Y directions, which are then combined to find the magnitude
    of the edges. This implementation includes an initial blur and a final
    binary thresholding step. 

    Args:
        image: The input grayscale image (expected uint8).
        blur_ksize: The kernel size for the initial Gaussian blur.
        threshold_val: The intensity threshold for binary edge visualization.

    Returns:
        The original image horizontally stacked with the binary edge map (uint8).
    """
    # 1. Initial Blur to reduce noise sensitivity
    image_blur = cv.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

    # 2. Apply Scharr Operator
    # We use CV_32F (32-bit float) for the output depth to handle negative and large values
    scharr_x = cv.Scharr(image_blur, cv.CV_32F, 1, 0) # Gradient in X direction
    scharr_y = cv.Scharr(image_blur, cv.CV_32F, 0, 1) # Gradient in Y direction

    # 3. Calculate Gradient Magnitude
    # Magnitude M = sqrt(Gx^2 + Gy^2)
    # Using cv.magnitude is cleaner and potentially faster than cv.sqrt(cv.addWeighted(...))
    gradient_magnitude = cv.magnitude(scharr_x, scharr_y)

    # 4. Normalize and Convert to 8-bit image for visualization
    # Scale results to 0-255 range
    magnitude_norm = cv.normalize(gradient_magnitude, None, 0, 255, cv.NORM_MINMAX)
    magnitude_8bit = np.uint8(magnitude_norm)

    # 5. Apply Binary Thresholding for clear edge map
    # cv.threshold returns a tuple (ret_value, result_image). We extract the result_image.
    _, thermal_edges = cv.threshold(
        magnitude_8bit,
        threshold_val,
        255,
        cv.THRESH_BINARY
    )

    # 6. Stack and return
    # The original image and the edge map must be the same data type (uint8) for hstack
    scharr_result = np.hstack([image, thermal_edges])
    cv.imwrite("StructureModule/Outputs/ScharrResult.png", scharr_result)

    return scharr_result


def main(image_path: str):
    """
    Main function to load an image, apply the Scharr edge detection operator,
    and display the results.

    Args:
        image_path: The file path to the input image.
    """
    # 1. Input Validation and Loading
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        sys.exit(1)

    # Load the image in grayscale (0 flag)
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not load image from '{image_path}'. Check file integrity.")
        sys.exit(1)

    print(f"Image loaded successfully: {image.shape}")

    # 2. Apply the Scharr filter
    # Edge map is stacked with the original image inside the function
    scharr_stacked_image = apply_scharr_operator(
        image=image,
        blur_ksize=3,
        threshold_val=50.0
    )

    # 3. Display Result
    cv.imshow("Original | Scharr Edge Map", scharr_stacked_image)

    # 4. Cleanup
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("Program finished.")


if __name__ == "__main__":
    # NOTE: Update this path to point to your actual image file.
    # Placeholder path for industrial standard structure.
    # Example: IMAGE_FILE_PATH = "C:/path/to/my/image.png"
    IMAGE_FILE_PATH = "StructureModule/Inputs/Input.jpg"
    
    # Execute the main function
    main(IMAGE_FILE_PATH)