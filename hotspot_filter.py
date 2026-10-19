import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_hotspots(img: np.ndarray, sigma_factor=2.0):
    """
    Detects hotspots (bright outliers) in a grayscale image.

    Args:
        img (np.ndarray): 2D grayscale image.
        sigma_factor (float): Standard deviation multiplier for hotspot detection.

    Returns:
        np.ndarray: Binary mask of detected hotspots.
    """
    if len(img.shape) != 2:
        raise ValueError("Input must be a 2D grayscale image.")

    # Compute mean and standard deviation
    mean, std_dev = cv2.meanStdDev(img)
    mean = mean.item()  # Convert to scalar
    std_dev = std_dev.item()  # Convert to scalar
# Set automatic thresholds
    thresh_high = min(255, mean + sigma_factor * std_dev)  # Cap at 255
    thresh_low = max(0, mean - sigma_factor * std_dev)  # Cap at 0

    # print(f"Mean: {mean:.2f}, Std Dev: {std_dev:.2f}")
    # print(f"Auto-Thresholds → Low: {thresh_low:.2f}, High: {thresh_high:.2f}")

    # Detect bright hotspots
    _, mask_high = cv2.threshold(img, thresh_high, 255, cv2.THRESH_BINARY)

    # Detect dark hotspots
    _, mask_low = cv2.threshold(img, thresh_low, 255, cv2.THRESH_BINARY_INV)

    # Combine both masks
    mask = cv2.bitwise_or(mask_high, mask_low)
    # Dilate the mask to ensure all hotspot regions are covered
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    return mask_dilated.astype(np.uint8)

def remove_hotspots_auto(image: np.ndarray, mask: np.ndarray, method="inpaint"):
    """
    Removes hotspots from a grayscale image using inpainting or blurring.

    Args:
        image (np.ndarray): 2D grayscale image.
        mask (np.ndarray): Binary mask of detected hotspots.
        method (str): "inpaint" (default) or "blur".

    Returns:
        np.ndarray: Image with hotspots removed.
    """
    if image.shape != mask.shape:
        raise ValueError("Image and mask must have the same dimensions.")

    if method == "inpaint":
        # Convert grayscale to BGR for inpainting
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        result_bgr = cv2.inpaint(image_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        result_gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)  # Convert back to grayscale

    elif method == "blur":
        # Apply Gaussian blur only on hotspot areas
        blurred = cv2.GaussianBlur(image, (1,1), 0)
        result_gray = image.copy()
        result_gray[mask == 255] = blurred[mask == 255]

    else:
        raise ValueError("Invalid method. Use 'inpaint' or 'blur'.")

    return result_gray

def remove_hotspots_manual(img, threshold=220, method="inpaint"):
    """
    Detects and removes hotspots from an image.

    Args:
        img (np.ndarray): Input image (grayscale or BGR).
        threshold (int): Pixel intensity threshold to detect hotspots (default=220).
        method (str): Method to filter hotspots - "blur" or "inpaint".

    Returns:
        np.ndarray: Processed image with hotspots removed.
    """
    # Ensure image is grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Create a mask for hotspots (bright regions)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Process hotspots based on the chosen method
    img_copy = img.copy()
    if method == "blur":
        # Apply Gaussian blur only on hotspots
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        img_copy[mask == 255] = blurred[mask == 255]
    elif method == "inpaint":
        # Convert grayscale to BGR if necessary for inpainting
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img.copy()
        img_copy = cv2.inpaint(img_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return img_copy

