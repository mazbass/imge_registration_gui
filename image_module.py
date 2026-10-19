import os
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, exposure
from skimage.transform import rescale
plt.ion()

def _to_float01(im):
    im = im.astype(np.float32)
    im -= im.min()
    vmax = im.max()
    if vmax > 0:
        im /= vmax
    return im

def to_gray01(img):
    """Convert to float32 grayscale in [0,1]."""
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    # Normalize to [0,1] robustly (2-98 percentile)
    p2, p98 = np.percentile(img, (2, 98))
    if p98 > p2:
        img = np.clip((img - p2) / (p98 - p2), 0, 1)
    else:
        # fallback
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def equalize_local(img, clip_limit=0.01, tile_grid_size=(8,8)):
    """Adaptive histogram equalization (CLAHE) to improve contrast."""
    return exposure.equalize_adapthist(img, clip_limit=clip_limit, nbins=256, kernel_size=tile_grid_size)

def color_fuse(fixed, moving, mode="R|GB", apply_clahe=False):
    """
    Color-fuse two grayscale images.
    mode options:
      - "R|GB": fixed->R, moving->G,B  (classic fusion; misalignment shows as magenta/cyan)
      - "RG|B": fixed->R,G, moving->B
      - "G|RB": fixed->G, moving->R,B
      - "C|M":  fixed->C (G+B), moving->M (R+B) — produces a slightly different visual
    """
    f = to_gray01(fixed)
    m = to_gray01(moving)
    if apply_clahe:
        f = equalize_local(f)
        m = equalize_local(m)
    H, W = f.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    if mode == "R|GB":
        rgb[..., 0] = f
        rgb[..., 1] = m
        rgb[..., 2] = m
    elif mode == "RG|B":
        rgb[..., 0] = f
        rgb[..., 1] = f
        rgb[..., 2] = m
    elif mode == "G|RB":
        rgb[..., 0] = m
        rgb[..., 1] = f
        rgb[..., 2] = m
    elif mode == "C|M":
        # fixed to Cyan (G+B), moving to Magenta (R+B)
        rgb[..., 0] = m
        rgb[..., 1] = f
        rgb[..., 2] = 0.5 * (f + m)
    else:
        raise ValueError("Unknown mode")
    return np.clip(rgb, 0, 1)


def load_image(path, dataset_key=None, channel=None, as_gray=True):
    """
    Load either:
      • A single channel from an HDF5 dataset (shape C×H×W), or
      • A 2D image file (PNG/JPG/TIFF/etc).

    Returns:
        img (np.ndarray): 2D image array.
        element (str|None): channel name (HDF5 only) or None.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.h5', '.hdf5'):
        if dataset_key is None or channel is None:
            raise ValueError("HDF5 inputs require dataset_key AND channel")
        with h5py.File(path, 'r') as f:
            arr = f[dataset_key][...]              # (C,H,W) or (C,H,W,3)
            names = f["MAPS/channel_names"][...]   # list of bytes
        channel_names = [n.decode() for n in names]
        img = arr[channel]
        element = channel_names[channel]

        # C, H, W = arr.shape
        # fig, axes = plt.subplots(1, C, figsize=(4*C, 4))
        # if C == 1:
        #     axes = [axes]  # make it iterable even when there's only one channel

        # for i, ax in enumerate(axes):
        #     im = arr[i]
        #     ax.imshow(im, cmap='gray')
        #     ax.set_title(channel_names[i])
        #     ax.axis('off')

        # plt.tight_layout()
        # plt.show()

        if as_gray and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    else:
        # image file
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image at {path!r}")
        element = None

        if as_gray and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, element

def rotate_image(
    img: np.ndarray,
    angle: float,
    scale: float = 1.0,
    *,
    border_value=None,                 # constant fill; None => 0 (per channel)
    interpolation: int = cv2.INTER_LINEAR,
    expand: bool = False,              # True => enlarge canvas to fit rotated image
    return_valid_mask: bool = False
):
    """
    Rotate an image by `angle` degrees about its center.

    - No wrap/reflect: uses BORDER_CONSTANT with `border_value` (default 0).
    - If expand=False (default), output size == input size (cropping possible).
    - If expand=True, enlarges canvas so the rotated image fits.
    - If return_valid_mask=True, also returns a boolean mask of valid (non-padded) pixels.

    Returns:
        rotated  [and valid_mask if return_valid_mask=True]
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Default border value (per channel if needed)
    if border_value is None:
        if img.ndim == 2:
            border_value = 0
        else:
            # one value per channel
            ch = img.shape[2]
            border_value = tuple([0] * ch) if not np.issubdtype(img.dtype, np.floating) \
                           else tuple([0.0] * ch)

    # Base rotation matrix (about original center)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Determine output size
    if expand:
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        # shift so the image is centered in the new canvas
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        out_size = (new_w, new_h)
    else:
        out_size = (w, h)

    # Rotate with constant border (no wrap/reflect)
    rotated = cv2.warpAffine(
        img,
        M,
        out_size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

    if not return_valid_mask:
        return rotated

    # Build valid-pixel mask by rotating a ones mask with nearest + constant 0
    mask_src = np.ones((h, w), dtype=np.uint8) * 255
    valid_u8 = cv2.warpAffine(
        mask_src,
        M,
        out_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    valid_mask = valid_u8 > 0
    return rotated, valid_mask

def snr(img):
    mean_signal = np.mean(img)
    std_noise = np.std(img)
    snr_ratio = 20 * np.log10(mean_signal / std_noise)
    return snr_ratio

def preprocess_with_canny(img,
                          low_threshold=0,
                          high_threshold=150,
                          gaussian_blur_ksize=(5,5),
                          gaussian_blur_sigma=1.5):
    """
    Preprocess image by:
      1) Normalizing to uint8
      2) Gaussian‐blurring to reduce noise
      3) Running Canny edge detector
      4) (Optionally) Dilating edges to thicken

    Returns:
        edge_img (np.ndarray): binary edge map (0 or 255)
    """
    # 1) normalize to [0,255] uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)

    # 2) smooth
    img_blur = cv2.GaussianBlur(img, gaussian_blur_ksize, gaussian_blur_sigma)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    mag = np.hypot(gx, gy)
    med = np.median(mag)
    # low_threshold  = 1.66 * med
    # high_threshold = 3.33 * med

    # 3) Canny edges
    edges = cv2.Canny(img_blur, low_threshold, high_threshold, L2gradient=True)

    # 4) (optional) thicken edges by dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


def center_crop_2d(A: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    H, W = A.shape[:2]
    if crop_h > H or crop_w > W:
        raise ValueError("crop size must be ≤ array size")
    start_i = (H - crop_h) // 2
    start_j = (W - crop_w) // 2
    return A[start_i:start_i + crop_h,
             start_j:start_j + crop_w]
def overlay_images(ref, aligned, alpha=0.5, element=None):
    """
    Display ref and aligned on the same axes in true grayscale,
    with the aligned image semi-transparent on top.
    """
    # ensure uint8 0–255
    ref8    = np.clip(ref,    0, 255).astype(np.uint8)
    aligned8 = np.clip(aligned, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(6,6))
    # show reference first
    im_ref = ax.imshow(ref8,    cmap='gray', vmin=0, vmax=255)
    # overlay aligned with alpha
    ax.imshow(aligned8, cmap='gray', vmin=0, vmax=255, alpha=alpha)

    ax.axis('off')
    title = f"{element}: Overlay" if element else "Overlay"
    ax.set_title(title)

    # one colorbar for the reference intensity scale
    cbar = fig.colorbar(im_ref, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pixel intensity (0–255)")

    plt.tight_layout()
    plt.show()


def overlay_images1(ref, aligned, alpha=0.5, element=None):
    """Blend to see where the small patch landed."""
    blended = cv2.addWeighted(ref.astype(np.float32), alpha,
                              aligned.astype(np.float32), 1-alpha, 0)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    plt.figure(figsize=(6,6))
    plt.imshow(blended, cmap="gray")
    title = f"{element}: Overlay" if element else "Overlay"
    plt.title(title)
    plt.axis("off")
    plt.show()


def find_angle(img1, img2):
    """Use log-polar transform centered at image center."""
    center = (img1.shape[1] // 2, img1.shape[0] // 2)
    M = img1.shape[0] / np.log(img1.shape[0])  # Magnitude scale
    # Convert to float32
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    log_polar1 = cv2.logPolar(img1_f, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    log_polar2 = cv2.logPolar(img2_f, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    # import pdb; pdb.set_trace()

    # Phase correlation gives rotation angle offset
    (shift, _) = cv2.phaseCorrelate(log_polar1, log_polar2)

    # Compensate for wrap-around (angle is in degrees)
    angle = shift[1]/img1.shape[1]*360  # Reverse rotation direction
    print(f"Estimated rotation angle: {angle:.2f} degrees")

    # Now apply rotation to img2 to align it with img1
    (h, w) = img2.shape
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(img2, rot_matrix, (w, h))

    # # Display results
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1); plt.imshow(img1, cmap='gray'); plt.title("Reference")
    # plt.subplot(1, 3, 2); plt.imshow(img2, cmap='gray'); plt.title("Rotated Input")
    # plt.subplot(1, 3, 3); plt.imshow(aligned, cmap='gray'); plt.title("Aligned (Rotation Only)")
    # plt.tight_layout(); plt.show()

    return angle, rot_matrix, aligned

def _to_uint8(img_float01):
    """float32 in [0,1] -> uint8 [0,255]"""
    return (np.clip(img_float01, 0, 1) * 255.0).astype(np.uint8)

def _to_float01(img_uint8):
    """uint8 [0,255] -> float32 in [0,1]"""
    return (img_uint8.astype(np.float32) / 255.0)

def _canny_auto_u8(im_u8, sigma=0.33, gain=1.0):
    m = float(np.median(im_u8))
    low  = int(max(0,   gain * (1.0 - sigma) * m))
    high = int(min(255, gain * (1.0 + sigma) * m))
    return cv2.Canny(im_u8, low, high, apertureSize=3, L2gradient=True)

def edge_distance_map(im_float01, sigma=0.33, dilate_px=1):
    im_u8 = (np.clip(im_float01,0,1)*255).astype(np.uint8)
    edges = _canny_auto_u8(im_u8, sigma=sigma)
    if dilate_px and dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
        edges = cv2.dilate(edges, k, iterations=1)
    inv = 255 - edges
    dt  = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
    if dt.max() <= 1e-6:
        return np.zeros_like(dt, dtype=np.float32)
    # "edges-bright": near edges -> high
    return (1.0 - dt / (dt.max() + 1e-6)).astype(np.float32)
def fourier_ring_correlation(img1, img2):
    assert img1.shape == img2.shape and img1.ndim == 2
    I1 = np.fft.fftshift(np.fft.fft2(img1))
    I2 = np.fft.fftshift(np.fft.fft2(img2))
    ny, nx = img1.shape
    cy, cx = (np.array(img1.shape) - 1) / 2.0
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((y - cy)**2 + (x - cx)**2)
    rbin = np.rint(r).astype(np.int32)
    rmax = rbin.max()
    num = np.bincount(rbin.ravel(), weights=(I1*np.conj(I2)).ravel().real, minlength=rmax+1)
    den1 = np.bincount(rbin.ravel(), weights=(np.abs(I1)**2).ravel(), minlength=rmax+1)
    den2 = np.bincount(rbin.ravel(), weights=(np.abs(I2)**2).ravel(), minlength=rmax+1)
    valid = (den1 > 0) & (den2 > 0)
    frc = np.zeros_like(num, dtype=np.float64)
    frc[valid] = num[valid] / np.sqrt(den1[valid] * den2[valid])
    freq = np.arange(len(frc)) / np.mean(img1.shape)  # cycles/pixel
    cutoff = 0.143
    idx = np.where(frc < cutoff)[0][0]
    freq_cut = freq[idx]
    return freq_cut
