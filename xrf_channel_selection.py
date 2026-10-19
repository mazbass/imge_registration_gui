
from __future__ import annotations

from dataclasses import dataclass
import ast
import h5py
from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Any

import numpy as np
from register_multiscale import preprocess
from pca_xrf_channels_sc import preprocess_f32 as _preprocess_f32
from image_module import rotate_image

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "This script requires OpenCV (cv2). Install via: pip install opencv-python"
    ) from e

try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "This script requires pandas. Install via: pip install pandas"
    ) from e


@dataclass(frozen=True)
class PSRResult:
    psr: float
    peak_value: float
    peak_loc_xy: Tuple[int, int]          # (x, y) in correlation-map coordinates
    second_peak_value: float
    second_peak_loc_xy: Tuple[int, int]
    sidelobe_mean: float
    sidelobe_std: float


def robust_normalize(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    p_low: float = 1.0,
    p_high: float = 99.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Robustly normalize img to approximately [0, 1] based on percentiles.
    mask: boolean array same shape as img; True means valid pixels.
    """
    x = img.astype(np.float32, copy=False)
    if mask is not None:
        m = mask.astype(bool)
        vals = x[m]
        if vals.size == 0:
            return np.zeros_like(x, dtype=np.float32)
    else:
        vals = x.reshape(-1)

    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < eps:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return y.astype(np.float32)


def gradient_magnitude(
    img: np.ndarray,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """
    Compute gradient magnitude using Sobel derivatives.
    blur_sigma: Gaussian blur sigma in pixels; set 0 to disable blur.
    """
    x = img.astype(np.float32, copy=False)
    if blur_sigma and blur_sigma > 0:
        # Kernel size 0 lets OpenCV compute it from sigma.
        x = cv2.GaussianBlur(x, ksize=(0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))

    gx = cv2.Sobel(x, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gy = cv2.Sobel(x, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    g = cv2.magnitude(gx, gy)
    return g.astype(np.float32)


def downsample(
    img: np.ndarray,
    factor: float,
) -> np.ndarray:
    """
    Downsample image by 'factor' using area interpolation.
    factor > 1 means reduce size by that factor.
    """
    if factor <= 0:
        raise ValueError("downsample factor must be > 0")

    if np.isclose(factor, 1.0):
        return img.astype(np.float32, copy=False)

    h, w = img.shape[:2]
    new_w = max(1, int(round(w / factor)))
    new_h = max(1, int(round(h / factor)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA).astype(np.float32)


# def rotate_image_keep_size(
#     img: np.ndarray,
#     angle_deg: float,
# ) -> np.ndarray:
#     """
#     Rotate image around center, keeping the same output size.
#     Uses border replicate to avoid introducing zeros at edges.
#     """
#     if np.isclose(angle_deg, 0.0):
#         return img.astype(np.float32, copy=False)
#
#     h, w = img.shape[:2]
#     center = (w * 0.5, h * 0.5)
#     M = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
#     out = cv2.warpAffine(
#         img.astype(np.float32, copy=False),
#         M,
#         dsize=(w, h),
#         flags=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_REPLICATE,
#     )
#     return out.astype(np.float32)


def compute_psr(
    corr_map: np.ndarray,
    peak_exclusion_radius: int = 10,
    eps: float = 1e-12,
) -> PSRResult:
    """
    Compute Peak-to-Sidelobe Ratio (PSR) for a correlation map.

    PSR = (peak - mean(sidelobe)) / std(sidelobe)

    sidelobe excludes a (2r+1)x(2r+1) window around the peak.
    """
    if corr_map.ndim != 2:
        raise ValueError("corr_map must be 2D")

    C = corr_map.astype(np.float32, copy=False)
    if C.size == 0 or not np.isfinite(C).any():
        return PSRResult(
            psr=float("-inf"),
            peak_value=float("nan"),
            peak_loc_xy=(0, 0),
            second_peak_value=float("nan"),
            second_peak_loc_xy=(0, 0),
            sidelobe_mean=float("nan"),
            sidelobe_std=float("nan"),
        )

    # Peak
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(C)
    peak_value = float(max_val)
    peak_x, peak_y = int(max_loc[0]), int(max_loc[1])

    # Build sidelobe mask
    h, w = C.shape
    mask = np.ones((h, w), dtype=bool)

    r = int(max(0, peak_exclusion_radius))
    x0 = max(0, peak_x - r)
    x1 = min(w, peak_x + r + 1)
    y0 = max(0, peak_y - r)
    y1 = min(h, peak_y + r + 1)
    mask[y0:y1, x0:x1] = False

    sidelobe_vals = C[mask]
    # If sidelobe is too small (e.g., tiny map), degrade gracefully.
    if sidelobe_vals.size < 10:
        sidelobe_mean = float(np.mean(C))
        sidelobe_std = float(np.std(C) + eps)
    else:
        sidelobe_mean = float(np.mean(sidelobe_vals))
        sidelobe_std = float(np.std(sidelobe_vals) + eps)

    psr = (peak_value - sidelobe_mean) / sidelobe_std

    # Second-best peak outside exclusion window
    C2 = C.copy()
    C2[~mask] = -np.inf
    _, second_val, _, second_loc = cv2.minMaxLoc(C2)
    second_peak_value = float(second_val)
    second_peak_loc_xy = (int(second_loc[0]), int(second_loc[1]))

    return PSRResult(
        psr=float(psr),
        peak_value=peak_value,
        peak_loc_xy=(peak_x, peak_y),
        second_peak_value=second_peak_value,
        second_peak_loc_xy=second_peak_loc_xy,
        sidelobe_mean=sidelobe_mean,
        sidelobe_std=sidelobe_std,
    )


def match_template_corr_normed(
    image: np.ndarray,
    templ: np.ndarray,
) -> np.ndarray:
    """
    Normalized correlation template match.
    Uses TM_CCORR_NORMED. (Stable for non-negative structure images like gradients.)
    """
    I = image.astype(np.float32, copy=False)
    T = templ.astype(np.float32, copy=False)

    th, tw = T.shape[:2]
    ih, iw = I.shape[:2]
    if th > ih or tw > iw:
        raise ValueError(f"Template {T.shape} is larger than image {I.shape}.")

    # Guard: constant template => correlation map can become unstable/meaningless
    if float(np.std(T)) < 1e-6:
        # Return a flat map so PSR will be low.
        out_h = ih - th + 1
        out_w = iw - tw + 1
        return np.zeros((out_h, out_w), dtype=np.float32)

    C = cv2.matchTemplate(I, T, method=cv2.TM_CCORR_NORMED)
    return C.astype(np.float32)


def rank_xrf_channels_by_peakiness(
    optical_img: np.ndarray,
    xrf_stack: np.ndarray,
    *,
    channel_names: Optional[Sequence[str]] = None,
    downsample_factor: float = 10.0,
    optical_blur_sigma: float = 1.0,
    xrf_blur_sigma: float = 1.0,
    peak_exclusion_radius: int = 10,
    angles_deg: Sequence[float] = (0.0,),
    # Optional: restrict search to ROI in optical to reduce false peaks and speed up.
    # roi = (x, y, w, h) in optical pixel coords.
    search_roi: Optional[Tuple[int, int, int, int]] = None,
) -> pd.DataFrame:
    """
    Rank XRF channels by PSR peakiness when matching each channel's structure image
    against the optical structure image.

    Inputs:
      optical_img: 2D overview image (H_o, W_o)
      xrf_stack:   3D (K, H_x, W_x) or 4D (K, H_x, W_x, 1) with channels as patches.

    Output:
      DataFrame with best angle per channel and PSR, peaks, etc.
    """
    if optical_img.ndim != 2:
        raise ValueError("optical_img must be a 2D array (grayscale).")
    if xrf_stack.ndim != 3:
        raise ValueError("xrf_stack must be shape (K, H, W).")

    K = xrf_stack.shape[0]
    if channel_names is None:
        channel_names = [f"ch_{k:03d}" for k in range(K)]
    if len(channel_names) != K:
        raise ValueError("channel_names length must match number of channels in xrf_stack.")

    # Prepare optical structure image
    opt_norm = robust_normalize(optical_img)
    opt_grad = gradient_magnitude(opt_norm, blur_sigma=optical_blur_sigma)

    # Optional ROI restriction (recommended for your "sub-region in overview" setup)
    if search_roi is not None:
        x, y, w, h = search_roi
        x = int(max(0, x)); y = int(max(0, y))
        w = int(min(opt_grad.shape[1] - x, w))
        h = int(min(opt_grad.shape[0] - y, h))
        if w <= 0 or h <= 0:
            raise ValueError("search_roi is empty or out of bounds.")
        opt_grad_search = opt_grad[y:y+h, x:x+w]
        roi_offset = (x, y)
    else:
        opt_grad_search = opt_grad
        roi_offset = (0, 0)

    rows: List[Dict[str, Any]] = []

    for k in range(K):
        name = channel_names[k]
        xrf_k = xrf_stack[k].astype(np.float32, copy=False)

        # Preprocess XRF channel:
        # 1) log1p compress
        xrf_k = np.log1p(np.maximum(xrf_k, 0.0)).astype(np.float32)

        # 2) robust normalize
        xrf_k = robust_normalize(xrf_k)

        # 3) structure image via gradients
        xrf_grad = gradient_magnitude(xrf_k, blur_sigma=xrf_blur_sigma)

        # 4) downsample to optical scale
        xrf_grad_ds = downsample(xrf_grad, factor=downsample_factor)

        best: Optional[Dict[str, Any]] = None

        for ang in angles_deg:
            templ = rotate_image(xrf_grad_ds, angle=float(ang))
            if isinstance(templ, tuple):
                templ = templ[0]

            # Another guard after rotation/downsample: constant template => not informative
            if float(np.std(templ)) < 1e-6 or templ.size < 25:
                psr_res = PSRResult(
                    psr=float("-inf"),
                    peak_value=float("nan"),
                    peak_loc_xy=(0, 0),
                    second_peak_value=float("nan"),
                    second_peak_loc_xy=(0, 0),
                    sidelobe_mean=float("nan"),
                    sidelobe_std=float("nan"),
                )
                corr_map = None
            else:
                corr_map = match_template_corr_normed(opt_grad_search, templ)
                psr_res = compute_psr(corr_map, peak_exclusion_radius=peak_exclusion_radius)

            # Convert peak location in corr-map coords to optical coords (top-left of template placement)
            # matchTemplate returns a map where (x,y) corresponds to template's top-left in the image.
            peak_x, peak_y = psr_res.peak_loc_xy
            peak_x_opt = peak_x + roi_offset[0]
            peak_y_opt = peak_y + roi_offset[1]

            second_x, second_y = psr_res.second_peak_loc_xy
            second_x_opt = second_x + roi_offset[0]
            second_y_opt = second_y + roi_offset[1]

            th, tw = templ.shape[:2]
            # Center location is often what you want for reporting
            peak_center_x_opt = peak_x_opt + tw / 2.0
            peak_center_y_opt = peak_y_opt + th / 2.0

            result_row = dict(
                channel_index=k,
                channel_name=name,
                angle_deg=float(ang),
                psr=float(psr_res.psr),
                peak_value=float(psr_res.peak_value),
                peak_topleft_x_opt=float(peak_x_opt),
                peak_topleft_y_opt=float(peak_y_opt),
                peak_center_x_opt=float(peak_center_x_opt),
                peak_center_y_opt=float(peak_center_y_opt),
                second_peak_value=float(psr_res.second_peak_value),
                second_peak_topleft_x_opt=float(second_x_opt),
                second_peak_topleft_y_opt=float(second_y_opt),
                sidelobe_mean=float(psr_res.sidelobe_mean),
                sidelobe_std=float(psr_res.sidelobe_std),
                template_h=int(th),
                template_w=int(tw),
            )

            if (best is None) or (result_row["psr"] > best["psr"]):
                best = result_row

        assert best is not None
        rows.append(best)

    df = pd.DataFrame(rows)
    df = df.sort_values(["psr", "peak_value"], ascending=[False, False]).reset_index(drop=True)
    return df


def rank_xrf_channels_by_pca_pc1(
    xrf_stack: np.ndarray,
    *,
    channel_names: Optional[Sequence[str]] = None,
    center: bool = True,
    scale: bool = False,
    sample: int = 0,
    xfit_png_path: Optional[str] = None,
    xfit_max_rows: int = 2000,
    xfit_plot: bool = False,
    xfit_cmap: str = "viridis",
) -> pd.DataFrame:
    """
    Rank XRF channels by absolute loadings on the first PCA component.

    Inputs:
      xrf_stack: 3D (K, H, W) with channels as slices.
      center/scale: PCA preprocessing controls (scale implies center).
      sample: optionally subsample pixels for PCA fit (0 = use all).
    """
    if xrf_stack.ndim != 3:
        raise ValueError("xrf_stack must be shape (K, H, W).")

    K = xrf_stack.shape[0]
    if channel_names is None:
        channel_names = [f"ch_{k:03d}" for k in range(K)]
    if len(channel_names) != K:
        raise ValueError("channel_names length must match number of channels in xrf_stack.")

    proc_stack = np.empty_like(xrf_stack, dtype=np.float32)
    for k in range(K):
        # proc_stack[k] = preprocess(xrf_stack[k], normalize=True)
        proc_stack[k] = _preprocess_f32(xrf_stack[k])
    X = proc_stack.reshape(K, -1).T  # (Npixels, K)
    if sample and sample > 0 and sample < X.shape[0]:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], size=sample, replace=False)
        X_fit = X[idx]
    else:
        X_fit = X

    do_center = bool(center or scale)
    if do_center:
        mean = X_fit.mean(axis=0, keepdims=True)
        X_fit = X_fit - mean
    if scale:
        std = X_fit.std(axis=0, keepdims=True) + 1e-8
        X_fit = X_fit / std

    if xfit_png_path or xfit_plot:
        # Visualize X_fit as a heatmap-like PNG (rows = pixels, cols = channels).
        X_vis = X_fit
        if X_vis.shape[0] > xfit_max_rows:
            rng = np.random.default_rng(0)
            idx = rng.choice(X_vis.shape[0], size=xfit_max_rows, replace=True)
            X_vis = X_vis[idx]
        # Robustly normalize for display.
        lo = np.percentile(X_vis, 1.0)
        hi = np.percentile(X_vis, 99.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            X_img = np.zeros_like(X_vis, dtype=np.uint8)
        else:
            X_img = np.clip((X_vis - lo) / (hi - lo), 0.0, 1.0)
            X_img = (X_img * 255).astype(np.uint8)
        if xfit_png_path:
            cv2.imwrite(str(xfit_png_path), X_img)
        if xfit_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.imshow(X_img, aspect="auto", cmap=xfit_cmap)
            plt.colorbar(label="X_fit (robust normalized)")
            plt.xlabel("Channels")
            plt.ylabel("Pixels (sampled)")
            plt.title("X_fit heatmap")
            plt.tight_layout()
            plt.show()

    U, S, Vt = np.linalg.svd(X_fit, full_matrices=False)
    var = (S ** 2) / max(X_fit.shape[0] - 1, 1)
    var_ratio = var / (var.sum() + 1e-12)

    pc1 = Vt[0].astype(np.float32, copy=False)
    rows = []
    for k in range(K):
        rows.append(
            dict(
                channel_index=k,
                channel_name=channel_names[k],
                pca_pc1_loading=float(pc1[k]),
                pca_pc1_abs=float(abs(pc1[k])),
                pca_pc1_variance_ratio=float(var_ratio[0]),
            )
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["pca_pc1_abs"], ascending=[False]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    def _read_settings(path: str) -> Dict[str, Any]:
        img_path_mov = None
        img_path_ref = None
        channels = None
        channel_stride = None
        ref_key = None
        mov_key = None

        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)

        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                continue
            if name == "img_path_mov":
                img_path_mov = value
            elif name == "img_path_ref":
                img_path_ref = value
            elif name == "channels":
                channels = value
            elif name == "channel_stride":
                channel_stride = value
            elif name == "ref_key":
                ref_key = value
            elif name == "mov_key":
                mov_key = value

        if img_path_mov is None or img_path_ref is None:
            raise ValueError("Missing img_path_mov/img_path_ref in settings.")
        return {
            "img_path_mov": img_path_mov,
            "img_path_ref": img_path_ref,
            "channels": channels,
            "channel_stride": channel_stride,
            "ref_key": ref_key,
            "mov_key": mov_key,
        }

    settings = _read_settings("register_ECC_flow_multipleChannel.py")
    img_path_mov = settings["img_path_mov"]
    img_path_ref = settings["img_path_ref"]
    channels = settings["channels"]
    channel_stride = settings["channel_stride"]
    ref_key = settings["ref_key"] or "optical"
    mov_key = settings["mov_key"] or "MAPS/XRF_fits"

    if img_path_ref.lower().endswith(".h5"):
        raise ValueError("Expected optical reference image (non-H5) for this use case.")

    optical_img = cv2.imread(img_path_ref, cv2.IMREAD_GRAYSCALE)
    if optical_img is None:
        raise ValueError(f"Failed to load optical image: {img_path_ref}")

    with h5py.File(img_path_mov, "r") as h5f:
        if mov_key not in h5f:
            raise KeyError(f"Dataset not found: {mov_key}")
        xrf_full = h5f[mov_key]
        if channels is None:
            if channel_stride is None:
                channels = list(range(xrf_full.shape[0]))
            else:
                channels = list(range(xrf_full.shape[0]))[::int(channel_stride)]
        xrf_stack = np.asarray(xrf_full[channels, :, :])
        channel_names = None
        if "MAPS/channel_names" in h5f:
            raw = h5f["MAPS/channel_names"][...]
            all_names = [n.decode("utf-8") for n in raw]
            channel_names = [all_names[c] for c in channels]

    df = rank_xrf_channels_by_peakiness(
        optical_img,
        xrf_stack,
        channel_names=channel_names,
        downsample_factor=10.0,
        angles_deg=(0.0,),
        peak_exclusion_radius=12,
        search_roi=None,
    )
    print(df.head(10))
