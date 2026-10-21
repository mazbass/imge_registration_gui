"""
xrf_registration_core.py
=====================

Self-contained backend for xrf_registration_gui.py.

All required preprocessing, PC1 fusion, SIFT, match-template, metrics and
overlay functionality is implemented in this file.

Nothing here imports Tk / Qt, so the module can be used head-less or tested
from a plain Python shell.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np
import h5py

from skimage.metrics import normalized_mutual_information, structural_similarity
import pandas as pd

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
H5_EXTS = (".h5", ".hdf5")
XRF_KEYS = ("XRF_fits", "XRF_roi", "XRF_roi_plus")
MAPS_GROUP = "MAPS"


# =======================================================================
#  Minimal image/PCA helpers used by this application
#  (kept here so the GUI has no dependency on notebook helper modules)
# =======================================================================
def _to_float01(img):
    arr = np.nan_to_num(np.asarray(img, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr, dtype=np.float32)


def color_fuse(fixed, moving, mode="R|GB", apply_clahe=False):
    fixed01 = _to_float01(fixed)
    moving01 = _to_float01(moving)
    if fixed01.shape != moving01.shape:
        raise ValueError("Overlay images must have the same shape.")
    rgb = np.zeros((*fixed01.shape, 3), dtype=np.float32)
    if mode == "R|GB":
        rgb[..., 0] = fixed01
        rgb[..., 1:] = moving01[..., None]
    else:
        raise ValueError(f"Unsupported fusion mode: {mode}")
    return np.clip(rgb, 0.0, 1.0)


def rotate_image(img, angle, border_value=0, expand=False):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    if expand:
        cos_a, sin_a = abs(matrix[0, 0]), abs(matrix[0, 1])
        out_w = int(h * sin_a + w * cos_a)
        out_h = int(h * cos_a + w * sin_a)
        matrix[0, 2] += out_w / 2.0 - center[0]
        matrix[1, 2] += out_h / 2.0 - center[1]
    else:
        out_w, out_h = w, h
    return cv2.warpAffine(
        img, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


def _preprocess_u8(img, normalize=True):
    arr = np.nan_to_num(np.asarray(img, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if not normalize:
        return arr
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mean, std = cv2.meanStdDev(arr)
    lo = max(0, float(mean[0, 0] - 2.0 * std[0, 0]))
    hi = min(255, float(mean[0, 0] + 2.0 * std[0, 0]))
    hot = cv2.bitwise_or(
        cv2.threshold(arr, hi, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(arr, lo, 255, cv2.THRESH_BINARY_INV)[1],
    )
    hot = cv2.dilate(hot, np.ones((3, 3), np.uint8), iterations=1)
    blurred = cv2.GaussianBlur(arr, (3, 3), 0)
    arr[hot == 255] = blurred[hot == 255]
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.equalizeHist(arr)


def _preprocess_f32(img, sigma_factor=3.0):
    arr = np.nan_to_num(np.asarray(img, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean, std = float(arr.mean()), float(arr.std())
    arr = np.clip(arr, mean - sigma_factor * std, mean + sigma_factor * std)
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr, dtype=np.float32)


def pca_fit(X, center=True, scale=False):
    arr = np.asarray(X, dtype=np.float32)
    mean = arr.mean(axis=0) if center else np.zeros(arr.shape[1], dtype=np.float32)
    arr = arr - mean
    std = arr.std(axis=0) + 1e-8 if scale else np.ones(arr.shape[1], dtype=np.float32)
    arr = arr / std
    _, singular_values, components = np.linalg.svd(arr, full_matrices=False)
    variance = singular_values ** 2 / max(arr.shape[0] - 1, 1)
    return {
        "mean": mean,
        "std": std,
        "components": components,
        "explained_variance": variance,
        "explained_variance_ratio": variance / (variance.sum() + 1e-12),
    }


def rank_xrf_channels_by_pca_pc1(xrf_stack, channel_names=None, center=True,
                                 scale=False, sample=0, xfit_plot=False):
    stack = np.asarray(xrf_stack, dtype=np.float32)
    n_channels = stack.shape[0]
    proc = np.stack([_preprocess_f32(stack[i]) for i in range(n_channels)], axis=0)
    X = proc.reshape(n_channels, -1).T
    if sample and 0 < sample < len(X):
        X = X[np.random.default_rng(0).choice(len(X), size=sample, replace=False)]
    model = pca_fit(X, center=center or scale, scale=scale)
    loading = model["components"][0]
    names = channel_names or [f"ch_{i:03d}" for i in range(n_channels)]
    table = pd.DataFrame({
        "channel_index": np.arange(n_channels),
        "channel_name": names,
        "pca_pc1_loading": loading,
        "pca_pc1_abs": np.abs(loading),
        "pca_pc1_variance_ratio": model["explained_variance_ratio"][0],
    })
    return table.sort_values("pca_pc1_abs", ascending=False).reset_index(drop=True)


def _ncc(a, b):
    x = np.asarray(a, dtype=np.float32).ravel()
    y = np.asarray(b, dtype=np.float32).ravel()
    x, y = x - x.mean(), y - y.mean()
    return float(x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def _image_metrics(ref_img, aligned_img):
    ref = _to_float01(ref_img)
    aligned = _to_float01(aligned_img)
    if ref.shape != aligned.shape:
        aligned = cv2.resize(aligned, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
    gx_ref = cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3)
    gy_ref = cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3)
    gx_aligned = cv2.Sobel(aligned, cv2.CV_32F, 1, 0, ksize=3)
    gy_aligned = cv2.Sobel(aligned, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "mse": float(np.mean((ref - aligned) ** 2)),
        "ncc": _ncc(ref, aligned),
        "gncc": 0.5 * (_ncc(gx_ref, gx_aligned) + _ncc(gy_ref, gy_aligned)),
        "ssim": float(structural_similarity(ref, aligned, data_range=1.0)),
        "mi": float(normalized_mutual_information(ref, aligned)),
    }

# periodic-table element symbols (for identifying elemental XRF channels)
_ELEMENT_SYMBOLS = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I "
    "Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu".split()
)
# scattering channel name variants seen across MAPS setups:
#   inelastic / Compton  : s_i, Compton, COMPTON_AMPLITUDE, ...
#   elastic   / Rayleigh : s_e, Elastic, Rayleigh, COHERENT_SCT_AMPLITUDE, ...
_SCATTER_INELASTIC = ("s_i", "compton")
_SCATTER_ELASTIC = ("s_e", "elastic", "rayleigh", "coherent")
# explicitly-excluded derived/metadata channels (never elements or scatter)
_EXCLUDE_EXACT = {
    "s_a", "tfy", "bkgnd", "background", "total_fluorescence_yield",
    "num_iter", "fit_residual", "sum", "max", "min", "snip_bkg",
}


def _is_scatter(name):
    n = str(name).strip().lower()
    if n in _EXCLUDE_EXACT:
        return False
    return any(t in n for t in _SCATTER_INELASTIC) or any(t in n for t in _SCATTER_ELASTIC)


def is_element_or_scatter(name):
    """
    True for an elemental XRF channel (e.g. 'Fe', 'Ba_L', 'Au_M') or a scattering
    channel (inelastic/Compton or elastic/Rayleigh), matched by name so it works
    for any dataset setup.  Excludes everything else (s_a, TFY, Bkgnd, Num_Iter,
    Fit_Residual, ...).  Re-evaluated per registration against the real channel
    names of the two loaded files.
    """
    n = str(name).strip()
    if n.lower() in _EXCLUDE_EXACT:
        return False
    if _is_scatter(n):
        return True
    base = n.split("_")[0]          # strip line suffix like _L, _M, _K
    return base in _ELEMENT_SYMBOLS


# =======================================================================
#  Channel-string parsing  (from the notebooks)
# =======================================================================
def parse_channels(expr, n_channels):
    """'all' | '0,1,2' | '0:100:2' -> sorted list of channel indices."""
    if expr is None or str(expr).strip().lower() in ("", "all"):
        return list(range(n_channels))

    out = []
    parts = [p.strip() for p in str(expr).split(",") if p.strip()]
    for part in parts:
        if ":" in part:
            fields = part.split(":")
            start = int(fields[0]) if fields[0] else 0
            end = int(fields[1]) if len(fields) > 1 and fields[1] else n_channels
            step = int(fields[2]) if len(fields) == 3 and fields[2] else 1
            out.extend(range(start, end, step))
        else:
            out.append(int(part))
    return sorted(set(i for i in out if 0 <= i < n_channels))


# =======================================================================
#  H5 inspection helpers
# =======================================================================
def list_h5_xrf_keys(path):
    """Return the XRF-like dataset keys available under MAPS/ in an h5 file."""
    keys = []
    with h5py.File(path, "r") as h5f:
        if MAPS_GROUP in h5f:
            grp = h5f[MAPS_GROUP]
            for k in XRF_KEYS:
                if k in grp and isinstance(grp[k], h5py.Dataset) and grp[k].ndim == 3:
                    keys.append(k)
    return keys


def read_channel_names(path, dataset_key):
    """Return a list of channel names for MAPS/<dataset_key>, or generic names."""
    full = f"{MAPS_GROUP}/{dataset_key}"
    with h5py.File(path, "r") as h5f:
        if full not in h5f:
            raise KeyError(f"{full} not found in {os.path.basename(path)}")
        n = h5f[full].shape[0]
        names = None
        if f"{MAPS_GROUP}/channel_names" in h5f:
            raw = h5f[f"{MAPS_GROUP}/channel_names"][...]
            alln = [n_.decode("utf-8") if isinstance(n_, bytes) else str(n_) for n_ in raw]
            if len(alln) >= n:
                names = alln[:n]
        if names is None:
            names = [f"ch_{i:03d}" for i in range(n)]
    return names


# =======================================================================
#  Orientation helpers (flip / rot90)  applied to a 2-D image
# =======================================================================
def apply_orientation(img, flip_x=False, flip_y=False, rot_k=0):
    """
    Manual preprocessing: 90-deg rotation(s) then flips.

    Rotation is applied FIRST so that Flip X / Flip Y always refer to the final
    image's own (local) horizontal/vertical axes, regardless of the rotation.
    Flip X mirrors left-right (columns), Flip Y mirrors up-down (rows).
    """
    out = np.asarray(img)
    if rot_k % 4 != 0:
        out = np.rot90(out, k=rot_k % 4)
    if flip_x:
        out = np.flip(out, axis=1)   # mirror along final X (columns)
    if flip_y:
        out = np.flip(out, axis=0)   # mirror along final Y (rows)
    return np.ascontiguousarray(out)


def flip_contrast_image(img):
    """Invert intensities (dark <-> bright) over the image's own min..max range."""
    a = np.nan_to_num(np.asarray(img, dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(a.min())
    hi = float(a.max())
    if hi <= lo:
        return a
    return (lo + hi) - a


def apply_crop(img, crop):
    """crop = (x0, y0, x1, y1) in pixel coords; None -> unchanged."""
    if crop is None:
        return img
    x0, y0, x1, y1 = [int(round(v)) for v in crop]
    h, w = img.shape[:2]
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    return img[y0:y1, x0:x1]


# =======================================================================
#  PC1 fusion  (lifted from the add_pc1 notebook, with channel-inclusion
#  strategies added for the GUI)
# =======================================================================
def align_pc1_contrast(pc1_img, xrf_stack_selected, pc1_loading=None):
    """Force PC1 image to have positive contrast vs the raw-channel composite."""
    composite = np.nanmean(xrf_stack_selected.astype(np.float32), axis=0)
    a = pc1_img.ravel()
    b = composite.ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() > 1:
        corr = np.corrcoef(a[valid], b[valid])[0, 1]
        if np.isfinite(corr) and corr < 0:
            pc1_img = -pc1_img
            if pc1_loading is not None:
                pc1_loading = -pc1_loading
    return pc1_img, pc1_loading


def _pca_from_stack(stack, n_components=3, center=True, scale=False):
    """PCA on a (C,H,W) stack -> (k,H,W) score images + model."""
    C, H, W = stack.shape
    X = stack.reshape(C, -1).T
    model = pca_fit(X, center=center or scale, scale=scale)
    k = min(n_components, C)
    X_proj = (X.astype(np.float32) - model["mean"]) / model["std"]
    scores = X_proj @ model["components"][:k].T
    return scores.T.reshape(k, H, W), model


def select_pc1_channels(values, index, mode, N_total, top_n=3, manual_threshold=0.15):
    """
    Decide which channels to keep for PC1 fusion.

    `values` are SIGNED PC1 loadings (already sign-aligned so most are positive).
    A negative loading means the channel contributes negatively to the structure.

    mode:
      'top_n'      -> the top_n channels by |loading| (values sorted by |loading| desc)
      'inv_sqrt_n' -> keep channels with |loading| >= 1/sqrt(N_total)
      'half_max'   -> keep channels with |loading| >= 0.5 * max(|loading|)
      'manual'     -> keep channels with SIGNED loading >= manual_threshold
                      (the notebook PC1_threshold; negatives are dropped by a
                      positive threshold)
    Returns a boolean mask aligned with `values`/`index`.
    """
    vals = np.asarray(values, dtype=float)
    absvals = np.abs(vals)

    if mode == "top_n":
        keep = np.zeros(len(absvals), dtype=bool)
        keep[: max(1, int(top_n))] = True          # df is sorted by |loading| desc
        return keep
    if mode == "inv_sqrt_n":
        thr = 1.0 / np.sqrt(max(1, int(N_total)))
        return absvals >= thr
    if mode == "half_max":
        thr = 0.5 * float(absvals.max()) if absvals.size else 0.0
        return absvals >= thr
    if mode == "manual":
        return vals >= float(manual_threshold)      # signed, keeps sign meaning
    raise ValueError(f"unknown pc1 selection mode: {mode!r}")


def build_pc1(
    path,
    dataset_key,
    pc1_pool_selection="all",
    selection_mode="inv_sqrt_n",
    top_n=3,
    manual_threshold=0.15,
    flip_x=False,
    flip_y=False,
    rot_k=0,
    crop=None,
):
    """
    Build a PC1 fusion image from an H5 XRF dataset.

    pc1_pool_selection : channel-string restricting the pool of channels that
                         enter the PCA ranking (default 'all').
    selection_mode     : how to pick the channels that actually build PC1
                         ('top_n' | 'inv_sqrt_n' | 'half_max' | 'manual').

    Returns (pc1_img, names_selected, info_dict).
    """
    full = f"{MAPS_GROUP}/{dataset_key}"
    with h5py.File(path, "r") as h5f:
        xrf_full = h5f[full]
        K = xrf_full.shape[0]
        all_names = None
        if f"{MAPS_GROUP}/channel_names" in h5f:
            raw = h5f[f"{MAPS_GROUP}/channel_names"][...]
            all_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in raw]

        # 'elem+scatter' token -> elemental channels + scattering (per this file)
        if str(pc1_pool_selection).strip().lower() in ("elem+scatter", "elements+scatter", "elem"):
            if all_names:
                ch_list = [i for i in range(K)
                           if i < len(all_names) and is_element_or_scatter(all_names[i])]
            else:
                ch_list = list(range(K))
            if not ch_list:
                ch_list = list(range(K))
        else:
            ch_list = parse_channels(pc1_pool_selection, K)

        xrf_stack = np.asarray(xrf_full[ch_list])
        channel_names = None
        if all_names:
            channel_names = [all_names[c] for c in ch_list]

    # orientation applied to every channel (axis 1=rows=Y, axis 2=cols=X).
    # rotate FIRST, then flip, so Flip X/Y refer to the final local axes.
    if rot_k % 4 != 0:
        xrf_stack = np.rot90(xrf_stack, k=rot_k % 4, axes=(1, 2))
    if flip_x:
        xrf_stack = np.flip(xrf_stack, axis=2)
    if flip_y:
        xrf_stack = np.flip(xrf_stack, axis=1)
    if crop is not None:
        xrf_stack = np.stack([apply_crop(xrf_stack[i], crop) for i in range(xrf_stack.shape[0])], axis=0)
    xrf_stack = np.ascontiguousarray(xrf_stack)

    # rank by PCA PC1 loading (returns a DataFrame sorted by |loading| desc)
    df = rank_xrf_channels_by_pca_pc1(
        xrf_stack,
        channel_names=channel_names,
        center=True,
        scale=False,
        sample=0,
        xfit_plot=False,
    )

    values = df["pca_pc1_loading"].to_numpy()
    index = df["channel_index"].to_numpy()
    names = (
        df["channel_name"].to_numpy()
        if "channel_name" in df
        else df["channel_index"].to_numpy()
    )

    # keep the sign convention from the notebook
    if np.sum(values < 0) >= len(values) // 2:
        values = values * -1

    keep = select_pc1_channels(
        values, index, selection_mode, N_total=len(values),
        top_n=top_n, manual_threshold=manual_threshold,
    )
    if not keep.any():
        keep[:1] = True  # always keep at least one channel

    index_selected = index[keep]
    names_selected = list(names[keep])
    values_selected = values[keep]

    # preprocess the selected channels and build PC1
    sel = np.zeros([len(index_selected), xrf_stack.shape[1], xrf_stack.shape[2]], dtype=np.float32)
    for i, ci in enumerate(index_selected):
        sel[i] = _preprocess_f32(xrf_stack[ci])

    img, model = _pca_from_stack(sel, n_components=max(1, len(index_selected)),
                                 center=True, scale=False)
    pc1_img = img[0]
    pc1_img, _ = align_pc1_contrast(pc1_img, sel, values_selected)

    info = {
        "n_pool": len(values),
        "n_selected": int(keep.sum()),
        "selection_mode": selection_mode,
        "manual_threshold": float(manual_threshold) if selection_mode == "manual" else None,
        "names_selected": names_selected,
        "loadings_selected": [float(v) for v in values_selected],
        "variance_ratio_pc1": float(df["pca_pc1_variance_ratio"].iloc[0]) if "pca_pc1_variance_ratio" in df else None,
    }
    return pc1_img.astype(np.float32), names_selected, info


# =======================================================================
#  Generic image source  ->  single 2-D float image
# =======================================================================
def load_single_channel(path, dataset_key, channel, flip_x=False, flip_y=False,
                        rot_k=0, crop=None):
    """Read one channel from an H5 XRF dataset as a 2-D float array."""
    full = f"{MAPS_GROUP}/{dataset_key}"
    with h5py.File(path, "r") as h5f:
        img = np.asarray(h5f[full][int(channel)]).astype(np.float32)
    img = apply_orientation(img, flip_x, flip_y, rot_k)
    img = apply_crop(img, crop)
    return np.ascontiguousarray(img)


def load_simple_image(path, flip_x=False, flip_y=False, rot_k=0, crop=None):
    """Read a tiff/png/jpg etc. as a 2-D grayscale float array."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.ndim == 3:
        # BGR(A) -> gray
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    img = apply_orientation(img, flip_x, flip_y, rot_k)
    img = apply_crop(img, crop)
    return np.ascontiguousarray(img)


# =======================================================================
#  Source description used by the GUI  (one per ref / tgt)
# =======================================================================
@dataclass
class ImageSource:
    path: str
    kind: str = "h5"                 # "h5" | "image"
    dataset_key: str = "XRF_fits"    # for h5
    channel: int = 0                 # for h5 single-channel
    use_pc1: bool = False            # for h5 PC1 fusion
    # PC1 fusion controls
    pc1_pool_selection: str = "all"
    pc1_selection_mode: str = "inv_sqrt_n"
    pc1_top_n: int = 3
    pc1_manual_threshold: float = 0.15   # for 'manual' mode: keep |loading| >= this
    # manual preprocessing
    flip_x: bool = False
    flip_y: bool = False
    rot_k: int = 0
    flip_contrast: bool = False       # invert intensities (dark<->bright)
    crop: Optional[tuple] = None     # (x0,y0,x1,y1) in ORIGINAL-oriented coords

    def to_raw_image(self):
        """Return (raw_float_image, info_dict). No standard preprocess applied."""
        info = {}
        if self.kind == "image":
            img = load_simple_image(self.path, self.flip_x, self.flip_y,
                                    self.rot_k, self.crop)
            info["source"] = "image file"
        elif self.use_pc1:
            img, names, pc1info = build_pc1(
                self.path, self.dataset_key,
                pc1_pool_selection=self.pc1_pool_selection,
                selection_mode=self.pc1_selection_mode,
                top_n=self.pc1_top_n,
                manual_threshold=self.pc1_manual_threshold,
                flip_x=self.flip_x, flip_y=self.flip_y,
                rot_k=self.rot_k, crop=self.crop,
            )
            info.update(pc1info)
            info["source"] = f"PC1 fusion ({self.dataset_key})"
        else:
            img = load_single_channel(
                self.path, self.dataset_key, self.channel,
                self.flip_x, self.flip_y, self.rot_k, self.crop,
            )
            info["source"] = f"{self.dataset_key}[{self.channel}]"

        img = np.asarray(img, dtype=np.float32)
        if self.flip_contrast:
            img = flip_contrast_image(img)
        info["shape"] = (int(img.shape[0]), int(img.shape[1]))
        return img, info


# =======================================================================
#  Standard preprocess (notebook `preprocess`)  -> uint8
# =======================================================================
def standard_preprocess(img, enabled=True):
    """NaN clean, hotspot removal and histogram equalization -> uint8."""
    return _preprocess_u8(img, normalize=enabled)


# =======================================================================
#  Metrics
# =======================================================================
def rmse_calc(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def standard_metrics(registered, target):
    """
    RMSE / NMI / SSIM + weighted score, on float[0,1] images of equal shape.
    (from 2018_Pourzal_sift_try.ipynb)
    """
    r = _to_float01(registered)
    t = _to_float01(target)
    if r.shape != t.shape:
        t = cv2.resize(t, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_LINEAR)
    rmse = rmse_calc(r, t)
    try:
        mi = float(normalized_mutual_information(r, t))
    except Exception:
        mi = float("nan")
    try:
        ssim = float(
            structural_similarity(
                r, t, data_range=1, full=False,
                win_size=7, gaussian_weights=True, sigma=1.5,
            )
        )
    except Exception:
        ssim = float("nan")
    weight = mi * ssim / (rmse + 1e-6) if np.isfinite(mi) and np.isfinite(ssim) else float("nan")
    return {"rmse": rmse, "nmi": mi, "ssim": ssim, "weighted": weight}


def extended_image_metrics(registered, target):
    """MSE/NCC/GNCC/SSIM/MI, resizing the aligned image when required."""
    m = _image_metrics(target, registered)
    return {k: m[k] for k in ("mse", "ncc", "gncc", "ssim", "mi") if k in m}


# =======================================================================
#  Multi-channel PCA-based metric  (the genuine notebook version)
# =======================================================================
def _load_oriented_stack(src):
    """Load the full (oriented, cropped) channel stack + channel names for an H5 source."""
    full = f"{MAPS_GROUP}/{src.dataset_key}"
    with h5py.File(src.path, "r") as h5f:
        stack = np.asarray(h5f[full][...]).astype(np.float32)   # (C,H,W)
        names = None
        if f"{MAPS_GROUP}/channel_names" in h5f:
            raw = h5f[f"{MAPS_GROUP}/channel_names"][...]
            names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in raw]
    if names is None or len(names) < stack.shape[0]:
        names = [f"ch_{i:03d}" for i in range(stack.shape[0])]
    else:
        names = names[: stack.shape[0]]

    # rotate FIRST, then flip, so Flip X/Y refer to the final local axes.
    if src.rot_k % 4 != 0:
        stack = np.rot90(stack, k=src.rot_k % 4, axes=(1, 2))
    if src.flip_x:
        stack = np.flip(stack, axis=2)
    if src.flip_y:
        stack = np.flip(stack, axis=1)
    if src.crop is not None:
        stack = np.stack([apply_crop(stack[i], src.crop) for i in range(stack.shape[0])], axis=0)
    return np.ascontiguousarray(stack), names


def pca_based_metrics_multichannel(ref_src, tgt_src, M_ref2tgt):
    """
    Genuine PCA-based channel assessment from
    2018_Pourzal_sift_try_with_PCA_sep_no_scale_affine_add_pc1.ipynb.

    Only the elemental channels plus the two scattering channels (s_i = Compton,
    s_e = elastic/Rayleigh) are used; all other channels (s_a, TFY, Bkgnd, etc.)
    are excluded.

    Over those COMMON channels of the two H5 datasets:
      1) warp the whole reference stack into the target frame with the found
         transform (ref -> tgt),
      2) PCA the registered reference stack  -> PC1 loading vector p_reg,
         PCA the target stack               -> PC1 loading vector p_tgt,
      3) compare p_tgt vs p_reg with cosine similarity and Pearson correlation
         (cos_corr_pc1 in the notebook).

    Returns {} if either source is not an H5 dataset or there are < 2 common channels.
    """
    if ref_src.kind != "h5" or tgt_src.kind != "h5":
        return {}
    try:
        ref_stack, ref_names = _load_oriented_stack(ref_src)
        tgt_stack, tgt_names = _load_oriented_stack(tgt_src)

        # common channels, restricted to elements + the two scattering channels
        common = [c for c in tgt_names if c in ref_names and is_element_or_scatter(c)]
        if len(common) < 2:
            return {"pca_note": "fewer than 2 common element/scatter channels; PCA metric skipped"}

        idx_ref = [ref_names.index(c) for c in common]
        idx_tgt = [tgt_names.index(c) for c in common]
        ref_common = ref_stack[idx_ref]
        tgt_common = tgt_stack[idx_tgt]

        # warp each ref channel into the tgt frame using M_ref2tgt (ref -> tgt)
        h_t, w_t = tgt_common.shape[1:]
        reg_common = np.zeros((len(common), h_t, w_t), dtype=np.float32)
        for k in range(len(common)):
            warped = cv2.warpAffine(ref_common[k], M_ref2tgt, (w_t, h_t), flags=cv2.INTER_LINEAR)
            reg_common[k] = _preprocess_f32(warped)
        tgt_proc = np.stack([_preprocess_f32(tgt_common[k]) for k in range(len(common))], axis=0)

        _, model_tgt = _pca_from_stack(tgt_proc, n_components=len(common), center=True, scale=False)
        _, model_reg = _pca_from_stack(reg_common, n_components=len(common), center=True, scale=False)
        p_tgt = np.asarray(model_tgt["components"][0], float)
        p_reg = np.asarray(model_reg["components"][0], float)

        # cos_corr_pc1: sign-align, L2-normalise, cosine + Pearson
        if np.dot(p_tgt, p_reg) < 0:
            p_reg = -p_reg
        a = p_tgt / (np.linalg.norm(p_tgt) + 1e-12)
        b = p_reg / (np.linalg.norm(p_reg) + 1e-12)
        cos_sim = float(np.dot(a, b))
        corr = float(np.corrcoef(a, b)[0, 1])
        return {
            "pca_cosine": cos_sim,
            "pca_pearson": corr,
            "pca_n_channels": len(common),
            "pca_channels": list(common),
        }
    except Exception as e:
        return {"pca_error": str(e)}


# =======================================================================
#  Affine builders (match-template) from the notebooks
# =======================================================================
def affine_from_scale_angle_loc(w_tgt, h_tgt, scale, angle_deg, loc_xy, rot_pad=True):
    """
    Original-tgt pixel coords -> ref coords, consistent with how the template
    was produced during matchTemplate:

        tgt_s = resize(img_tgt, scale)
        tmpl  = rotate_image(tgt_s, angle, expand=rot_pad)   # pad (True) or cut (False)
        loc_xy = matchTemplate top-left of tmpl in img_ref

    rot_pad=True  (pad): rotation expands the canvas to (tw, th); the rotated
                  image is recentered, so the template origin shifts.
    rot_pad=False (cut): rotation keeps the scaled size (w_s, h_s); the rotation
                  is about the scaled-image centre with NO expand shift, so the
                  template top-left corresponds directly to loc_xy.  This shift
                  difference changes the mapped lower-left corner, so it must be
                  reflected in the overlay transform.

    Returns (A_tgt2ref 2x3, (tw, th) template size actually used).
    """
    w_s = int(round(w_tgt * scale))
    h_s = int(round(h_tgt * scale))
    cx, cy = (w_s - 1) / 2.0, (h_s - 1) / 2.0
    R = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0).astype(np.float32)
    if rot_pad:
        cos_a = abs(R[0, 0])
        sin_a = abs(R[0, 1])
        tw = int(np.ceil(h_s * sin_a + w_s * cos_a))
        th = int(np.ceil(h_s * cos_a + w_s * sin_a))
        R[0, 2] += (tw - 1) / 2.0 - cx
        R[1, 2] += (th - 1) / 2.0 - cy
    else:
        # cut: keep the scaled size, no recentering shift
        tw, th = w_s, h_s
    x0, y0 = loc_xy
    T3 = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float32)
    S3 = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
    R3 = np.vstack([R, [0, 0, 1]]).astype(np.float32)
    A_tgt2ref = (T3 @ R3 @ S3)[:2, :]
    return A_tgt2ref.astype(np.float32), (tw, th)


# =======================================================================
#  Overlay (union canvas, tgt warped into ref frame) from notebook
# =======================================================================
def build_overlay(img_ref, img_tgt, M_ref2tgt):
    """
    M_ref2tgt : 2x3 affine mapping ref->tgt (as returned by estimateAffinePartial2D
                when called with (ref_pts, tgt_pts), OR the composed match-template
                transform's inverse convention used in the notebook).

    Returns an RGB overlay (float [0,1]) of the target placed in the reference
    coordinate frame on a union canvas, plus the canvas size.
    """
    h_ref, w_ref = img_ref.shape[:2]
    h_tgt, w_tgt = img_tgt.shape[:2]

    M_inv = cv2.invertAffineTransform(M_ref2tgt)          # tgt -> ref
    M_inv3 = np.vstack([M_inv, [0, 0, 1]]).astype(np.float64)

    tgt_corners = np.array(
        [[0, 0, 1], [w_tgt, 0, 1], [w_tgt, h_tgt, 1], [0, h_tgt, 1]],
        dtype=np.float64,
    ).T
    tgt_in_ref = (M_inv3 @ tgt_corners)[:2].T
    ref_corners = np.array(
        [[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float64
    )
    all_pts = np.vstack([ref_corners, tgt_in_ref])
    x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_pts.max(axis=0)).astype(int)
    W_out = int(x_max - x_min)
    H_out = int(y_max - y_min)

    # guard against a degenerate matrix blowing up the canvas
    max_dim = 4 * max(h_ref, w_ref, h_tgt, w_tgt)
    if W_out <= 0 or H_out <= 0 or W_out > max_dim or H_out > max_dim:
        x_min, y_min = 0, 0
        W_out, H_out = int(w_ref), int(h_ref)

    T_shift = np.array(
        [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64
    )
    M_inv_shift = (T_shift @ M_inv3)[:2, :].astype(np.float32)
    M_ref_shift = T_shift[:2, :].astype(np.float32)

    registered_tgt = cv2.warpAffine(img_tgt, M_inv_shift, (W_out, H_out), flags=cv2.INTER_LINEAR)
    ref_canvas = cv2.warpAffine(img_ref, M_ref_shift, (W_out, H_out), flags=cv2.INTER_LINEAR)
    overlay = color_fuse(registered_tgt, ref_canvas, mode="R|GB", apply_clahe=False)
    return overlay, (W_out, H_out)


def tgt_corner_ll_in_ref(img_tgt_shape, M_ref2tgt):
    """
    Lower-left corner of the target, expressed in the REFERENCE pixel frame.

    With origin='lower' display, the lower-left corner is target pixel (x=0, y=0).
    M_ref2tgt maps ref->tgt, so its inverse maps tgt->ref.
    Returns (x, y) in reference pixels.
    """
    M_inv = cv2.invertAffineTransform(M_ref2tgt)
    pt = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    xy = (np.vstack([M_inv, [0, 0, 1]]).astype(np.float64) @ pt)[:2]
    return float(xy[0]), float(xy[1])


# =======================================================================
#  SIFT registration
# =======================================================================
@dataclass
class RegResult:
    ok: bool = False
    method: str = ""
    overlay: Optional[np.ndarray] = None       # RGB float [0,1]
    registered: Optional[np.ndarray] = None     # tgt warped into ref frame (gray)
    M: Optional[np.ndarray] = None              # 2x3 ref->tgt
    scale: float = float("nan")
    angle: float = float("nan")
    size_relation: str = ""                     # coarse/fine statement
    corner_ll: Optional[tuple] = None          # (x, y) lower-left of tgt in ref frame
    metrics: dict = field(default_factory=dict)
    sift_stats: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)


def register_sift(img_ref_u8, img_tgt_u8, ref_raw, tgt_raw,
                  lowe_ratio=0.9, ransac_conf=0.9, ransac_thresh=5.0,
                  max_iters=5000):
    """
    SIFT + ratio test + RANSAC partial-affine.
    Convention matches the notebook: estimateAffinePartial2D(src=ref_pts,
    dst=tgt_pts) so M maps ref->tgt; the overlay inverts it.
    """
    res = RegResult(method="SIFT")
    sift = cv2.SIFT_create()
    kps1, des1 = sift.detectAndCompute(img_ref_u8, None)
    kps2, des2 = sift.detectAndCompute(img_tgt_u8, None)
    n_kp1 = len(kps1) if kps1 else 0
    n_kp2 = len(kps2) if kps2 else 0

    n_matches = n_good = n_inliers = 0
    if des1 is not None and des2 is not None and n_kp1 >= 2 and n_kp2 >= 2:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = bf.knnMatch(des1, des2, k=2)
        knn = [pair for pair in knn if len(pair) == 2]
        n_matches = len(knn)
        good = [m for m, n in knn if m.distance < lowe_ratio * n.distance]
        n_good = len(good)
    else:
        good = []

    M = None
    inliers = None
    if n_good >= 2:
        src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh, maxIters=max_iters,
            confidence=ransac_conf, refineIters=10,
        )

    res.ok = M is not None
    if not res.ok:
        res.messages.append(f"SIFT registration FAILED (good matches: {n_good}); showing identity overlay.")
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        inliers = np.zeros((n_good, 1), dtype=np.uint8)

    n_inliers = int(np.sum(inliers)) if inliers is not None else 0

    res.M = M
    # M maps ref -> tgt; report the tgt -> ref convention so SIFT scale/angle
    # match the match-template definition (template = tgt resized by `scale`).
    s_ref2tgt = float(np.hypot(M[0, 0], M[1, 0]))
    a_ref2tgt = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    res.scale = (1.0 / s_ref2tgt) if s_ref2tgt > 1e-9 else float("nan")
    res.angle = -a_ref2tgt

    # registered = tgt warped into ref frame (for metrics), notebook uses the
    # inverse convention; here we warp tgt by M_inv onto ref grid.
    h_ref, w_ref = img_ref_u8.shape[:2]
    M_inv = cv2.invertAffineTransform(M)
    registered = cv2.warpAffine(img_tgt_u8, M_inv, (w_ref, h_ref), flags=cv2.INTER_LINEAR)
    res.registered = registered

    res.overlay, _ = build_overlay(img_ref_u8, img_tgt_u8, M)
    res.corner_ll = tgt_corner_ll_in_ref(img_tgt_u8.shape, M)

    res.metrics.update(standard_metrics(registered, img_ref_u8))
    res.metrics.update(extended_image_metrics(registered, img_ref_u8))

    res.sift_stats = {
        "keypoints_ref": n_kp1,
        "keypoints_tgt": n_kp2,
        "matches": n_matches,
        "good_matches": n_good,
        "ransac_inliers": n_inliers,
    }
    return res


# =======================================================================
#  Match-template registration (grid search over scale & rotation)
# =======================================================================
def register_match_template(img_ref_u8, img_tgt_u8, ref_raw, tgt_raw,
                            scale_min=0.2, scale_max=0.6, scale_steps=41,
                            angle_min=-5.0, angle_max=5.0, angle_steps=21,
                            rot_pad=True, progress=None):
    """
    Brute-force scale+rotation matchTemplate (TM_CCOEFF_NORMED).

    rot_pad : if True, rotation expands the canvas (pad the parts that go out);
              if False, rotation keeps the template size (cut the overflow).
    progress: optional callable(frac) in [0,1].
    """
    res = RegResult(method="Match Template")

    scale_list = np.linspace(scale_min, scale_max, max(1, int(scale_steps)))
    angle_list = np.linspace(angle_min, angle_max, max(1, int(angle_steps)))
    H, W = img_ref_u8.shape[:2]

    best = {"val": -np.inf, "loc": None, "scale": None, "angle": None, "size": None}
    total = len(scale_list) * len(angle_list)
    done = 0

    for scale in scale_list:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        tgt_s = cv2.resize(img_tgt_u8, (0, 0), fx=scale, fy=scale, interpolation=interp)
        for angle in angle_list:
            tmpl_rot = rotate_image(tgt_s, angle, border_value=0, expand=bool(rot_pad))
            tmpl = standard_preprocess(tmpl_rot, enabled=True)
            th, tw = tmpl.shape[:2]
            done += 1
            if progress and total:
                progress(done / total)
            if th > H or tw > W or th < 2 or tw < 2:
                continue
            if float(np.std(tmpl)) < 1e-6:
                continue
            result = cv2.matchTemplate(img_ref_u8, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best["val"]:
                best.update(val=float(max_val), loc=max_loc, scale=float(scale),
                            angle=float(angle), size=(th, tw))

    if best["loc"] is None:
        res.ok = False
        res.messages.append("Match-template failed: template never fit inside reference at any scale/angle.")
        res.M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        res.registered = img_tgt_u8.copy()
        res.overlay, _ = build_overlay(img_ref_u8, img_tgt_u8, res.M)
        return res

    res.ok = True
    res.scale = best["scale"]
    res.angle = best["angle"]

    # affine mapping original tgt -> ref (must match pad/cut used when matching)
    A_tgt2ref, _ = affine_from_scale_angle_loc(
        w_tgt=img_tgt_u8.shape[1], h_tgt=img_tgt_u8.shape[0],
        scale=best["scale"], angle_deg=best["angle"], loc_xy=best["loc"],
        rot_pad=bool(rot_pad),
    )
    # express as ref->tgt to reuse build_overlay / notebook convention
    M_ref2tgt = cv2.invertAffineTransform(A_tgt2ref)
    res.M = M_ref2tgt

    h_ref, w_ref = img_ref_u8.shape[:2]
    registered = cv2.warpAffine(img_tgt_u8, A_tgt2ref, (w_ref, h_ref), flags=cv2.INTER_LINEAR)
    res.registered = registered

    res.overlay, _ = build_overlay(img_ref_u8, img_tgt_u8, M_ref2tgt)
    res.corner_ll = tgt_corner_ll_in_ref(img_tgt_u8.shape, M_ref2tgt)

    res.metrics.update(standard_metrics(registered, img_ref_u8))
    res.metrics.update(extended_image_metrics(registered, img_ref_u8))
    res.sift_stats = {}
    return res


# =======================================================================
#  Top-level entry point used by the GUI
# =======================================================================
def run_registration(
    ref_src: ImageSource,
    tgt_src: ImageSource,
    method="sift",
    standard_pp=True,
    sift_params=None,
    mt_params=None,
    progress=None,
):
    """
    Full pipeline: load both sources -> (optional) standard preprocess ->
    SIFT or match-template -> result with overlay, metrics, transform.
    """
    sift_params = sift_params or {}
    mt_params = mt_params or {}

    ref_raw, ref_info = ref_src.to_raw_image()
    tgt_raw, tgt_info = tgt_src.to_raw_image()

    ref_u8 = standard_preprocess(ref_raw, enabled=standard_pp)
    tgt_u8 = standard_preprocess(tgt_raw, enabled=standard_pp)
    # standard_preprocess may return float32 when disabled; coerce to uint8 for cv2 feature ops
    if ref_u8.dtype != np.uint8:
        ref_u8 = cv2.normalize(ref_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if tgt_u8.dtype != np.uint8:
        tgt_u8 = cv2.normalize(tgt_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if method.lower() == "sift":
        res = register_sift(
            ref_u8, tgt_u8, ref_raw, tgt_raw,
            lowe_ratio=sift_params.get("lowe_ratio", 0.9),
            ransac_conf=sift_params.get("ransac_conf", 0.9),
            ransac_thresh=sift_params.get("ransac_thresh", 5.0),
            max_iters=sift_params.get("max_iters", 5000),
        )
    elif method.lower() in ("match_template", "template", "mt"):
        res = register_match_template(
            ref_u8, tgt_u8, ref_raw, tgt_raw,
            scale_min=mt_params.get("scale_min", 0.2),
            scale_max=mt_params.get("scale_max", 0.6),
            scale_steps=mt_params.get("scale_steps", 41),
            angle_min=mt_params.get("angle_min", -5.0),
            angle_max=mt_params.get("angle_max", 5.0),
            angle_steps=mt_params.get("angle_steps", 21),
            rot_pad=mt_params.get("rot_pad", True),
            progress=progress,
        )
    else:
        raise ValueError(f"unknown method: {method!r}")

    # genuine multi-channel PCA-based metric (both sources must be H5)
    if res.M is not None:
        res.metrics.update(pca_based_metrics_multichannel(ref_src, tgt_src, res.M))

    # coarse/fine relation inferred from the tgt->ref scale
    # scale < 1 : target shrunk to match reference  -> target is finer
    # scale > 1 : target enlarged to match reference -> target is coarser
    if np.isfinite(res.scale):
        if res.scale < 1.0:
            res.size_relation = ("reference = coarse, target = fine "
                                 f"(target {1.0/res.scale:.2f}x finer)")
        elif res.scale > 1.0:
            res.size_relation = ("reference = fine, target = coarse "
                                 f"(target {res.scale:.2f}x coarser)")
        else:
            res.size_relation = "reference and target at the same scale"

    res.ref_image = ref_u8
    res.tgt_image = tgt_u8
    res.ref_info = ref_info
    res.tgt_info = tgt_info
    return res
