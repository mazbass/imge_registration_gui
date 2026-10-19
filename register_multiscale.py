import cv2
import numpy as np
import os
import h5py
import cv2
import matplotlib.pyplot as plt
plt.ion()

from hotspot_filter import remove_hotspots_auto, detect_hotspots
from skimage.metrics import normalized_mutual_information
from image_module import snr, load_image, rotate_image, center_crop_2d,find_angle, color_fuse


def compare_match_window_pixel_bbox(
    top_left,
    win_size,
    corners_ref_px=None,
    x_mov=None,
    y_mov=None,
):
    """
    Compare a matchTemplate window to a moving-image footprint
    using PURE PIXEL COORDINATES.

    Parameters
    ----------
    top_left : (j0, i0)
        matchTemplate top-left corner (col, row) in reference image.
    win_size : (W_win, H_win)
        Width and height of the matched window (pixels).
    corners_ref_px : (4, 2) array-like, optional
        Moving image footprint corners in reference pixel coords [j, i].
    x_mov, y_mov : (4,) arrays, optional
        Alternative to corners_ref_px: columns and rows of 4 corners.

    Returns
    -------
    dict with:
        bbox_ref, bbox_mov, metrics (pixel units)
    """

    # ---- Reference window bbox ----
    j0, i0 = top_left
    W_win, H_win = win_size

    j1 = j0 + W_win - 1
    i1 = i0 + H_win - 1

    bbox_ref = {
        "x_min": float(j0),
        "x_max": float(j1),
        "y_min": float(i0),
        "y_max": float(i1),
    }

    # ---- Moving image bbox from corners ----
    if corners_ref_px is None:
        if x_mov is None or y_mov is None:
            raise ValueError("Provide corners_ref_px or x_mov and y_mov")
        corners_ref_px = np.stack([x_mov, y_mov], axis=1)
    else:
        corners_ref_px = np.asarray(corners_ref_px)

    j_c = corners_ref_px[:, 0]
    i_c = corners_ref_px[:, 1]

    bbox_mov = {
        "x_min": float(np.min(j_c)),
        "x_max": float(np.max(j_c)),
        "y_min": float(np.min(i_c)),
        "y_max": float(np.max(i_c)),
    }

    # ---- Metrics ----
    cx_ref = 0.5 * (bbox_ref["x_min"] + bbox_ref["x_max"])
    cy_ref = 0.5 * (bbox_ref["y_min"] + bbox_ref["y_max"])
    cx_mov = 0.5 * (bbox_mov["x_min"] + bbox_mov["x_max"])
    cy_mov = 0.5 * (bbox_mov["y_min"] + bbox_mov["y_max"])

    center_dx = cx_ref - cx_mov
    center_dy = cy_ref - cy_mov
    center_dist = float(np.sqrt(center_dx**2 + center_dy**2))

    w_ref = bbox_ref["x_max"] - bbox_ref["x_min"]
    h_ref = bbox_ref["y_max"] - bbox_ref["y_min"]
    w_mov = bbox_mov["x_max"] - bbox_mov["x_min"]
    h_mov = bbox_mov["y_max"] - bbox_mov["y_min"]

    ix_min = max(bbox_ref["x_min"], bbox_mov["x_min"])
    ix_max = min(bbox_ref["x_max"], bbox_mov["x_max"])
    iy_min = max(bbox_ref["y_min"], bbox_mov["y_min"])
    iy_max = min(bbox_ref["y_max"], bbox_mov["y_max"])

    inter_w = max(0.0, ix_max - ix_min)
    inter_h = max(0.0, iy_max - iy_min)
    inter_area = inter_w * inter_h

    area_ref = max(0.0, w_ref) * max(0.0, h_ref)
    area_mov = max(0.0, w_mov) * max(0.0, h_mov)
    union_area = area_ref + area_mov - inter_area
    iou = float(inter_area / union_area) if union_area > 0 else 0.0

    metrics = {
        "center_dx": float(center_dx),
        "center_dy": float(center_dy),
        "center_dist": center_dist,
        "ref_width": float(w_ref),
        "ref_height": float(h_ref),
        "mov_width": float(w_mov),
        "mov_height": float(h_mov),
        "delta_width": float(w_ref - w_mov),
        "delta_height": float(h_ref - h_mov),
        "intersection_area": float(inter_area),
        "iou": iou,
    }

    return {
        "bbox_ref": bbox_ref,
        "bbox_mov": bbox_mov,
        "metrics": metrics,
    }
def compare_match_window_physical_bbox(
    x_ref, y_ref,
    top_left, win_size,
    x_mov=None, y_mov=None,
    corners_ref_px=None,
):
    """
    Compare:
      (A) matchTemplate window bbox (in reference image)
    vs
      (B) moving-image bbox given by 4 corner pixel indices in *reference pixel coordinates*.

    Parameters
    ----------
    x_ref : (W_ref,) array
        Physical x coordinate for each column of reference image.
    y_ref : (H_ref,) array
        Physical y coordinate for each row of reference image.
    top_left : (j0, i0)
        Template match location in reference pixel coords (col, row).
    win_size : (W_win, H_win)
        Window size in pixels (width, height).

    corners_ref_px : (4, 2) array-like, optional
        Four corners in reference pixel coords as (col=j, row=i).
        Example: [[j0,i0],[j1,i0],[j1,i1],[j0,i1]] (order doesn't matter for bbox).
    x_mov, y_mov : array-like, optional
        Alternative to corners_ref_px:
        1) length-4 pixel indices in reference frame (legacy behavior), or
        2) full physical coordinate axes of the moving image. In this case,
           moving bbox is built from [min,max] of the provided axes.

    Returns
    -------
    result : dict with bbox_ref, bbox_mov, metrics (center offset, size mismatch, IoU)
    """
    x_ref = np.asarray(x_ref)
    y_ref = np.asarray(y_ref)
    W_ref = x_ref.shape[0]
    H_ref = y_ref.shape[0]

    # --- Reference bbox from template match window ---
    j0, i0 = top_left
    W_win, H_win = win_size

    j0 = int(np.clip(j0, 0, W_ref - 1))
    i0 = int(np.clip(i0, 0, H_ref - 1))
    j1 = int(np.clip(j0 + W_win - 1, 0, W_ref - 1))
    i1 = int(np.clip(i0 + H_win - 1, 0, H_ref - 1))

    xr0, xr1 = float(x_ref[j0]), float(x_ref[j1])
    yr0, yr1 = float(y_ref[i0]), float(y_ref[i1])

    bbox_ref = {
        "x_min": min(xr0, xr1),
        "x_max": max(xr0, xr1),
        "y_min": min(yr0, yr1),
        "y_max": max(yr0, yr1),
    }

    # --- Moving bbox ---
    if corners_ref_px is not None:
        corners_ref_px = np.asarray(corners_ref_px)
        if corners_ref_px.shape != (4, 2):
            raise ValueError(f"corners_ref_px must have shape (4,2), got {corners_ref_px.shape}")

        j_c = corners_ref_px[:, 0].astype(int)  # col indices
        i_c = corners_ref_px[:, 1].astype(int)  # row indices

        # Clamp to reference bounds
        j_c = np.clip(j_c, 0, W_ref - 1)
        i_c = np.clip(i_c, 0, H_ref - 1)

        # Convert corner pixels to physical coords
        x_c = x_ref[j_c]
        y_c = y_ref[i_c]
        bbox_mov = {
            "x_min": float(np.min(x_c)),
            "x_max": float(np.max(x_c)),
            "y_min": float(np.min(y_c)),
            "y_max": float(np.max(y_c)),
        }
    else:
        if x_mov is None or y_mov is None:
            raise ValueError("Provide either corners_ref_px or both x_mov and y_mov.")
        x_mov_arr = np.asarray(x_mov).reshape(-1)
        y_mov_arr = np.asarray(y_mov).reshape(-1)

        if x_mov_arr.size == 4 and y_mov_arr.size == 4:
            # Legacy path: 4 corner pixel indices in reference frame.
            j_c = np.clip(x_mov_arr.astype(int), 0, W_ref - 1)
            i_c = np.clip(y_mov_arr.astype(int), 0, H_ref - 1)
            x_c = x_ref[j_c]
            y_c = y_ref[i_c]
            bbox_mov = {
                "x_min": float(np.min(x_c)),
                "x_max": float(np.max(x_c)),
                "y_min": float(np.min(y_c)),
                "y_max": float(np.max(y_c)),
            }
        else:
            # New path: moving physical axes (e.g., H5 x_axis/y_axis vectors).
            bbox_mov = {
                "x_min": float(np.min(x_mov_arr)),
                "x_max": float(np.max(x_mov_arr)),
                "y_min": float(np.min(y_mov_arr)),
                "y_max": float(np.max(y_mov_arr)),
            }

    # --- Metrics: center offset, size mismatch, overlap / IoU ---
    cx_ref = 0.5 * (bbox_ref["x_min"] + bbox_ref["x_max"])
    cy_ref = 0.5 * (bbox_ref["y_min"] + bbox_ref["y_max"])
    cx_mov = 0.5 * (bbox_mov["x_min"] + bbox_mov["x_max"])
    cy_mov = 0.5 * (bbox_mov["y_min"] + bbox_mov["y_max"])

    center_dx = cx_ref - cx_mov
    center_dy = cy_ref - cy_mov
    center_dist = float(np.sqrt(center_dx**2 + center_dy**2))

    w_ref = bbox_ref["x_max"] - bbox_ref["x_min"]
    h_ref = bbox_ref["y_max"] - bbox_ref["y_min"]
    w_mov = bbox_mov["x_max"] - bbox_mov["x_min"]
    h_mov = bbox_mov["y_max"] - bbox_mov["y_min"]

    ix_min = max(bbox_ref["x_min"], bbox_mov["x_min"])
    ix_max = min(bbox_ref["x_max"], bbox_mov["x_max"])
    iy_min = max(bbox_ref["y_min"], bbox_mov["y_min"])
    iy_max = min(bbox_ref["y_max"], bbox_mov["y_max"])

    inter_w = max(0.0, ix_max - ix_min)
    inter_h = max(0.0, iy_max - iy_min)
    inter_area = inter_w * inter_h

    area_ref = max(0.0, w_ref) * max(0.0, h_ref)
    area_mov = max(0.0, w_mov) * max(0.0, h_mov)
    union_area = area_ref + area_mov - inter_area
    iou = float(inter_area / union_area) if union_area > 0 else 0.0

    metrics = {
        "center_dx": float(center_dx),
        "center_dy": float(center_dy),
        "center_dist": center_dist,
        "ref_width": float(w_ref),
        "ref_height": float(h_ref),
        "mov_width": float(w_mov),
        "mov_height": float(h_mov),
        "delta_width": float(w_ref - w_mov),
        "delta_height": float(h_ref - h_mov),
        "intersection_area": float(inter_area),
        "iou": iou,
    }

    return {"bbox_ref": bbox_ref, "bbox_mov": bbox_mov, "metrics": metrics}
def preprocess(img, normalize=True):
    """
    1) Normalize → remove hotspots → clip outliers → rescale → equalize → blur.
    Set normalize=False to skip intensity normalization/equalization.
    """
    # Always sanitize NaNs/Infs.
    img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if not normalize:
        return img.astype(np.float32, copy=False)

    # 1) bring to 0–255 uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)

    # 2) remove bright/dark hotspots
    mask = detect_hotspots(img, sigma_factor=2)
    img = remove_hotspots_auto(img, mask, method="blur")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)

    # import pdb; pdb.set_trace()


    # 3) robust clipping to [mean–2σ, mean+2σ]
    # arr = img.astype(np.float32)
    # m, s = np.nanmean(arr), np.nanstd(arr)
    # arr = np.clip(arr, m - 2*s, m + 2*s)

    # 4) rescale back to full 0–255 range (guard against flat image)
    # vmin = float(arr.min())
    # vmax = float(arr.max())
    # if vmax > vmin:
    #     arr = (arr - vmin) / (vmax - vmin)
    # else:
    #     arr = np.zeros_like(arr, dtype=np.float32)
    # img = (arr * 255).astype(np.uint8)

    # 5) histogram equalization
    img = cv2.equalizeHist(img)

    # 6) final smoothing
    # img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def draw_matches(img1, kp1, img2, kp2, matches, element=None, top_n=20):
    """Draw top_n matches between img1 & img2."""
    img_m = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches[:top_n], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(12, 6))
    plt.imshow(img_m, cmap='gray')
    title = f"{element}: Top {top_n} Matches" if element else f"Top {top_n} Matches"
    plt.title(title)
    plt.axis('off')
    plt.show()



def assign_intensity_centroid_orientation(img: np.ndarray, keypoints, patch_radius=16):
    """
    For each cv2.KeyPoint in keypoints, compute its angle
    as the arctan2 of the intensity‐centroid of a local patch.
    """
    h, w = img.shape
    for kp in keypoints:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        r = int(round(kp.size/2)) if kp.size>0 else patch_radius

        # extract patch, clamped
        x0, x1 = max(x-r,0), min(x+r+1, w)
        y0, y1 = max(y-r,0), min(y+r+1, h)
        patch = img[y0:y1, x0:x1].astype(np.float32)

        # coords relative to center
        ys, xs = np.mgrid[y0:y1, x0:x1]
        u = xs - x
        v = ys - y

        # moments
        m00 = patch.sum() + 1e-6
        m10 = (u * patch).sum()
        m01 = (v * patch).sum()

        # centroid vector and angle
        cx, cy = m10/m00, m01/m00
        angle = (np.degrees(np.arctan2(cy, cx)) + 360) % 360
        kp.angle = angle

    return keypoints


def register_cropped_region(
    img_path_ref, ref_key,
    img_path_tgt, tgt_key,
    channel,
    draw_n_matches=20
):
    """
    Align a small target patch (tgt) back into the large reference (ref).
    Returns: aligned, ref, tgt, transform M, error_metrics
    """
    # ── Load & preprocess ──────────────────────────────────────────────────────
    ref_img, ref_element, x_coords_ref, y_coords_ref = load_image(img_path_ref, ref_key, channel)
    ref = preprocess(ref_img)
    scale =66
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    ref = cv2.resize(ref, (0, 0), fx=scale, fy=scale, interpolation=interp)

    # ref = rotate_image(ref, angle=45)

    tgt_img, tgt_element, x_coords_tgt, y_coords_tgt = load_image(img_path_tgt, tgt_key, channel)
    tgt = preprocess(tgt_img[:,:]) #100
    # import pdb; pdb.set_trace()

    # angle, rot_matrix, aligned = find_angle(tgt, ref[1:1948,1:2048])

    ### ── Detect ORB keypoints only ───────────────────────────────────
    ## use a 64×64 patch around each keypoint
    orb = cv2.ORB_create(
      nfeatures=10000,
      patchSize=128,         # ← increase this to cover a larger neighborhood
      edgeThreshold=64,  # must be >= patchSize/2 + 1
      scaleFactor=1.2,
      nlevels=16
    )

    kp_ref = orb.detect(ref, None)
    kp_tgt = orb.detect(tgt, None)
    # # ── Assign orientation by intensity‐centroid ──────────────────
    kp_ref = assign_intensity_centroid_orientation(ref, kp_ref)
    kp_tgt = assign_intensity_centroid_orientation(tgt, kp_tgt)
    kp_ref, des_ref = orb.compute(ref, kp_ref)
    kp_tgt, des_tgt = orb.compute(tgt, kp_tgt)

    if des_ref is None or des_tgt is None:
        raise RuntimeError("No descriptors found in one of the images")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ####### -------use SIFT -------------------------
    # sift = cv2.SIFT_create(
    #     nfeatures=5000,
    #     nOctaveLayers=4,
    #     contrastThreshold=0.03,
    #     edgeThreshold=10,
    #     sigma=1.4
    # )
    # kp_ref = sift.detect(ref, None)
    # kp_tgt = sift.detect(tgt, None)
    # # ── Assign orientation by intensity‐centroid ──────────────────
    # kp_ref = assign_intensity_centroid_orientation(ref, kp_ref)
    # kp_tgt = assign_intensity_centroid_orientation(tgt, kp_tgt)
    # kp_ref, des_ref = sift.compute(ref, kp_ref)
    # kp_tgt, des_tgt = sift.compute(tgt, kp_tgt)
    # if des_ref is None or des_tgt is None:
    #   raise RuntimeError("No descriptors found in one of the images")

    ############################################################
    # after you’ve computed des_ref, des_tgt and kp_ref, kp_tgt…

    # 1) forward and backward matches
    # bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_ref, des_tgt,k=2)   # A→B
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # ← ratio threshold
            good.append(m)

    print(f"{len(good)} mutual matches found")

    # 4) guard against empty set
    if len(good) < 3:
        raise RuntimeError("Too few mutual matches; aborting registration")

    # 5) sort by distance and then build your point arrays
    matches = sorted(good, key=lambda m: m.distance)

    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts_tgt = np.float32([kp_tgt[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    draw_matches(ref, kp_ref, tgt, kp_tgt,
               matches, element=ref_element,
               top_n=draw_n_matches)



    # ── Estimate transform ────────────────────────────────────────────────────
    # M_affine, mask = cv2.estimateAffinePartial2D(pts_tgt, pts_ref,
    #                                           method=cv2.RANSAC)
    M_affine, mask = cv2.estimateAffinePartial2D(
        pts_tgt, pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,   # pixels: lower → stricter
        maxIters=2000,
        confidence=0.995
    )
    if M_affine is not None:
        M = M_affine.astype(np.float32)
        warp_fn = cv2.warpAffine
    else:
        M_homo, mask = cv2.findHomography(pts_tgt, pts_ref,
                                          cv2.RANSAC, 3.0)
        if M_homo is None:
            raise RuntimeError("Transform estimation failed")
        M = M_homo.astype(np.float32)
        warp_fn = cv2.warpPerspective

    # ── Warp target into reference frame ───────────────────────────────────────
    # M = np.load("my_array.npy")
    H_ref, W_ref = ref.shape
    aligned = warp_fn(tgt, M, (W_ref, H_ref))
    # plt.figure()
    # plt.imshow(aligned)
    # plt.show()


    # ── Compute error metrics over overlap ────────────────────────────────────
    overlap = (aligned > 0)
    diff = (aligned.astype(float) - ref.astype(float)) * overlap
    rmse = np.sqrt(np.mean(diff[overlap]**2))
    mi   = normalized_mutual_information(aligned*overlap, ref*overlap)
    md   = np.mean([m.distance for m in matches])
    inl  = int(mask.sum()) if mask is not None else 0
    err = np.array([md, len(matches), rmse, mi])

    # # Show absolute‐difference map
    # plt.figure(figsize=(6,6))
    # plt.imshow(diff, cmap="gray")
    # plt.colorbar(label="|aligned – ref|")
    # plt.title("Absolute Difference (darker = better)")
    # plt.axis("off")
    # plt.show()

    return aligned, ref, tgt, ref_img, tgt_img, M, err

# — example usage —
if __name__ == "__main__":
    h5_a = "data/2018c2bnp_2018c32ide_Pourzal/2xfm_0006.h5"
    h5_b = "data/2018c2bnp_2018c32ide_Pourzal/2xfm_0012.h5"
    channels =[21] #[4, 5, 6, 15, 22] # np.append(np.arange(4, 16, 1),[22]) #  [2, 3, 5, 6, 8, 22] #np.arange(6,9) #
    # Pre-allocate an array of shape (n_channels, error_dim)
    error_all = np.zeros((len(channels), 4), dtype=float)
    snr_ratio = np.zeros((len(channels), 1), dtype=float)
    # layout: 2 rows × ceil(n/2) columns (adjust as you like)
    n = len(channels)
    cols = len(channels)  # for 6 channels, 2×3 grid; tweak for your n
    rows = 3 #int(np.ceil(n/cols))+1
    fig = plt.figure(figsize=(4*n, 8))
    gs  = fig.add_gridspec(rows, n, height_ratios=[1,1.2, 1.2], hspace=0.3)


    for i, ch in enumerate(channels):
        row = i // cols
        col = i %  cols
        ax  = fig.add_subplot(gs[row, col])
        # 1) register
        try:
            aligned, ref_img, tgt_img, ref_img0, tgt_img0, M, error = register_cropped_region(
              h5_a, "MAPS/XRF_fits",
              h5_b, "MAPS/XRF_fits",
              channel=ch
            )
            print(np.max(ref_img0))

            error_all[i] = error
            snr_ratio[i] = snr(ref_img0)
        except RuntimeError as e:
            print(f"Channel {ch}: {e} — skipping")
            continue    # ← valid here inside the loop

        # 2) print metrics
        print(f"Channel {ch} error metrics: {error}")

        # 3) get the human‐readable channel name (if HDF5)
        _, channel_name, _, _= load_image(h5_a, "MAPS/XRF_fits", ch)

        ## 4) overlay & show
        # overlay_images(ref_img, aligned, element=channel_name)
        # build the blended overlay
        # blended = cv2.addWeighted(
        #     ref_img.astype(np.float32), 0.5,
        #     aligned.astype(np.float32), 0.5,
        #     0
        # )
        # blended = np.clip(blended, 0, 255).astype(np.uint8)
        blended = color_fuse(ref_img, aligned, mode="R|GB", apply_clahe=False)

        # plot into subplot
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(blended, cmap="gray")
        ax.axis("off")
        ax.set_title(channel_name)


    A = error_all
    # 1) Compute per‐column minima and maxima
    col_min = A.min(axis=0)   # shape (6,)
    col_max = A.max(axis=0)   # shape (6,)

    # 2) Avoid divide‐by‐zero: if max==min, set denom to 1
    denom = np.where(col_max > col_min, col_max - col_min, 1)

    # 3) Normalize
    A_norm = (A - col_min) / denom
    ax_snr = fig.add_subplot(gs[1, :])  # row = n_rows-1, all columns
    ax_snr.plot(np.arange(1,n+1),snr_ratio[:], marker='o', linewidth=2)
    ax_snr.set_ylabel("SNR")
    ax_err = fig.add_subplot(gs[2, :])  # row = n_rows-1, all columns

# plot only the last two columns of errors
    ax_err.plot(np.arange(1,n+1),A_norm[:, :], marker='o', linewidth=2)
    ax_err.set_xticks(np.arange(1,n+1))
    ax_err.set_xlabel("Channel index")
    ax_err.set_ylabel("Normalized error")
    ax_err.legend(["md","#matches","rmse","mi"], loc="best")
    ax_err.set_title("Normalized errors (last two metrics)")

    plt.tight_layout()
    plt.show()
