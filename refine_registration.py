
import numpy as np
import cv2
from image_module import _to_float01
from skimage.transform import warp, AffineTransform, SimilarityTransform, rescale
from skimage.registration import optical_flow_tvl1, optical_flow_ilk, phase_cross_correlation
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as nmi

def _ncc(a, b, eps=1e-12):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + eps)
    return float(a.dot(b) / denom)


def _image_metrics(ref_img, aligned_img):
    """
    Compute MSE, NCC, gradient-NCC, SSIM after automatically resizing the
    *coarser* image to the *finer* image's shape.

    Heuristic for "finer": larger pixel grid area (H*W). The smaller one is upsampled
    to match the larger one. If shapes already match, no resize is done.

    Returns a dict with metrics + which image was resized.
    """

    rf = _to_float01(ref_img)
    al = _to_float01(aligned_img)

    Hr, Wr = rf.shape[:2]
    Ha, Wa = al.shape[:2]

    # Decide which grid is finer (more pixels). Upsample the other to match it.
    area_r = Hr * Wr
    area_a = Ha * Wa

    resized = None
    if (Hr, Wr) != (Ha, Wa):
        if area_r >= area_a:
            # ref is finer (or equal); upsample aligned -> ref shape
            al = cv2.resize(al, (Wr, Hr), interpolation=cv2.INTER_CUBIC)
            resized = "aligned_to_ref"
            eval_shape = (Hr, Wr)
        else:
            # aligned is finer; upsample ref -> aligned shape
            rf = cv2.resize(rf, (Wa, Ha), interpolation=cv2.INTER_CUBIC)
            resized = "ref_to_aligned"
            eval_shape = (Ha, Wa)
    else:
        eval_shape = (Hr, Wr)

    # MSE
    mse = float(np.mean((rf - al) ** 2))

    # NCC (zero-mean)
    ncc = _ncc(rf, al)

    # Gradient NCC
    rx = cv2.Sobel(rf, cv2.CV_32F, 1, 0, ksize=3)
    ry = cv2.Sobel(rf, cv2.CV_32F, 0, 1, ksize=3)
    wx = cv2.Sobel(al, cv2.CV_32F, 1, 0, ksize=3)
    wy = cv2.Sobel(al, cv2.CV_32F, 0, 1, ksize=3)
    gncc = 0.5 * (_ncc(rx, wx) + _ncc(ry, wy))

    # SSIM
    ssim_val = float(ssim(rf, al, data_range=1.0))

    # Mutual information (normalized)
    mi_val = float(nmi(rf, al))

    return {
        "mse": mse,
        "ncc": ncc,
        "ssim": ssim_val,
        "mi": mi_val,
        "gncc": gncc,
        "eval_shape": eval_shape,
        "resized": resized,                 # None / "aligned_to_ref" / "ref_to_aligned"
        "ref_shape": (Hr, Wr),
        "aligned_shape": (Ha, Wa),
    }
def _warp_by_flow(img01, v, u):
    H, W = img01.shape[:2]
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords = np.array([rr + v, cc + u])
    out = warp(img01, coords, output_shape=(H, W), order=1, preserve_range=True)
    return out.astype(np.float32)
def _ecc_affine(ref, mov, use_edt=False, gaussFiltSize=1, iters=500, eps=1e-6):
    """
    Affine ECC (rigid+shear+scale). If use_edt=True, run ECC on edge-distance-like
    inputs (robust multimodal); otherwise on intensities.
    Returns: aligned_moving (float32), A3 (3x3 float32), ecc_val (float or None)
    """
    # prep
    ref_f = _to_float01(ref)
    mov_f = _to_float01(mov)

    if use_edt:
        gx, gy = cv2.Sobel(ref_f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(ref_f, cv2.CV_32F, 0, 1, ksize=3)
        ref_g = cv2.normalize(np.hypot(gx, gy), None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        gx, gy = cv2.Sobel(mov_f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(mov_f, cv2.CV_32F, 0, 1, ksize=3)
        mov_g = cv2.normalize(np.hypot(gx, gy), None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        I0, I1 = ref_g, mov_g
    else:
        I0, I1 = ref_f, mov_f

    # ECC expects float32, same size
    W = np.array([[1.,0.,0.],[0.,1.,0.]], np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    ecc_val = None
    try:
        ecc_val, W = cv2.findTransformECC(I0, I1, W, motionType=cv2.MOTION_EUCLIDEAN, criteria=criteria)
        ecc_val = float(ecc_val)
    except cv2.error as e:
        print("[ECC] not run:", e)

    A3 = np.vstack([W, [0,0,1]]).astype(np.float32)
    T = AffineTransform(matrix=A3)
    H, W_ = ref_f.shape
    mov_aff = warp(mov_f, T.inverse, output_shape=(H, W_), order=1, preserve_range=True).astype(np.float32)
    return mov_aff, A3, ecc_val

def _flow_refine(ref, mov, *, flow_scale=0.5, attachment=12, tightness=0.40,
                 num_warp=7, num_iter=30, smooth_sigma=1.0):
    # prepare
    ref_f = cv2.GaussianBlur(_to_float01(ref), (3,3), 0)
    mov_f = cv2.GaussianBlur(_to_float01(mov), (3,3), 0)
    # downscale for flow
    rs = rescale(ref_f, flow_scale, anti_aliasing=False, preserve_range=True).astype(np.float32)
    ms = rescale(mov_f, flow_scale, anti_aliasing=False, preserve_range=True).astype(np.float32)
    v_s, u_s = optical_flow_tvl1(rs, ms, attachment=attachment, tightness=tightness,
                                 num_warp=num_warp, num_iter=num_iter, tol=1e-4, prefilter=False)
    H, W = ref_f.shape
    v = cv2.resize(v_s, (W, H), interpolation=cv2.INTER_LINEAR) / max(flow_scale, 1e-6)
    u = cv2.resize(u_s, (W, H), interpolation=cv2.INTER_LINEAR) / max(flow_scale, 1e-6)
    if smooth_sigma and smooth_sigma > 0:
        v = cv2.GaussianBlur(v, (0,0), smooth_sigma)
        u = cv2.GaussianBlur(u, (0,0), smooth_sigma)
    mov_warp = _warp_by_flow(mov_f, v, u)
    return mov_warp, v.astype(np.float32), u.astype(np.float32)

def refine_registration(ref_crop, tgt_crop,
                        mode="flow",          # "affine" | "flow" | "affine_then_flow" | "sift" | "sift_then_flow"
                        use_edt=False,        # if True, ECC runs on geometry-like inputs
                        flow_kwargs=None,     # dict for _flow_refine
                        ecc_kwargs=None,      # dict for _ecc_affine
                        sift_kwargs=None,     # dict for SIFT params & RANSAC
                        sift_use_gradients=False  # if True, SIFT works on gradient magnitude images
                        ):
    """
    Returns:
      aligned (float32),
      A3 (3x3 float32; identity if no affine),
      (v,u)  (None,None) unless mode uses flow.

    New modes:
      - "sift":          SIFT + RANSAC affine/similarity, returns zeros flow.
      - "sift_then_flow": SIFT first, then optical-flow refinement.
    """
    flow_kwargs = flow_kwargs or {}
    ecc_kwargs  = ecc_kwargs or {}
    sift_kwargs = dict(sift_kwargs or {})
    if "use_gradients" in sift_kwargs:
        sift_use_gradients = bool(sift_kwargs.pop("use_gradients"))

    # ---- helpers ----
    def _zeros_flow_like(img):
        H, W = img.shape[:2]
        z = np.zeros((H, W), dtype=np.float32)
        return (z, z)

    def _sift_feature_image(img, use_gradients):
        x = _to_float01(img)
        if use_gradients:
            gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            mmax = float(np.max(mag))
            if mmax > 1e-8:
                mag = mag / mmax
            return (mag * 255.0).astype(np.uint8)
        return (x * 255.0).astype(np.uint8)

    def _sift_affine(ref_img, mov_img,
                     nfeatures=2000, ratio=0.75, ransac_thresh=6.0, max_iters=2000, conf=0.99,
                     border_mode=cv2.BORDER_CONSTANT, interp=cv2.INTER_LINEAR, use_gradients=False):
        """
        Estimate an affine (similarity) transform using SIFT + RANSAC,
        then warp mov_img into ref_img frame. Falls back to identity if it fails.
        Returns: aligned(float32), A3(3x3 float32)
        """
        # SIFT
        ref_u8 = _sift_feature_image(ref_img, use_gradients=use_gradients)
        mov_u8 = _sift_feature_image(mov_img, use_gradients=use_gradients)
        # cv2.imwrite("ref_img.png", ref_img)
        # cv2.imwrite("mov_img.png", mov_img)
        # # --- visualize inputs (optional) ---
        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.title("SIFT ref_u8")
        # plt.imshow(ref_u8, cmap="gray")
        # plt.axis("off")

        # plt.figure()
        # plt.title("SIFT mov_u8")
        # plt.imshow(mov_u8, cmap="gray")
        # plt.axis("off")

        sift = cv2.SIFT_create(nfeatures=nfeatures)
        k1, d1 = sift.detectAndCompute(ref_u8, None)
        k2, d2 = sift.detectAndCompute(mov_u8, None)
        if d1 is None or d2 is None or len(k1) < 3 or len(k2) < 3:
            print("sift fail 1")
            A3 = np.eye(3, dtype=np.float32)
            aligned = mov_img.astype(np.float32)
            metrics = {
                "num_matches": 0,
                "num_inliers": 0,
                "inlier_ratio": 0.0,
                "mean_inlier_error": np.inf,
                "median_inlier_error": np.inf,
                "normalized_mean_error": np.inf,
            }
            return aligned, A3, metrics

        # match + ratio test
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = bf.knnMatch(d2, d1, k=2)  # mov -> ref
        good = []
        for m, n in knn:
            if m.distance < ratio * n.distance:
                good.append(m)
        if len(good) < 3:
            print("sift fail 2")
            A3 = np.eye(3, dtype=np.float32)
            aligned = mov_img.astype(np.float32)
            metrics = {
                "num_matches": len(good),
                "num_inliers": 0,
                "inlier_ratio": 0.0,
                "mean_inlier_error": np.inf,
                "median_inlier_error": np.inf,
                "normalized_mean_error": np.inf,
            }
            return aligned, A3, metrics

        # --- visualize matches (optional) ---
        # Draw top N good matches (say 50)
        drawN = min(50, len(good))
        match_vis = cv2.drawMatches(
            mov_u8, [k2[m.queryIdx] for m in good[:drawN]],
            ref_u8, [k1[m.trainIdx] for m in good[:drawN]],
            [cv2.DMatch(i, i, 0) for i in range(drawN)],
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        plt.figure()
        plt.title(f"SIFT matches (showing {drawN} / {len(good)} good)")
        plt.imshow(match_vis[..., ::-1])   # BGR→RGB
        plt.axis("off")
        plt.show()

        pts_m = np.float32([k2[m.queryIdx].pt for m in good])  # moving
        pts_r = np.float32([k1[m.trainIdx].pt for m in good])  # reference

        # robust affine/similarity (2x3) with RANSAC
        # A2, inliers = cv2.estimateAffinePartial2D(pts_m, pts_r, method =cv2.LMEDS)
        # print("pts_m shape:", pts_m.shape, "pts_r shape:", pts_r.shape)
        # print("dtype:", pts_m.dtype, pts_r.dtype)
        # print("pts_m[:5]:\n", pts_m[:5])
        # print("pts_r[:5]:\n", pts_r[:5])
        A2, inliers = cv2.estimateAffinePartial2D(
            pts_m, pts_r,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=max_iters,
            confidence=conf
        )
        # print(A2)

        H, W = ref_img.shape[:2]
        if A2 is None or inliers is None:
            print("sift fail 3")
            A3 = np.eye(3, dtype=np.float32)
            aligned = mov_img.astype(np.float32)
            metrics = {
                "num_matches": len(good),
                "num_inliers": 0,
                "inlier_ratio": 0.0,
                "mean_inlier_error": np.inf,
                "median_inlier_error": np.inf,
                "normalized_mean_error": np.inf,
            }
            return aligned, A3, metrics

        # ---------- error metric from SIFT matches + affine model ----------
        inlier_mask = inliers.ravel().astype(bool)
        num_matches = len(good)
        num_inliers = int(inlier_mask.sum())
        inlier_ratio = num_inliers / max(num_matches, 1)

        # Reproject moving keypoints under A2
        proj = (A2[:, :2] @ pts_m.T).T + A2[:, 2]  # (N,2)
        errors = np.linalg.norm(proj - pts_r, axis=1)  # pixel reprojection error

        inlier_errors = errors[inlier_mask]
        if inlier_errors.size > 0:
            mean_inlier_error = float(inlier_errors.mean())
            median_inlier_error = float(np.median(inlier_errors))
        else:
            print("sift found 0 inliers")
            mean_inlier_error = np.inf
            median_inlier_error = np.inf

        # Normalize w.r.t. image diagonal
        diag = np.sqrt(H**2 + W**2)
        normalized_mean_error = float(mean_inlier_error / diag) if np.isfinite(mean_inlier_error) else np.inf

        metrics = {
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "inlier_ratio": inlier_ratio,
            "mean_inlier_error": mean_inlier_error,
            "median_inlier_error": median_inlier_error,
            "normalized_mean_error": normalized_mean_error,
        }

        # warp moving into reference frame
        aligned = cv2.warpAffine(mov_img, A2, (W, H), flags=interp, borderMode=border_mode)

        # promote to 3x3
        A3 = np.eye(3, dtype=np.float32)
        A3[:2, :3] = A2.astype(np.float32)

        if abs(np.linalg.det(A3)) < 1e-6:
            print(f"transformation degenerate")

        return aligned.astype(np.float32), A3, metrics

    # ---- existing modes ----
    if mode == "affine":
        aligned, A3, ecc_val = _ecc_affine(ref_crop, tgt_crop, use_edt=use_edt, **ecc_kwargs)
        metrics = _image_metrics(ref_crop, aligned)
        metrics.update({
            "mode": "affine",
            "ecc_score": float(ecc_val) if ecc_val is not None else float("nan"),
        })
        return aligned, A3, _zeros_flow_like(ref_crop), metrics

    if mode == "flow":
        aligned, v, u = _flow_refine(ref_crop, tgt_crop, **flow_kwargs)
        metrics = _image_metrics(ref_crop, aligned)
        metrics["mode"] = "flow"
        return aligned, np.eye(3, dtype=np.float32), (v, u), metrics

    if mode == "affine_then_flow":
        mov_aff, A3, ecc_val = _ecc_affine(ref_crop, tgt_crop, use_edt=use_edt, **ecc_kwargs)
        affine_metrics = _image_metrics(ref_crop, mov_aff)
        affine_metrics.update({
            "ecc_score": float(ecc_val) if ecc_val is not None else float("nan"),
        })

        aligned, v, u = _flow_refine(ref_crop, mov_aff * 1.0, **flow_kwargs)
        final_metrics = _image_metrics(ref_crop, aligned)

        metrics = {
            "mode": "affine_then_flow",
            "affine": affine_metrics,
            "final": final_metrics,
        }
        return aligned, A3, (v, u), metrics

    if mode == "sift":
        aligned, A3, sift_metrics = _sift_affine(
            ref_crop, tgt_crop, use_gradients=sift_use_gradients, **sift_kwargs
        )
        metrics = _image_metrics(ref_crop, aligned)
        metrics = metrics | sift_metrics
        metrics.update({
            "mode": "affine",
        })
        return aligned, A3, _zeros_flow_like(ref_crop), metrics

    if mode == "sift_then_flow":
        mov_aff, A3, sift_metrics = _sift_affine(
            ref_crop, tgt_crop, use_gradients=sift_use_gradients, **sift_kwargs
        )
        affine_img_metrics = _image_metrics(ref_crop, mov_aff)

        aligned, v, u = _flow_refine(ref_crop, mov_aff * 1.0, **flow_kwargs)
        final_img_metrics = _image_metrics(ref_crop, aligned)

        metrics = {
            "mode": "sift_then_flow",
            "affine": {
                "image": affine_img_metrics,
                "sift": sift_metrics,
            },
            "final": final_img_metrics,
        }
        return aligned, A3, (v, u), metrics

    raise ValueError("mode must be 'affine', 'flow', 'affine_then_flow', 'sift', or 'sift_then_flow'")
