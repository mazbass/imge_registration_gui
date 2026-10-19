import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from register_multiscale import preprocess


def resolve_channels(ch_arg, channel_names, n_channels):
    if ch_arg is None or ch_arg.lower() == "all":
        return list(range(n_channels))
    parts = [c.strip() for c in ch_arg.split(",") if c.strip()]
    out = []
    for p in parts:
        if p.isdigit():
            idx = int(p)
        else:
            idx = None
            for i, name in enumerate(channel_names or []):
                if name == p or name.lower() == p.lower():
                    idx = i
                    break
            if idx is None:
                raise ValueError(f"Unknown channel name: {p}")
        if idx < 0 or idx >= n_channels:
            raise ValueError(f"Channel out of range: {idx}")
        out.append(idx)
    return out


def pca_fit(X, center=True, scale=False):
    if center:
        mean = X.mean(axis=0, keepdims=True)
        X = X - mean
    else:
        mean = np.zeros((1, X.shape[1]), dtype=X.dtype)
    if scale:
        std = X.std(axis=0, keepdims=True) + 1e-8
        X = X / std
    else:
        std = np.ones((1, X.shape[1]), dtype=X.dtype)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n = X.shape[0]
    var = (S ** 2) / max(n - 1, 1)
    var_ratio = var / (var.sum() + 1e-12)
    return {
        "mean": mean.squeeze(0),
        "std": std.squeeze(0),
        "components": Vt,
        "singular_values": S,
        "explained_variance": var,
        "explained_variance_ratio": var_ratio,
    }


def run_pca(
    h5_path,
    dataset="MAPS/XRF_fits",
    channels="all",
    n_components=3,
    sample=0,
    center=False,
    scale=False,
    output=None,
    plot=False,
    preprocess_enabled=True,
):
    with h5py.File(h5_path, "r") as h5f:
        if dataset not in h5f:
            raise KeyError(f"Dataset not found: {dataset}")
        data = h5f[dataset]
        if data.ndim != 3:
            raise ValueError("Expected dataset shape (C, H, W).")
        n_channels, H, W = data.shape

        channel_names = None
        if "MAPS/channel_names" in h5f:
            raw = h5f["MAPS/channel_names"][...]
            channel_names = [n.decode("utf-8") for n in raw]

        channels = resolve_channels(channels, channel_names, n_channels)
        channels = sorted(channels)
        stack = data[channels, :, :]

    C = stack.shape[0]
    if preprocess_enabled:
        proc_stack = np.empty_like(stack, dtype=np.float32)
        for i in range(C):
            proc_stack[i] = preprocess(stack[i])
        stack = proc_stack
    else:
        stack = stack.astype(np.float32)

    X = stack.reshape(C, -1).T  # (Npixels, C)

    if sample and sample > 0 and sample < X.shape[0]:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], size=sample, replace=False)
        X_fit = X[idx]
    else:
        X_fit = X

    model = pca_fit(X_fit, center=center or scale, scale=scale)

    k = min(n_components, C)
    mean = model["mean"]
    std = model["std"]

    X_proj = X.astype(np.float32)
    X_proj = X_proj - mean
    X_proj = X_proj / std
    scores = X_proj @ model["components"][:k].T
    score_imgs = scores.T.reshape(k, H, W)

    print("PCA explained variance ratio:")
    for i in range(k):
        print(f"  PC{i+1}: {model['explained_variance_ratio'][i]:.6f}")

    if plot:
        ncols = min(k, 3)
        nrows = int(np.ceil(k / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.ravel()
        for i in range(k):
            ax = axes[i]
            im = ax.imshow(score_imgs[i], cmap="gray")
            ax.set_title(f"PC{i+1}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for j in range(k, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        # Plot first component loadings across channels.
        comp0 = model["components"][0]
        fig2, ax2 = plt.subplots(figsize=(max(6, 0.5 * len(comp0)), 3))
        x = np.arange(len(comp0))
        ax2.bar(x, comp0, color="tab:blue")
        if channel_names:
            labels = [channel_names[i] for i in channels]
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=45, ha="right")
        else:
            ax2.set_xlabel("Channel index")
        ax2.set_title("PC1 loadings")
        ax2.set_ylabel("Loading")
        plt.tight_layout()
        plt.show()

    if output:
        out_path = output
        if os.path.exists(out_path):
            raise FileExistsError(f"Output already exists: {out_path}")
        np.savez(
            out_path,
            channels=np.array(channels),
            channel_names=np.array(channel_names or [], dtype=object),
            mean=mean,
            std=std,
            components=model["components"],
            explained_variance=model["explained_variance"],
            explained_variance_ratio=model["explained_variance_ratio"],
            score_images=score_imgs,
        )

    return score_imgs, model


def main():
    parser = argparse.ArgumentParser(
        description="Stack XRF channels and run PCA via SVD."
    )
    parser.add_argument("--h5", required=True, help="Path to H5 file.")
    parser.add_argument(
        "--dataset",
        default="MAPS/XRF_fits",
        help="Dataset key with shape (C,H,W).",
    )
    parser.add_argument(
        "--channels",
        default="all",
        help="Comma-separated channel indices or names, or 'all'.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Number of principal components to project to.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Randomly sample this many pixels for PCA fit (0 = use all).",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center features before PCA.",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Scale features to unit variance before PCA.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output .npz to save PCA model and component images.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plots of component score images.",
    )
    parser.add_argument(
        "--no-preprocess",
        dest="preprocess",
        action="store_false",
        help="Disable per-channel preprocessing before PCA.",
    )
    args = parser.parse_args()
    if not hasattr(args, "preprocess"):
        args.preprocess = True

    score_imgs, _ = run_pca(
        args.h5,
        dataset=args.dataset,
        channels=args.channels,
        n_components=args.n_components,
        sample=args.sample,
        center=args.center,
        scale=args.scale,
        output=args.output,
        plot=args.plot,
        preprocess_enabled=args.preprocess,
    )
    return score_imgs


if __name__ == "__main__":
    main()
