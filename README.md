# XRF / Image Registration GUI

A desktop GUI for registering two images (XRF element maps and/or simple
images). 

---

## Run

### Windows
Double-click `run_gui.bat`, or run with a Python that has the dependencies
(the Anaconda base interpreter here already does):

```
python registration_gui.py
```

### Linux / macOS
```
# 1) system Python needs Tk support (once):
#    Debian/Ubuntu:  sudo apt install python3-tk
#    Fedora:         sudo dnf install python3-tkinter
# 2) install the Python packages into the SAME interpreter you launch with:
<python> -m pip install --user -r requirements.txt
# 3) launch:
<python> registration_gui.py          # or:  ./run_gui.sh
```

`registration_gui.py` adds its own folder to `sys.path`, so it can be launched
from anywhere and still finds the backend and helper modules. All paths in the
code are relative/portable; nothing is hardcoded to a machine.

> Tip: install and launch with the **same** `python`. The most common error
> ("No module named 'cv2'") means the interpreter you ran is not the one pip
> installed into.

---

## What to transfer to another computer

Copy the **whole `image_registration_gui/` folder** (all `.py` files plus
`requirements.txt`). Data files are chosen at runtime, so they need not sit next
to the code. 

---

## Dependencies

Python 3.9+ with Tk, plus (`requirements.txt`):
`numpy, opencv-python, h5py, scikit-image, pandas, scipy, matplotlib, tifffile`.
(`tkinter` ships with conda/standard Python; on Linux install the OS `tk`
package if missing. On headless nodes use `opencv-python-headless`.)

---

## Files

| file | purpose |
|------|---------|
| `registration_gui.py` | Tkinter GUI (front end) |
| `registration_core.py` | backend: loading, PC1 fusion, SIFT, match-template, metrics, overlay |
| `image_module.py`, `hotspot_filter.py`, `register_multiscale.py`, `pca_xrf_channels.py`, `pca_xrf_channels_sc.py`, `xrf_channel_selection.py`, `refine_registration.py` | helper modules |
| `requirements.txt` | pip dependency list |
| `run_gui.bat` / `run_gui.sh` | launchers (Windows / Linux-macOS) |

---

## Layout

* **Loading** (top-left, ~2/3 width × 2/3 height): Reference and Target panels
  side by side.
* **Method** (below loading): method choice and parameters.
* **Result** (right, full height, ~1/3 width): overlay + metrics.
* **Messages / Errors** console pinned at the bottom.

All pane boundaries are draggable to resize.

---

## Loading a source (per panel)

1. **Browse...** an `.h5` file or a simple image (`tif/tiff/png/jpg/bmp`).
2. For H5:
   * **Dataset key**: `XRF_fits`, `XRF_roi`, or `XRF_roi_plus`.
   * **Channel**: pick a single channel, **or**
   * **Use PC1 fusion** to build a PC1 image from several channels:
     * **Channel pool** (default `elem+scatter` = elemental channels + the two
       scattering channels s_i / s_e, resolved per file; also accepts `all`,
       `0,1,2`, `0:20:2`).
     * **Include channels** rule (default **Top N**, N = 3):
       * *Top N*
       * *Above 1/sqrt(N)* (N = number of channels in the pool)
       * *Above half the highest loading*
       * *Manually set* (comma-separated original channel indices)
     * The channels actually used (with PC1 loadings) are printed in the console
       and listed in the result panel.
3. **Manual preprocess**: Flip X, Flip Y, Rotate 90 (×K), **Flip contrast**
   (invert intensities), and **Set crop** / **Clear crop** (drag a rectangle on
   the live preview). The image **size (H×W)** is shown under the preview.
4. **Display / colour scale**: vmin / vmax sliders, **Log scale** toggle, and
   **Reset range**. The preview updates **live** as you change any control.

---

## Method & parameters

* **Standard preprocess** (checkbox): the notebook `preprocess` pipeline —
  NaN cleanup → min-max to uint8 → hotspot removal (blur-fill) → re-normalize →
  histogram equalization.
* **SIFT**: Lowe's ratio, RANSAC confidence, RANSAC reproj threshold.
* **Match Template** (brute-force scale + rotation search, `TM_CCOEFF_NORMED`):
  scale min/max/steps, angle min/max/steps, and on rotation **Pad (keep all)**
  or **Cut (keep size)**. The overlay and reported lower-left position account
  for the pad/cut choice.

Click **Run registration** (runs in a background thread; a progress bar shows
match-template progress).

---

## Result

* Overlay of the target placed in the reference coordinate frame.
* **Scale (tgt→ref)** and **Rotation (deg)** — same convention for both methods.
* **Size relation** — coarse/fine statement inferred from the scale.
* **Lower-left (x,y)** — the target's lower-left corner in reference pixels.
* **Standard metrics** (in order): `ncc, rmse, mi, ssim, gncc`.
* **PCA-based metrics**: `pca_cosine`, `pca_pearson`. Computed by applying the
  transform to **all common element + scattering channels** of the two H5
  datasets, taking the PC1 loading array of the target stack and of the
  registered-reference stack, then comparing them (cosine + Pearson). The exact
  channels used are listed. Skipped if either source is not H5.
* For **SIFT** also: keypoints (ref/tgt), matches, good matches, RANSAC inliers.

---

## Notes / caveats

* The PCA-based metric needs both sources to be H5 (multi-channel). With a
  simple image on either side it is skipped.
* "No module named 'cv2'" on a new machine = wrong interpreter; see the Run
  section.
