"""
registration_gui.py
====================

Tkinter GUI for XRF / optical image registration, built on registration_core.py.

Features
--------
* Two image sources (Reference, Target), each one of:
    - H5 file: choose dataset key (XRF_fits / XRF_roi / XRF_roi_plus),
      pick a channel OR enable PC1 fusion.
    - Simple image: tiff / png / jpg / bmp.
* PC1 fusion channel-inclusion strategies:
    - Top N
    - Above 1/sqrt(N)   (N = number of channels in the pool)
    - Above half the highest loading
    - Manually set
* Manual preprocess: flip X, flip Y, 90-degree rotations, interactive crop.
* Standard notebook preprocess toggle (hotspot removal + CLAHE/equalise).
* Method: SIFT or Match-Template.
    - SIFT params: Lowe's ratio, RANSAC confidence (+ reproj threshold).
    - Match-template params: scale range/step, rotation range/step,
      and pad-vs-cut on rotation.
* Results: overlay of target in the reference coordinate frame, metrics
  (standard + PCA-based), scale, rotation; for SIFT also keypoints, matches,
  good matches and RANSAC inliers.
* Bottom console that shows error / status messages.

Run with the interpreter that has cv2, h5py, skimage, pandas, matplotlib, PyQt/Tk:
    python registration_gui.py
"""

import os
import sys
import traceback
import threading
import queue

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.widgets import RectangleSelector

# make sure local modules are importable even if launched from elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registration_core as rc


SELECTION_MODES = [
    ("Top N", "top_n"),
    ("Above 1/sqrt(N)", "inv_sqrt_n"),
    ("Above half the highest loading", "half_max"),
    ("Manually set", "manual"),
]
MODE_LABEL_TO_KEY = {lbl: key for lbl, key in SELECTION_MODES}
MODE_KEY_TO_LABEL = {key: lbl for lbl, key in SELECTION_MODES}


# =======================================================================
#  Scrollable container (vertical + horizontal scrollbars)
# =======================================================================
class ScrollableFrame(ttk.Frame):
    """A frame whose `.body` child scrolls with attached scrollbars."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.body = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>",
                       lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # mouse wheel scrolling when the pointer is over the canvas
        self.canvas.bind("<Enter>", lambda e: self._bind_wheel(True))
        self.canvas.bind("<Leave>", lambda e: self._bind_wheel(False))

    def _on_canvas_configure(self, event):
        # make inner frame at least as wide as the canvas
        self.canvas.itemconfigure(self._win, width=max(event.width, self.body.winfo_reqwidth()))

    def _bind_wheel(self, on):
        if on:
            self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        else:
            self.canvas.unbind_all("<MouseWheel>")

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


# =======================================================================
#  One image-source panel (Reference or Target)
# =======================================================================
class SourcePanel(ttk.LabelFrame):
    def __init__(self, master, title, log_fn, **kw):
        super().__init__(master, text=title, **kw)
        self.log = log_fn
        self._crop = None            # (x0,y0,x1,y1) in oriented coords
        self._preview_raw = None     # last previewed raw image
        self._rect_selector = None

        self.var_path = tk.StringVar()
        self.var_kind = tk.StringVar(value="h5")
        self.var_key = tk.StringVar(value="XRF_fits")
        self.var_channel = tk.StringVar(value="0")
        self.var_use_pc1 = tk.BooleanVar(value=False)

        self.var_pool = tk.StringVar(value="elem+scatter")
        self.var_mode = tk.StringVar(value=MODE_KEY_TO_LABEL["top_n"])
        self.var_topn = tk.StringVar(value="3")
        self.var_manual = tk.StringVar(value="0.15")   # manual loading threshold

        self.var_flipx = tk.BooleanVar(value=False)
        self.var_flipy = tk.BooleanVar(value=False)
        self.var_rotk = tk.IntVar(value=0)
        self.var_flip_contrast = tk.BooleanVar(value=False)

        # display / colour-scale controls
        self.var_vmin = tk.DoubleVar(value=0.0)
        self.var_vmax = tk.DoubleVar(value=1.0)
        self.var_log = tk.BooleanVar(value=False)
        self._data_lo = 0.0           # data range of current preview
        self._data_hi = 1.0
        self._im = None               # matplotlib image handle
        self._update_job = None       # debounce handle for live update
        self._crop_patch = None

        self._channel_names = []
        self._build()

    # ---------------- UI ----------------
    def _build(self):
        r = 0
        frm = ttk.Frame(self)
        frm.grid(row=r, column=0, sticky="ew", padx=4, pady=2)
        frm.columnconfigure(1, weight=1)

        ttk.Button(frm, text="Browse...", command=self._browse).grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_path).grid(row=0, column=1, sticky="ew", padx=4)
        r += 1

        kf = ttk.Frame(self)
        kf.grid(row=r, column=0, sticky="ew", padx=4, pady=2)
        ttk.Label(kf, text="Dataset key:").grid(row=0, column=0, sticky="w")
        self.cmb_key = ttk.Combobox(kf, textvariable=self.var_key, width=14,
                                    values=list(rc.XRF_KEYS), state="readonly")
        self.cmb_key.grid(row=0, column=1, sticky="w", padx=4)
        self.cmb_key.bind("<<ComboboxSelected>>", lambda e: self._refresh_channels())
        self._bind_combobox_arrows(self.cmb_key, self._refresh_channels)

        ttk.Label(kf, text="Channel:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.cmb_ch = ttk.Combobox(kf, textvariable=self.var_channel, width=22)
        self.cmb_ch.grid(row=0, column=3, sticky="w", padx=4)
        self._bind_combobox_arrows(self.cmb_ch)   # live update via var_channel trace
        r += 1

        ttk.Checkbutton(self, text="Use PC1 fusion (H5 only)",
                        variable=self.var_use_pc1,
                        command=self._toggle_pc1).grid(row=r, column=0, sticky="w", padx=6)
        r += 1

        # PC1 options block
        self.pc1_frame = ttk.LabelFrame(self, text="PC1 fusion options")
        self.pc1_frame.grid(row=r, column=0, sticky="ew", padx=4, pady=2)
        pf = self.pc1_frame
        ttk.Label(pf, text="Channel pool:").grid(row=0, column=0, sticky="w")
        ttk.Entry(pf, textvariable=self.var_pool, width=18).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(pf, text="(elem+scatter / all / 0,1,2 / 0:20:2)").grid(row=0, column=2, sticky="w")

        ttk.Label(pf, text="Include channels:").grid(row=1, column=0, sticky="w")
        self.cmb_mode = ttk.Combobox(pf, textvariable=self.var_mode, width=26,
                                     values=[lbl for lbl, _ in SELECTION_MODES],
                                     state="readonly")
        self.cmb_mode.grid(row=1, column=1, columnspan=2, sticky="w", padx=4)
        self.cmb_mode.bind("<<ComboboxSelected>>", lambda e: self._toggle_mode())
        self._bind_combobox_arrows(self.cmb_mode, self._toggle_mode)

        self.lbl_topn = ttk.Label(pf, text="N =")
        self.lbl_topn.grid(row=2, column=0, sticky="e")
        self.ent_topn = ttk.Entry(pf, textvariable=self.var_topn, width=6)
        self.ent_topn.grid(row=2, column=1, sticky="w", padx=4)

        self.lbl_manual = ttk.Label(pf, text="Manual threshold:")
        self.lbl_manual.grid(row=3, column=0, sticky="e")
        self.ent_manual = ttk.Entry(pf, textvariable=self.var_manual, width=8)
        self.ent_manual.grid(row=3, column=1, sticky="w", padx=4)
        ttk.Label(pf, text="(keep loading >= thr; signed)").grid(row=3, column=2, sticky="w")
        r += 1

        # orientation
        of = ttk.LabelFrame(self, text="Manual preprocess")
        of.grid(row=r, column=0, sticky="ew", padx=4, pady=2)
        ttk.Checkbutton(of, text="Flip X", variable=self.var_flipx,
                        command=self._schedule_update).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(of, text="Flip Y", variable=self.var_flipy,
                        command=self._schedule_update).grid(row=0, column=1, sticky="w")
        ttk.Label(of, text="Rotate 90 x").grid(row=0, column=2, sticky="e", padx=(10, 0))
        ttk.Spinbox(of, from_=0, to=3, width=3, textvariable=self.var_rotk,
                    command=self._schedule_update).grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(of, text="Flip contrast", variable=self.var_flip_contrast,
                        command=self._schedule_update).grid(row=0, column=4, sticky="w", padx=(10, 0))
        ttk.Button(of, text="Set crop", command=self._enable_crop).grid(row=0, column=5, padx=6)
        ttk.Button(of, text="Clear crop", command=self._clear_crop).grid(row=0, column=6, padx=4)
        self.lbl_crop = ttk.Label(of, text="crop: full")
        self.lbl_crop.grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 0))
        self.lbl_size = ttk.Label(of, text="size: -")
        self.lbl_size.grid(row=1, column=4, columnspan=3, sticky="w", pady=(2, 0))
        r += 1

        # colour-scale controls
        cf = ttk.LabelFrame(self, text="Display / colour scale")
        cf.grid(row=r, column=0, sticky="ew", padx=4, pady=2)
        cf.columnconfigure(1, weight=1)
        ttk.Label(cf, text="vmin").grid(row=0, column=0, sticky="w")
        self.sld_vmin = ttk.Scale(cf, from_=0.0, to=1.0, variable=self.var_vmin,
                                  command=lambda v: self._on_scale_change())
        self.sld_vmin.grid(row=0, column=1, sticky="ew", padx=4)
        self.lbl_vmin = ttk.Label(cf, text="0.00", width=8)
        self.lbl_vmin.grid(row=0, column=2, sticky="e")
        ttk.Label(cf, text="vmax").grid(row=1, column=0, sticky="w")
        self.sld_vmax = ttk.Scale(cf, from_=0.0, to=1.0, variable=self.var_vmax,
                                  command=lambda v: self._on_scale_change())
        self.sld_vmax.grid(row=1, column=1, sticky="ew", padx=4)
        self.lbl_vmax = ttk.Label(cf, text="1.00", width=8)
        self.lbl_vmax.grid(row=1, column=2, sticky="e")
        ttk.Checkbutton(cf, text="Log scale", variable=self.var_log,
                        command=self._schedule_update).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(cf, text="Reset range", command=self._reset_range).grid(row=2, column=2, sticky="e")
        r += 1

        # preview canvas (expands to fill the panel)
        self.fig = Figure(figsize=(3.2, 2.6), dpi=90)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=r, column=0, sticky="nsew", padx=4, pady=2)
        self.rowconfigure(r, weight=1)
        self.columnconfigure(0, weight=1)

        # live-update triggers on the data-affecting inputs
        self.var_key.trace_add("write", lambda *a: self._schedule_update())
        self.var_channel.trace_add("write", lambda *a: self._schedule_update())
        self.var_use_pc1.trace_add("write", lambda *a: self._schedule_update())
        self.var_pool.trace_add("write", lambda *a: self._schedule_update())
        self.var_mode.trace_add("write", lambda *a: self._schedule_update())
        self.var_topn.trace_add("write", lambda *a: self._schedule_update())
        self.var_manual.trace_add("write", lambda *a: self._schedule_update())

        self._toggle_pc1()
        self._toggle_mode()

    # ---------------- callbacks ----------------
    def _bind_combobox_arrows(self, combo, on_change=None):
        """
        Enable Up/Down (and PageUp/PageDown) arrow keys to step through a
        combobox's values, even when it is readonly, firing on_change after.
        """
        def step(delta):
            values = list(combo.cget("values"))
            if not values:
                return "break"
            cur = combo.get()
            try:
                i = values.index(cur)
            except ValueError:
                i = 0
            i = max(0, min(len(values) - 1, i + delta))
            combo.set(values[i])
            combo.event_generate("<<ComboboxSelected>>")
            if on_change is not None:
                on_change()
            return "break"

        combo.bind("<Up>", lambda e: step(-1))
        combo.bind("<Down>", lambda e: step(+1))
        combo.bind("<Prior>", lambda e: step(-1))   # PageUp
        combo.bind("<Next>", lambda e: step(+1))    # PageDown
        # make it focusable by click so the keys apply
        combo.bind("<Button-1>", lambda e: combo.focus_set(), add="+")
        # mouse wheel over the combobox also steps
        combo.bind("<MouseWheel>", lambda e: step(-1 if e.delta > 0 else +1))

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select image / H5 file",
            filetypes=[
                ("All supported", "*.h5 *.hdf5 *.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                ("H5 files", "*.h5 *.hdf5"),
                ("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.var_path.set(path)
        ext = os.path.splitext(path)[1].lower()
        if ext in rc.H5_EXTS:
            self.var_kind.set("h5")
            try:
                keys = rc.list_h5_xrf_keys(path)
                if keys:
                    self.cmb_key["values"] = keys
                    if self.var_key.get() not in keys:
                        self.var_key.set(keys[0])
                self._refresh_channels()
                self.log(f"Loaded H5: {os.path.basename(path)} keys={keys}")
            except Exception as e:
                self.log(f"ERROR reading H5 keys: {e}")
        else:
            self.var_kind.set("image")
            self.var_use_pc1.set(False)
            self.log(f"Loaded image: {os.path.basename(path)}")
        self._toggle_pc1()
        self.update_preview()

    def _refresh_channels(self):
        if self.var_kind.get() != "h5":
            return
        try:
            names = rc.read_channel_names(self.var_path.get(), self.var_key.get())
            self._channel_names = names
            vals = [f"{i}: {n}" for i, n in enumerate(names)]
            self.cmb_ch["values"] = vals
            if vals:
                cur = self.var_channel.get()
                idx = self._parse_channel_index(cur)
                if idx is None or idx >= len(vals):
                    idx = 0
                self.var_channel.set(vals[idx])
        except Exception as e:
            self.log(f"ERROR reading channels: {e}")

    @staticmethod
    def _parse_channel_index(s):
        s = str(s).strip()
        if not s:
            return None
        try:
            return int(s.split(":")[0])
        except Exception:
            return None

    def _toggle_pc1(self):
        is_h5 = self.var_kind.get() == "h5"
        pc1 = self.var_use_pc1.get() and is_h5
        state = "normal" if pc1 else "disabled"
        for child in self.pc1_frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass
        # channel combobox disabled when PC1 is on
        self.cmb_ch.configure(state=("disabled" if pc1 else "normal"))
        self.cmb_key.configure(state=("readonly" if is_h5 else "disabled"))
        if pc1:
            self._toggle_mode()

    def _toggle_mode(self):
        key = MODE_LABEL_TO_KEY.get(self.var_mode.get(), "inv_sqrt_n")
        topn_on = key == "top_n"
        manual_on = key == "manual"
        self.ent_topn.configure(state=("normal" if topn_on else "disabled"))
        self.lbl_topn.configure(state=("normal" if topn_on else "disabled"))
        self.ent_manual.configure(state=("normal" if manual_on else "disabled"))
        self.lbl_manual.configure(state=("normal" if manual_on else "disabled"))

    # ---------------- crop ----------------
    def _enable_crop(self):
        if self._preview_raw is None:
            self.update_preview()
        if self._preview_raw is None:
            self.log("Nothing to crop: preview failed.")
            return
        self.log("Crop mode: drag a rectangle on the preview; release to set.")

        def onselect(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if None in (x0, y0, x1, y1):
                return
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            self._crop = (x0, y0, x1, y1)
            self.lbl_crop.configure(
                text=f"crop: x[{int(x0)}:{int(x1)}] y[{int(y0)}:{int(y1)}]"
            )
            self.log(f"{self['text']} crop set to x[{int(x0)}:{int(x1)}] y[{int(y0)}:{int(y1)}]")

        try:
            self._rect_selector = RectangleSelector(
                self.ax, onselect, useblit=True, button=[1],
                minspanx=3, minspany=3, spancoords="pixels", interactive=True,
            )
        except TypeError:
            # older matplotlib signature
            self._rect_selector = RectangleSelector(
                self.ax, onselect, drawtype="box", useblit=True, button=[1],
                minspanx=3, minspany=3, spancoords="pixels", interactive=True,
            )
        self.canvas.draw_idle()

    def _clear_crop(self):
        self._crop = None
        self.lbl_crop.configure(text="crop: full")
        if self._rect_selector is not None:
            try:
                self._rect_selector.set_active(False)
            except Exception:
                pass
            self._rect_selector = None
        self.update_preview()

    # ---------------- model ----------------
    def build_source(self):
        """Return a configured rc.ImageSource (without crop baked only when previewed)."""
        path = self.var_path.get().strip()
        if not path:
            raise ValueError(f"{self['text']}: no file selected")
        kind = "h5" if os.path.splitext(path)[1].lower() in rc.H5_EXTS else "image"
        try:
            manual_threshold = float(self.var_manual.get())
        except (ValueError, tk.TclError):
            manual_threshold = 0.15
        src = rc.ImageSource(
            path=path,
            kind=kind,
            dataset_key=self.var_key.get(),
            channel=self._parse_channel_index(self.var_channel.get()) or 0,
            use_pc1=bool(self.var_use_pc1.get()) and kind == "h5",
            pc1_pool_selection=self.var_pool.get().strip() or "all",
            pc1_selection_mode=MODE_LABEL_TO_KEY.get(self.var_mode.get(), "inv_sqrt_n"),
            pc1_top_n=int(self.var_topn.get() or 3),
            pc1_manual_threshold=manual_threshold,
            flip_x=bool(self.var_flipx.get()),
            flip_y=bool(self.var_flipy.get()),
            rot_k=int(self.var_rotk.get()),
            flip_contrast=bool(self.var_flip_contrast.get()),
            crop=self._crop,
        )
        return src

    # ---------------- live preview ----------------
    def _schedule_update(self, *_):
        """Debounced full redraw so rapid control changes don't re-read H5 repeatedly."""
        if not self.var_path.get().strip():
            return
        if self._update_job is not None:
            try:
                self.after_cancel(self._update_job)
            except Exception:
                pass
        self._update_job = self.after(250, self.update_preview)

    def _reset_range(self):
        self.var_vmin.set(self._data_lo)
        self.var_vmax.set(self._data_hi)
        self._apply_clim()

    def _on_scale_change(self):
        # keep vmin <= vmax
        if self.var_vmin.get() > self.var_vmax.get():
            self.var_vmin.set(self.var_vmax.get())
        self._apply_clim()

    def _apply_clim(self):
        """Lightweight update: colour limits + log scale only (no data reload)."""
        if self._im is None:
            return
        import matplotlib.colors as mcolors
        lo = float(self.var_vmin.get())
        hi = float(self.var_vmax.get())
        if hi <= lo:
            hi = lo + 1e-6
        self.lbl_vmin.configure(text=f"{lo:.3g}")
        self.lbl_vmax.configure(text=f"{hi:.3g}")
        if self.var_log.get():
            eps = 1e-6 * (self._data_hi - self._data_lo + 1e-9)
            norm = mcolors.LogNorm(vmin=max(lo, eps) if lo > 0 else eps, vmax=max(hi, eps * 10))
            self._im.set_norm(norm)
        else:
            self._im.set_norm(mcolors.Normalize(vmin=lo, vmax=hi))
        self.canvas.draw_idle()

    def update_preview(self):
        """Full redraw: re-read the (oriented, uncropped) image and reset display."""
        self._update_job = None
        try:
            src = self.build_source()
            src.crop = None   # preview uncropped so a crop rectangle can be drawn
            raw, info = src.to_raw_image()
            self._preview_raw = raw

            lo = float(np.nanmin(raw))
            hi = float(np.nanmax(raw))
            if not np.isfinite(lo):
                lo = 0.0
            if not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0
            self._data_lo, self._data_hi = lo, hi

            # configure slider ranges to the data range
            for s in (self.sld_vmin, self.sld_vmax):
                s.configure(from_=lo, to=hi)
            self.var_vmin.set(lo)
            self.var_vmax.set(hi)

            self.ax.clear()
            self._im = self.ax.imshow(raw, origin="lower", cmap="inferno",
                                      vmin=lo, vmax=hi)
            self._crop_patch = None
            if self._crop is not None:
                x0, y0, x1, y1 = self._crop
                self._crop_patch = matplotlib.patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", lw=1.5)
                self.ax.add_patch(self._crop_patch)
            self.ax.set_axis_off()
            h, w = raw.shape[:2]
            self.ax.set_title(f"{info.get('source', '')}  [{h}x{w}]", fontsize=8)
            # size label: full (uncropped) dims and, if a crop is set, the cropped dims
            if self._crop is not None:
                x0, y0, x1, y1 = [int(round(v)) for v in self._crop]
                self.lbl_size.configure(
                    text=f"size: {h}x{w} (HxW); crop {y1 - y0}x{x1 - x0}")
            else:
                self.lbl_size.configure(text=f"size: {h}x{w} (HxW)")
            self.fig.tight_layout()
            self._apply_clim()
            self.canvas.draw_idle()
            if "n_selected" in info:
                names = info.get("names_selected") or []
                loads = info.get("loadings_selected") or []
                pairs = ", ".join(
                    f"{n}({l:+.3f})" for n, l in zip(names, loads)
                ) if loads else ", ".join(map(str, names))
                self.log(
                    f"{self['text']} PC1 fusion [{info.get('selection_mode')}]: "
                    f"{info['n_selected']}/{info['n_pool']} channels -> {pairs}")
        except Exception as e:
            self.log(f"ERROR preview {self['text']}: {e}")


# =======================================================================
#  Main application
# =======================================================================
class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self._result_queue = queue.Queue()
        self._build()
        self.after(150, self._poll_queue)

    def _build(self):
        # bottom console first so it is always visible
        self._build_console()

        # outer split: [ left area (loading + method) | results ]
        self.hpanes = ttk.PanedWindow(self, orient="horizontal")
        self.hpanes.pack(fill="both", expand=True, side="top")

        # inner split (left area): loading on top, method below
        self.vpanes = ttk.PanedWindow(self.hpanes, orient="vertical")

        # ---- loading pane: ref & tgt side by side ----
        loading_scroll = ScrollableFrame(self.vpanes)
        loading = loading_scroll.body
        self.ref_panel = SourcePanel(loading, "Reference (coarse)", self.log)
        self.ref_panel.grid(row=0, column=0, sticky="nsew", padx=4, pady=3)
        self.tgt_panel = SourcePanel(loading, "Target (fine)", self.log)
        self.tgt_panel.grid(row=0, column=1, sticky="nsew", padx=4, pady=3)
        loading.rowconfigure(0, weight=1)
        loading.columnconfigure(0, weight=1, uniform="src")
        loading.columnconfigure(1, weight=1, uniform="src")

        # ---- method pane (below loading) ----
        method_scroll = ScrollableFrame(self.vpanes)
        self._build_method(method_scroll.body)

        self.vpanes.add(loading_scroll, weight=2)   # loading: top 2/3 height
        self.vpanes.add(method_scroll, weight=1)    # method: bottom 1/3 height

        # ---- results pane (full height, right) ----
        right = ttk.Frame(self.hpanes)
        self._build_results(right)

        self.hpanes.add(self.vpanes, weight=2)      # left area: 2/3 width
        self.hpanes.add(right, weight=1)            # results: 1/3 width

        self.after(120, self._init_sashes)

    def _init_sashes(self):
        try:
            w = self.hpanes.winfo_width()
            h = self.vpanes.winfo_height()
            if w < 50 or h < 50:
                self.after(120, self._init_sashes)
                return
            self.hpanes.sashpos(0, int(w * 2 / 3))   # loading/method area = 2/3 width
            self.vpanes.sashpos(0, int(h * 2 / 3))   # loading = 2/3 height
        except Exception:
            pass

    def _build_method(self, parent):
        mf = ttk.LabelFrame(parent, text="Registration method")
        mf.pack(fill="x", padx=4, pady=3)

        self.var_method = tk.StringVar(value="sift")
        self.var_stdpp = tk.BooleanVar(value=True)
        ttk.Radiobutton(mf, text="SIFT", variable=self.var_method,
                        value="sift", command=self._toggle_method).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mf, text="Match Template", variable=self.var_method,
                        value="match_template", command=self._toggle_method).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(mf, text="Standard preprocess", variable=self.var_stdpp).grid(
            row=0, column=2, sticky="w", padx=8)

        # SIFT params
        self.sift_frame = ttk.LabelFrame(mf, text="SIFT params")
        self.sift_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=3)
        self.var_lowe = tk.StringVar(value="0.9")
        self.var_conf = tk.StringVar(value="0.9")
        self.var_rthresh = tk.StringVar(value="5.0")
        ttk.Label(self.sift_frame, text="Lowe's ratio:").grid(row=0, column=0, sticky="e")
        ttk.Entry(self.sift_frame, textvariable=self.var_lowe, width=7).grid(row=0, column=1, sticky="w", padx=3)
        ttk.Label(self.sift_frame, text="RANSAC confidence:").grid(row=0, column=2, sticky="e")
        ttk.Entry(self.sift_frame, textvariable=self.var_conf, width=7).grid(row=0, column=3, sticky="w", padx=3)
        ttk.Label(self.sift_frame, text="RANSAC reproj thresh:").grid(row=1, column=0, sticky="e")
        ttk.Entry(self.sift_frame, textvariable=self.var_rthresh, width=7).grid(row=1, column=1, sticky="w", padx=3)

        # Match-template params
        self.mt_frame = ttk.LabelFrame(mf, text="Match-template params")
        self.mt_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=3)
        self.var_smin = tk.StringVar(value="0.2")
        self.var_smax = tk.StringVar(value="0.6")
        self.var_ssteps = tk.StringVar(value="41")
        self.var_amin = tk.StringVar(value="-5")
        self.var_amax = tk.StringVar(value="5")
        self.var_asteps = tk.StringVar(value="21")
        self.var_rotpad = tk.StringVar(value="pad")
        ttk.Label(self.mt_frame, text="Scale min/max/steps:").grid(row=0, column=0, sticky="e")
        ttk.Entry(self.mt_frame, textvariable=self.var_smin, width=6).grid(row=0, column=1)
        ttk.Entry(self.mt_frame, textvariable=self.var_smax, width=6).grid(row=0, column=2)
        ttk.Entry(self.mt_frame, textvariable=self.var_ssteps, width=6).grid(row=0, column=3)
        ttk.Label(self.mt_frame, text="Angle min/max/steps:").grid(row=1, column=0, sticky="e")
        ttk.Entry(self.mt_frame, textvariable=self.var_amin, width=6).grid(row=1, column=1)
        ttk.Entry(self.mt_frame, textvariable=self.var_amax, width=6).grid(row=1, column=2)
        ttk.Entry(self.mt_frame, textvariable=self.var_asteps, width=6).grid(row=1, column=3)
        ttk.Label(self.mt_frame, text="On rotation:").grid(row=2, column=0, sticky="e")
        ttk.Radiobutton(self.mt_frame, text="Pad (keep all)", variable=self.var_rotpad,
                        value="pad").grid(row=2, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(self.mt_frame, text="Cut (keep size)", variable=self.var_rotpad,
                        value="cut").grid(row=2, column=3, sticky="w")

        runf = ttk.Frame(mf)
        runf.grid(row=3, column=0, columnspan=3, sticky="ew", pady=4)
        self.btn_run = ttk.Button(runf, text="Run registration", command=self._run)
        self.btn_run.pack(side="left")
        self.progress = ttk.Progressbar(runf, mode="determinate", length=160, maximum=1.0)
        self.progress.pack(side="left", padx=8)

        self._toggle_method()

    def _toggle_method(self):
        sift = self.var_method.get() == "sift"
        for c in self.sift_frame.winfo_children():
            try:
                c.configure(state=("normal" if sift else "disabled"))
            except tk.TclError:
                pass
        for c in self.mt_frame.winfo_children():
            try:
                c.configure(state=("disabled" if sift else "normal"))
            except tk.TclError:
                pass

    def _build_results(self, parent):
        rf = ttk.LabelFrame(parent, text="Result")
        rf.pack(fill="both", expand=True, padx=4, pady=3)

        self.res_fig = Figure(figsize=(3.2, 2.6), dpi=90)
        self.res_ax = self.res_fig.add_subplot(111)
        self.res_ax.set_axis_off()
        self.res_canvas = FigureCanvasTkAgg(self.res_fig, master=rf)
        self.res_canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.res_canvas, rf)

        self.txt_metrics = tk.Text(rf, height=13, width=40, state="disabled",
                                   font=("Consolas", 9))
        self.txt_metrics.pack(fill="x", padx=2, pady=2)

    def _build_console(self):
        cf = ttk.LabelFrame(self, text="Messages / Errors")
        cf.pack(fill="x", side="bottom")
        self.console = tk.Text(cf, height=7, state="disabled",
                               background="#1e1e1e", foreground="#e0e0e0",
                               font=("Consolas", 9))
        self.console.pack(fill="x", side="left", expand=True)
        sb = ttk.Scrollbar(cf, command=self.console.yview)
        sb.pack(fill="y", side="right")
        self.console.configure(yscrollcommand=sb.set)
        self.console.tag_config("err", foreground="#ff6b6b")
        self.console.tag_config("ok", foreground="#7CFC00")

    # ---------------- logging ----------------
    def log(self, msg, level="info"):
        self.console.configure(state="normal")
        tag = "err" if (level == "err" or str(msg).startswith("ERROR")) else (
            "ok" if level == "ok" else None)
        self.console.insert("end", str(msg) + "\n", tag)
        self.console.see("end")
        self.console.configure(state="disabled")

    # ---------------- run ----------------
    def _collect_params(self):
        method = self.var_method.get()
        sift_params = dict(
            lowe_ratio=float(self.var_lowe.get()),
            ransac_conf=float(self.var_conf.get()),
            ransac_thresh=float(self.var_rthresh.get()),
        )
        mt_params = dict(
            scale_min=float(self.var_smin.get()),
            scale_max=float(self.var_smax.get()),
            scale_steps=int(self.var_ssteps.get()),
            angle_min=float(self.var_amin.get()),
            angle_max=float(self.var_amax.get()),
            angle_steps=int(self.var_asteps.get()),
            rot_pad=(self.var_rotpad.get() == "pad"),
        )
        return method, sift_params, mt_params

    def _run(self):
        try:
            ref_src = self.ref_panel.build_source()
            tgt_src = self.tgt_panel.build_source()
            method, sift_params, mt_params = self._collect_params()
        except Exception as e:
            self.log(f"ERROR invalid input: {e}", "err")
            return

        self.btn_run.configure(state="disabled")
        self.progress["value"] = 0
        self.log(f"Running {method} registration ...")

        def progress_cb(frac):
            self._result_queue.put(("progress", frac))

        def worker():
            try:
                res = rc.run_registration(
                    ref_src, tgt_src, method=method,
                    standard_pp=bool(self.var_stdpp.get()),
                    sift_params=sift_params, mt_params=mt_params,
                    progress=progress_cb,
                )
                self._result_queue.put(("done", res))
            except Exception:
                self._result_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._result_queue.get_nowait()
                if kind == "progress":
                    self.progress["value"] = payload
                elif kind == "error":
                    self.btn_run.configure(state="normal")
                    self.log("ERROR during registration:", "err")
                    for line in str(payload).splitlines():
                        self.log("  " + line, "err")
                elif kind == "done":
                    self.btn_run.configure(state="normal")
                    self.progress["value"] = 1.0
                    self._show_result(payload)
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)

    def _show_result(self, res):
        for m in res.messages:
            self.log(m, "err" if not res.ok else "info")

        # report the channels that built each PC1 fusion image (if used)
        for label, info in (("Reference", getattr(res, "ref_info", None)),
                            ("Target", getattr(res, "tgt_info", None))):
            if info and "names_selected" in info:
                names = info.get("names_selected") or []
                loads = info.get("loadings_selected") or []
                pairs = ", ".join(f"{n}({l:+.3f})" for n, l in zip(names, loads)) \
                    if loads else ", ".join(map(str, names))
                self.log(f"{label} PC1 fusion channels "
                         f"({info.get('n_selected')}/{info.get('n_pool')}): {pairs}")

        self.res_ax.clear()
        if res.overlay is not None:
            self.res_ax.imshow(res.overlay, origin="lower")
        title = f"{res.method} overlay" + ("" if res.ok else "  (FAILED)")
        self.res_ax.set_title(title, fontsize=9)
        self.res_ax.set_axis_off()
        self.res_fig.tight_layout()
        self.res_canvas.draw_idle()

        lines = []
        lines.append(f"Method        : {res.method}")
        lines.append(f"Success       : {res.ok}")
        lines.append(f"Scale (tgt->ref): {res.scale:.4f}")
        lines.append(f"Rotation (deg): {res.angle:.3f}")
        if res.size_relation:
            lines.append(f"Size relation : {res.size_relation}")
        if res.corner_ll is not None:
            lines.append(f"Lower-left (x,y): ({res.corner_ll[0]:.1f}, {res.corner_ll[1]:.1f})")
        lines.append("")
        lines.append("-- Standard metrics --")
        for k in ("ncc", "rmse", "mi", "ssim", "gncc"):
            if k in res.metrics:
                lines.append(f"  {k:<5}: {res.metrics[k]:.5f}")
        lines.append("-- PCA-based metrics --")
        for k in ("pca_cosine", "pca_pearson"):
            if k in res.metrics and isinstance(res.metrics[k], float):
                lines.append(f"  {k:<11}: {res.metrics[k]:.5f}")
        if "pca_n_channels" in res.metrics:
            lines.append(f"  (channels used: {res.metrics['pca_n_channels']})")
        if res.metrics.get("pca_channels"):
            chs = res.metrics["pca_channels"]
            lines.append("  " + ", ".join(chs))
        if res.metrics.get("pca_note"):
            lines.append(f"  {res.metrics['pca_note']}")
        if res.metrics.get("pca_error"):
            lines.append(f"  pca_error: {res.metrics['pca_error']}")
        for label, info in (("Ref", getattr(res, "ref_info", None)),
                            ("Tgt", getattr(res, "tgt_info", None))):
            if info and info.get("names_selected"):
                lines.append(f"-- {label} PC1 fusion channels "
                             f"({info.get('n_selected')}/{info.get('n_pool')}) --")
                lines.append("  " + ", ".join(map(str, info["names_selected"])))
        if res.method == "SIFT":
            lines.append("-- SIFT stats --")
            s = res.sift_stats
            lines.append(f"  keypoints ref : {s.get('keypoints_ref')}")
            lines.append(f"  keypoints tgt : {s.get('keypoints_tgt')}")
            lines.append(f"  matches       : {s.get('matches')}")
            lines.append(f"  good matches  : {s.get('good_matches')}")
            lines.append(f"  RANSAC inliers: {s.get('ransac_inliers')}")

        self.txt_metrics.configure(state="normal")
        self.txt_metrics.delete("1.0", "end")
        self.txt_metrics.insert("end", "\n".join(lines))
        self.txt_metrics.configure(state="disabled")
        self.log("Registration done.", "ok")


def main():
    root = tk.Tk()
    root.title("XRF / Image Registration GUI")
    root.geometry("1200x860")
    try:
        App(root)
    except Exception:
        messagebox.showerror("Startup error", traceback.format_exc())
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
