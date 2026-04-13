from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk
from segment_anything import SamPredictor, sam_model_registry


ROOT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = ROOT_DIR / "input" / "source"
DEFAULT_MASK_DIR = ROOT_DIR / "input" / "masks"
DEFAULT_CHECKPOINT = ROOT_DIR / ".venv" / "sam_checkpoints" / "sam_vit_b_01ec64.pth"
DEFAULT_MODEL_TYPE = "vit_b"
MAX_CANVAS_WIDTH = 1100
MAX_CANVAS_HEIGHT = 820
MASK_OVERLAY_COLOR = (0, 255, 0, 110)
FG_POINT_COLOR = "#27ae60"
BG_POINT_COLOR = "#e74c3c"
BOX_COLOR = "#f1c40f"


class MaskGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SAM Masking Tool")
        self.geometry("1600x980")
        self.minsize(1300, 820)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = DEFAULT_CHECKPOINT
        self.model_type = DEFAULT_MODEL_TYPE
        self.predictor = self._load_predictor()

        self.source_dir = SOURCE_DIR
        self.image_paths: list[Path] = []
        self.current_image_path: Path | None = None
        self.current_pil_image: Image.Image | None = None
        self.current_np_image: np.ndarray | None = None
        self.display_image: Image.Image | None = None
        self.tk_display_image: ImageTk.PhotoImage | None = None
        self.current_mask: np.ndarray | None = None
        self.mask_candidates: list[np.ndarray] = []
        self.mask_scores: list[float] = []
        self.current_mask_index = 0

        self.prompt_history: list[dict[str, object]] = []
        self.box_start: tuple[float, float] | None = None
        self.active_box_canvas_id: int | None = None
        self.canvas_scale = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

        self.mode_var = tk.StringVar(value="fg")
        self.status_var = tk.StringVar(value="Ready")
        self.mask_info_var = tk.StringVar(value="No mask generated")

        self._build_ui()
        self._bind_events()
        self._load_source_images()

        if self.image_paths:
            self._load_image(self.image_paths[0])

    def _load_predictor(self) -> SamPredictor:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {self.checkpoint_path}\n"
                "Install the checkpoint first, then relaunch the GUI."
            )

        sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
        sam.to(device=self.device)
        sam.eval()
        return SamPredictor(sam)

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self, padding=12)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.columnconfigure(0, weight=1)

        viewer = ttk.Frame(self, padding=(0, 12, 12, 12))
        viewer.grid(row=0, column=1, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(0, weight=1)

        ttk.Label(sidebar, text="Source Images").grid(row=0, column=0, sticky="w")

        self.image_listbox = tk.Listbox(sidebar, height=14, exportselection=False)
        self.image_listbox.grid(row=1, column=0, sticky="nsew", pady=(6, 10))

        image_buttons = ttk.Frame(sidebar)
        image_buttons.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        image_buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(image_buttons, text="Reload Folder", command=self._load_source_images).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(image_buttons, text="Open File", command=self._open_file).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        ttk.Label(sidebar, text="Prompt Mode").grid(row=3, column=0, sticky="w")
        mode_frame = ttk.Frame(sidebar)
        mode_frame.grid(row=4, column=0, sticky="ew", pady=(6, 16))
        for idx, (label, value) in enumerate(
            [("Foreground", "fg"), ("Background", "bg"), ("Box", "box")]
        ):
            ttk.Radiobutton(mode_frame, text=label, variable=self.mode_var, value=value).grid(
                row=idx, column=0, sticky="w", pady=2
            )

        actions = ttk.LabelFrame(sidebar, text="Mask Actions", padding=10)
        actions.grid(row=5, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)

        ttk.Button(actions, text="Predict / Refresh", command=self._predict_mask).grid(
            row=0, column=0, sticky="ew", pady=2
        )
        ttk.Button(actions, text="Next Candidate", command=self._next_mask_candidate).grid(
            row=1, column=0, sticky="ew", pady=2
        )
        ttk.Button(actions, text="Undo Last Prompt", command=self._undo_prompt).grid(
            row=2, column=0, sticky="ew", pady=2
        )
        ttk.Button(actions, text="Clear Prompts", command=self._clear_prompts).grid(
            row=3, column=0, sticky="ew", pady=2
        )

        exports = ttk.LabelFrame(sidebar, text="Export", padding=10)
        exports.grid(row=6, column=0, sticky="ew", pady=(16, 0))
        exports.columnconfigure(0, weight=1)

        ttk.Button(exports, text="Save Mask PNG", command=self._save_mask).grid(
            row=0, column=0, sticky="ew", pady=2
        )
        ttk.Button(exports, text="Save Overlay PNG", command=self._save_overlay).grid(
            row=1, column=0, sticky="ew", pady=2
        )
        ttk.Button(exports, text="Save Cutout PNG", command=self._save_cutout).grid(
            row=2, column=0, sticky="ew", pady=2
        )

        info = ttk.LabelFrame(sidebar, text="Status", padding=10)
        info.grid(row=7, column=0, sticky="ew", pady=(16, 0))
        info.columnconfigure(0, weight=1)
        ttk.Label(info, text=f"Device: {self.device}").grid(row=0, column=0, sticky="w")
        ttk.Label(info, text=f"Model: {self.model_type}").grid(row=1, column=0, sticky="w")
        ttk.Label(info, textvariable=self.mask_info_var, wraplength=260).grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(info, textvariable=self.status_var, wraplength=260).grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )

        self.canvas = tk.Canvas(
            viewer,
            background="#1f1f1f",
            highlightthickness=0,
            width=MAX_CANVAS_WIDTH,
            height=MAX_CANVAS_HEIGHT,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

    def _bind_events(self) -> None:
        self.image_listbox.bind("<<ListboxSelect>>", self._on_image_select)
        self.canvas.bind("<Button-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.bind("<Control-z>", lambda event: self._undo_prompt())
        self.bind("<Escape>", lambda event: self._clear_box_preview())
        self.bind("<Configure>", lambda event: self._redraw_canvas())

    def _load_source_images(self) -> None:
        self.image_paths = sorted(
            [path for path in self.source_dir.iterdir() if path.is_file()],
            key=lambda path: path.name.lower(),
        ) if self.source_dir.exists() else []

        self.image_listbox.delete(0, tk.END)
        for path in self.image_paths:
            self.image_listbox.insert(tk.END, path.name)

        if not self.image_paths:
            self.status_var.set(f"No images found in {self.source_dir}")
            return

        if self.current_image_path in self.image_paths:
            index = self.image_paths.index(self.current_image_path)
        else:
            index = 0
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(index)
        self.image_listbox.activate(index)

    def _on_image_select(self, _event: tk.Event) -> None:
        selection = self.image_listbox.curselection()
        if not selection:
            return
        self._load_image(self.image_paths[selection[0]])

    def _open_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
            initialdir=str(self.source_dir if self.source_dir.exists() else ROOT_DIR),
        )
        if file_path:
            self._load_image(Path(file_path))

    def _load_image(self, image_path: Path) -> None:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Open Image Failed", str(exc))
            return

        self.current_image_path = image_path
        self.current_pil_image = image
        self.current_np_image = np.array(image)
        self.predictor.set_image(self.current_np_image)
        self._clear_prompts(reset_status=False)

        self.status_var.set(f"Loaded {image_path.name}")
        self._redraw_canvas()

    def _clear_prompts(self, reset_status: bool = True) -> None:
        self.prompt_history.clear()
        self.current_mask = None
        self.mask_candidates.clear()
        self.mask_scores.clear()
        self.current_mask_index = 0
        self.box_start = None
        self._clear_box_preview()
        self.mask_info_var.set("No mask generated")
        if reset_status:
            self.status_var.set("Cleared prompts")
        self._redraw_canvas()

    def _undo_prompt(self) -> None:
        if not self.prompt_history:
            return
        self.prompt_history.pop()
        self.status_var.set("Removed last prompt")
        if self.prompt_history:
            self._predict_mask()
        else:
            self.current_mask = None
            self.mask_candidates.clear()
            self.mask_scores.clear()
            self.mask_info_var.set("No mask generated")
            self._redraw_canvas()

    def _next_mask_candidate(self) -> None:
        if not self.mask_candidates:
            self.status_var.set("No mask candidates to cycle")
            return
        self.current_mask_index = (self.current_mask_index + 1) % len(self.mask_candidates)
        self.current_mask = self.mask_candidates[self.current_mask_index]
        score = self.mask_scores[self.current_mask_index]
        self.mask_info_var.set(
            f"Mask {self.current_mask_index + 1}/{len(self.mask_candidates)} "
            f"(score {score:.3f})"
        )
        self.status_var.set("Switched mask candidate")
        self._redraw_canvas()

    def _on_canvas_press(self, event: tk.Event) -> None:
        image_coords = self._canvas_to_image(event.x, event.y)
        if image_coords is None:
            return

        if self.mode_var.get() == "box":
            self.box_start = image_coords
            self._clear_box_preview()
        else:
            label = 1 if self.mode_var.get() == "fg" else 0
            self.prompt_history.append(
                {"type": "point", "label": label, "x": image_coords[0], "y": image_coords[1]}
            )
            self.status_var.set(
                f"Added {'foreground' if label == 1 else 'background'} point at "
                f"({int(image_coords[0])}, {int(image_coords[1])})"
            )
            self._predict_mask()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self.mode_var.get() != "box" or self.box_start is None:
            return

        image_coords = self._canvas_to_image(event.x, event.y)
        if image_coords is None:
            return

        x0, y0 = self._image_to_canvas(*self.box_start)
        x1, y1 = self._image_to_canvas(*image_coords)
        self._clear_box_preview()
        self.active_box_canvas_id = self.canvas.create_rectangle(
            x0, y0, x1, y1, outline=BOX_COLOR, width=2, dash=(6, 4)
        )

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self.mode_var.get() != "box" or self.box_start is None:
            return

        image_coords = self._canvas_to_image(event.x, event.y)
        if image_coords is None:
            self.box_start = None
            self._clear_box_preview()
            return

        x0, y0 = self.box_start
        x1, y1 = image_coords
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))

        self.box_start = None
        self._clear_box_preview()

        if right - left < 2 or bottom - top < 2:
            self.status_var.set("Ignored tiny box")
            return

        self.prompt_history = [prompt for prompt in self.prompt_history if prompt["type"] != "box"]
        self.prompt_history.append(
            {"type": "box", "x0": left, "y0": top, "x1": right, "y1": bottom}
        )
        self.status_var.set(
            f"Added box ({int(left)}, {int(top)}) -> ({int(right)}, {int(bottom)})"
        )
        self._predict_mask()

    def _predict_mask(self) -> None:
        if self.current_np_image is None:
            return

        point_coords = []
        point_labels = []
        box = None

        for prompt in self.prompt_history:
            if prompt["type"] == "point":
                point_coords.append([prompt["x"], prompt["y"]])
                point_labels.append(prompt["label"])
            elif prompt["type"] == "box":
                box = np.array(
                    [prompt["x0"], prompt["y0"], prompt["x1"], prompt["y1"]],
                    dtype=np.float32,
                )

        if not point_coords and box is None:
            self.status_var.set("Add points or a box first")
            return

        coords_array = np.array(point_coords, dtype=np.float32) if point_coords else None
        labels_array = np.array(point_labels, dtype=np.int32) if point_labels else None

        try:
            masks, scores, _ = self.predictor.predict(
                point_coords=coords_array,
                point_labels=labels_array,
                box=box,
                multimask_output=True,
            )
        except Exception as exc:
            messagebox.showerror("Prediction Failed", str(exc))
            return

        order = np.argsort(scores)[::-1]
        self.mask_candidates = [masks[idx] for idx in order]
        self.mask_scores = [float(scores[idx]) for idx in order]
        self.current_mask_index = 0
        self.current_mask = self.mask_candidates[0]
        self.mask_info_var.set(
            f"Mask 1/{len(self.mask_candidates)} (score {self.mask_scores[0]:.3f})"
        )
        self.status_var.set("Mask generated")
        self._redraw_canvas()

    def _redraw_canvas(self) -> None:
        self.canvas.delete("all")
        if self.current_pil_image is None:
            return

        canvas_width = max(self.canvas.winfo_width(), 100)
        canvas_height = max(self.canvas.winfo_height(), 100)
        img_width, img_height = self.current_pil_image.size
        self.canvas_scale = min(canvas_width / img_width, canvas_height / img_height)
        self.canvas_scale = min(self.canvas_scale, 1.0)

        new_size = (
            max(1, int(img_width * self.canvas_scale)),
            max(1, int(img_height * self.canvas_scale)),
        )
        self.canvas_offset_x = (canvas_width - new_size[0]) // 2
        self.canvas_offset_y = (canvas_height - new_size[1]) // 2

        base_image = self.current_pil_image.copy().resize(new_size, Image.Resampling.LANCZOS)

        if self.current_mask is not None:
            overlay = Image.new("RGBA", self.current_pil_image.size, (0, 0, 0, 0))
            overlay_pixels = np.zeros((self.current_mask.shape[0], self.current_mask.shape[1], 4), dtype=np.uint8)
            overlay_pixels[self.current_mask.astype(bool)] = MASK_OVERLAY_COLOR
            overlay = Image.fromarray(overlay_pixels, mode="RGBA")
            overlay = overlay.resize(new_size, Image.Resampling.NEAREST)
            base_image = Image.alpha_composite(base_image.convert("RGBA"), overlay)
        else:
            base_image = base_image.convert("RGBA")

        draw = ImageDraw.Draw(base_image)
        radius = max(4, int(6 * self.canvas_scale))
        for prompt in self.prompt_history:
            if prompt["type"] == "point":
                x, y = self._image_to_display(prompt["x"], prompt["y"])
                color = FG_POINT_COLOR if prompt["label"] == 1 else BG_POINT_COLOR
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white")
            elif prompt["type"] == "box":
                x0, y0 = self._image_to_display(prompt["x0"], prompt["y0"])
                x1, y1 = self._image_to_display(prompt["x1"], prompt["y1"])
                draw.rectangle((x0, y0, x1, y1), outline=BOX_COLOR, width=max(2, int(2 * self.canvas_scale)))

        self.display_image = base_image
        self.tk_display_image = ImageTk.PhotoImage(base_image)
        self.canvas.create_image(
            self.canvas_offset_x,
            self.canvas_offset_y,
            image=self.tk_display_image,
            anchor="nw",
        )

    def _save_mask(self) -> None:
        if not self._ensure_mask():
            return
        path = self._save_path("Save Mask PNG", "_mask.png")
        if not path:
            return
        mask_image = Image.fromarray((self.current_mask.astype(np.uint8) * 255), mode="L")
        mask_image.save(path)
        self.status_var.set(f"Saved mask to {path}")

    def _save_overlay(self) -> None:
        if not self._ensure_mask():
            return
        path = self._save_path("Save Overlay PNG", "_overlay.png")
        if not path:
            return
        self._build_overlay_image().save(path)
        self.status_var.set(f"Saved overlay to {path}")

    def _save_cutout(self) -> None:
        if not self._ensure_mask():
            return
        path = self._save_path("Save Cutout PNG", "_cutout.png")
        if not path:
            return
        rgba = self.current_pil_image.convert("RGBA")
        alpha = Image.fromarray((self.current_mask.astype(np.uint8) * 255), mode="L")
        rgba.putalpha(alpha)
        rgba.save(path)
        self.status_var.set(f"Saved cutout to {path}")

    def _ensure_mask(self) -> bool:
        if self.current_mask is None or self.current_pil_image is None:
            self.status_var.set("Generate a mask first")
            return False
        return True

    def _save_path(self, title: str, suffix: str) -> Path | None:
        DEFAULT_MASK_DIR.mkdir(parents=True, exist_ok=True)
        stem = self.current_image_path.stem if self.current_image_path else "mask"
        file_path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=".png",
            initialdir=str(DEFAULT_MASK_DIR),
            initialfile=f"{stem}{suffix}",
            filetypes=[("PNG files", "*.png")],
        )
        return Path(file_path) if file_path else None

    def _build_overlay_image(self) -> Image.Image:
        base = self.current_pil_image.convert("RGBA")
        overlay_pixels = np.zeros((self.current_mask.shape[0], self.current_mask.shape[1], 4), dtype=np.uint8)
        overlay_pixels[self.current_mask.astype(bool)] = MASK_OVERLAY_COLOR
        overlay = Image.fromarray(overlay_pixels, mode="RGBA")
        return Image.alpha_composite(base, overlay)

    def _clear_box_preview(self) -> None:
        if self.active_box_canvas_id is not None:
            self.canvas.delete(self.active_box_canvas_id)
            self.active_box_canvas_id = None

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        if self.current_pil_image is None:
            return None

        rel_x = x - self.canvas_offset_x
        rel_y = y - self.canvas_offset_y
        disp_width = int(self.current_pil_image.width * self.canvas_scale)
        disp_height = int(self.current_pil_image.height * self.canvas_scale)

        if rel_x < 0 or rel_y < 0 or rel_x > disp_width or rel_y > disp_height:
            return None

        image_x = rel_x / self.canvas_scale
        image_y = rel_y / self.canvas_scale
        image_x = min(max(image_x, 0), self.current_pil_image.width - 1)
        image_y = min(max(image_y, 0), self.current_pil_image.height - 1)
        return image_x, image_y

    def _image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        dx, dy = self._image_to_display(x, y)
        return dx + self.canvas_offset_x, dy + self.canvas_offset_y

    def _image_to_display(self, x: float, y: float) -> tuple[float, float]:
        return x * self.canvas_scale, y * self.canvas_scale


def main() -> None:
    try:
        app = MaskGui()
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("SAM Masking Tool", str(exc))
        root.destroy()
        raise SystemExit(1) from exc
    app.mainloop()


if __name__ == "__main__":
    main()
