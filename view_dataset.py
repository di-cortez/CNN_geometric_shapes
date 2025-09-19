
# view_domino_dataset.py
# A tiny pixel-art viewer for synthetic domino datasets.
# Lines <= 75 chars. Comments in English. No NN code.

import os
import io
import zipfile
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw


# ----------------------------- helpers ---------------------------------

def _is_image_name(name: str) -> bool:
    """Return True if 'name' looks like an image entry in the zip."""
    low = name.lower()
    if not low.endswith(".png"):
        return False
    base = os.path.basename(low)
    return base.replace(".png", "").isdigit()


def _discover_archives(folder: str) -> Tuple[Optional[str], Optional[str]]:
    """Find (images_zip, labels_zip) inside a folder using heuristics."""
    imgs, labs = None, None
    for f in os.listdir(folder):
        lf = f.lower()
        if not lf.endswith(".zip"):
            continue
        full = os.path.join(folder, f)
        if "images" in lf and imgs is None:
            imgs = full
        if "labels" in lf and labs is None:
            labs = full
    return imgs, labs


def _list_candidate_folders(base_dir: str) -> List[str]:
    """Return subfolders that likely contain an images zip."""
    out: List[str] = []
    for root, dirs, files in os.walk(base_dir):
        # one level deep is enough for UX
        if root != base_dir:
            dirs[:] = []
        if any("images" in f.lower() and f.lower().endswith(".zip")
               for f in files):
            out.append(root)
    out.sort()
    return out


# ----------------------------- dataset ---------------------------------

class ZipDataset:
    """Lazy reader for images_*.zip and labels_*.zip archives."""

    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.img_zip_path, self.lab_zip_path = _discover_archives(folder)
        if not self.img_zip_path:
            raise FileNotFoundError("images_*.zip not found in folder")
        self.img_zf = zipfile.ZipFile(self.img_zip_path, "r")
        self.lab_zf = None
        if self.lab_zip_path and os.path.isfile(self.lab_zip_path):
            self.lab_zf = zipfile.ZipFile(self.lab_zip_path, "r")
        # list image names sorted by numeric index
        names = [n for n in self.img_zf.namelist() if _is_image_name(n)]
        def key(n: str) -> int:
            b = os.path.basename(n)
            return int(b.split(".")[0])
        self.names = sorted(names, key=key)
        self.count = len(self.names)

    def close(self) -> None:
        if self.img_zf:
            self.img_zf.close()
        if self.lab_zf:
            self.lab_zf.close()

    def load_image(self, idx: int) -> Image.Image:
        idx = max(0, min(idx, self.count - 1))
        name = self.names[idx]
        data = self.img_zf.read(name)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img

    def load_label(self, idx: int) -> str:
        if not self.lab_zf:
            return ""
        base = os.path.basename(self.names[idx])
        txt = base.replace(".png", ".txt")
        # try plain and labels/ prefix
        candidates = [txt, os.path.join("labels", txt)]
        for c in candidates:
            try:
                return self.lab_zf.read(c).decode().strip()
            except KeyError:
                pass
        return ""


# ------------------------------ GUI ------------------------------------

class Viewer(tk.Tk):
    """Simple pixel-art viewer with nearest-neighbour zoom."""

    def __init__(self, base_dir: str = ".") -> None:
        super().__init__()
        self.title("Domino Dataset Viewer")
        self.base_dir = base_dir
        self.dataset: Optional[ZipDataset] = None
        self.cur_idx = 0
        self.cur_zoom = 800  # percent
        self.show_grid = tk.BooleanVar(value=False)
        self._tk_img = None  # keep ref

        self._build_ui()
        self._populate_folders()
        self.bind_all("<Left>", lambda e: self.prev())
        self.bind_all("<Right>", lambda e: self.next())
        self.bind_all("<Home>", lambda e: self.first())
        self.bind_all("<End>", lambda e: self.last())
        self.bind_all("<Prior>", lambda e: self.jump(-10))
        self.bind_all("<Next>", lambda e: self.jump(+10))

    # ------------------------- UI building ------------------------------

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=6)
        top.grid(row=0, column=0, sticky="ew")
        # folder selector
        ttk.Label(top, text="Folder:").grid(row=0, column=0, sticky="w")
        self.folder_var = tk.StringVar()
        self.folder_cb = ttk.Combobox(
            top, textvariable=self.folder_var, width=60, state="readonly"
        )
        self.folder_cb.grid(row=0, column=1, padx=4, sticky="ew")
        self.folder_cb.bind("<<ComboboxSelected>>", self._on_folder)
        ttk.Button(top, text="Reload", command=self._populate_folders
                   ).grid(row=0, column=2, padx=4)

        # index controls
        nav = ttk.Frame(self, padding=(6, 0))
        nav.grid(row=1, column=0, sticky="w")
        ttk.Button(nav, text="<<", command=self.first).grid(row=0, column=0)
        ttk.Button(nav, text="<", command=self.prev).grid(row=0, column=1)
        ttk.Button(nav, text=">", command=self.next).grid(row=0, column=2)
        ttk.Button(nav, text=">>", command=self.last).grid(row=0, column=3)
        ttk.Label(nav, text="Index:").grid(row=0, column=4, padx=(8, 2))
        self.idx_var = tk.StringVar(value="0")
        e = ttk.Entry(nav, textvariable=self.idx_var, width=6)
        e.grid(row=0, column=5)
        e.bind("<Return>", lambda e: self._goto_index())
        ttk.Label(nav, text="Zoom:").grid(row=0, column=6, padx=(12, 2))
        self.zoom_var = tk.StringVar(value="800%")
        self.zoom_cb = ttk.Combobox(
            nav,
            textvariable=self.zoom_var,
            values=["100%", "200%", "400%", "800%", "1600%", "2500%", "2800%", "3200%", "6400%"],
            state="readonly",
            width=7,
        )
        self.zoom_cb.grid(row=0, column=7)
        self.zoom_cb.bind("<<ComboboxSelected>>", self._on_zoom)
        ttk.Checkbutton(nav, text="Grid 1px", variable=self.show_grid,
                        command=self._refresh
                        ).grid(row=0, column=8, padx=(10, 0))

        # image area
        self.image_lbl = ttk.Label(self, background="#202020")
        self.image_lbl.grid(row=2, column=0, padx=6, pady=6)

        # label text
        self.label_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.label_var).grid(
            row=3, column=0, sticky="w", padx=6, pady=(0, 6)
        )

    # --------------------- dataset/folder logic -------------------------

    def _populate_folders(self) -> None:
        folders = _list_candidate_folders(self.base_dir)
        if not folders:
            messagebox.showerror(
                "No datasets",
                "No subfolders with images_*.zip found."
            )
            return
        self.folder_cb["values"] = folders
        self.folder_cb.current(0)
        self._load_folder(folders[0])

    def _on_folder(self, _evt=None) -> None:
        self._load_folder(self.folder_var.get())

    def _load_folder(self, folder: str) -> None:
        if self.dataset:
            self.dataset.close()
            self.dataset = None
        try:
            self.dataset = ZipDataset(folder)
        except Exception as e:
            messagebox.showerror("Open error", str(e))
            return
        self.cur_idx = 0
        self.idx_var.set("0")
        self._refresh()

    # --------------------------- navigation -----------------------------

    def first(self) -> None:
        self.cur_idx = 0
        self.idx_var.set("0")
        self._refresh()

    def last(self) -> None:
        if not self.dataset:
            return
        self.cur_idx = max(0, self.dataset.count - 1)
        self.idx_var.set(str(self.cur_idx))
        self._refresh()

    def prev(self) -> None:
        if not self.dataset:
            return
        self.cur_idx = max(0, self.cur_idx - 1)
        self.idx_var.set(str(self.cur_idx))
        self._refresh()

    def next(self) -> None:
        if not self.dataset:
            return
        self.cur_idx = min(self.dataset.count - 1, self.cur_idx + 1)
        self.idx_var.set(str(self.cur_idx))
        self._refresh()

    def jump(self, delta: int) -> None:
        if not self.dataset:
            return
        self.cur_idx = min(self.dataset.count - 1,
                           max(0, self.cur_idx + delta))
        self.idx_var.set(str(self.cur_idx))
        self._refresh()

    def _goto_index(self) -> None:
        try:
            idx = int(self.idx_var.get())
        except ValueError:
            idx = 0
        if not self.dataset:
            return
        idx = min(self.dataset.count - 1, max(0, idx))
        self.cur_idx = idx
        self.idx_var.set(str(self.cur_idx))
        self._refresh()

    def _on_zoom(self, _evt=None) -> None:
        val = self.zoom_var.get().strip().rstrip("%")
        try:
            self.cur_zoom = max(100, int(val))
        except ValueError:
            self.cur_zoom = 800
        self._refresh()

    # ---------------------------- drawing -------------------------------

    def _draw_grid(self, img: Image.Image, step_x: int, step_y: int) -> Image.Image:
        # não desenha grid se não houver ampliação (evita tela preta)
        if step_x < 2 and step_y < 2:
            return img
        w, h = img.size
        out = img.copy()
        draw = ImageDraw.Draw(out)
        color = (0, 0, 0)
        for x in range(0, w, step_x):
            draw.line((x, 0, x, h - 1), fill=color)
        for y in range(0, h, step_y):
            draw.line((0, y, w - 1, y), fill=color)
        return out


    def _refresh(self) -> None:
        if not self.dataset:
            return
        try:
            img = self.dataset.load_image(self.cur_idx)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        label = self.dataset.load_label(self.cur_idx)
        self.label_var.set(
            f"Folder: {self.dataset.folder} | Index: {self.cur_idx} | "
            f"Size: {img.size[0]}x{img.size[1]} | Label: {label}"
        )
        # scale with nearest neighbour
        scale = self.cur_zoom / 100.0
        new_size = (int(img.width * scale), int(img.height * scale))
        scaled = img.resize(new_size, Image.NEAREST)
        if self.show_grid.get():
            step_x = max(1, new_size[0] // img.width)
            step_y = max(1, new_size[1] // img.height)
            scaled = self._draw_grid(scaled, step_x, step_y)
        tk_img = ImageTk.PhotoImage(scaled)
        self.image_lbl.configure(image=tk_img)
        self.image_lbl.image = tk_img
        self._tk_img = tk_img  # keep ref


# ----------------------------- main ------------------------------------

def main() -> None:
    root = Viewer(".")
    root.mainloop()


if __name__ == "__main__":
    main()
