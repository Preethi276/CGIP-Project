"""
CGIP - Creative Image Studio
Image Cartoonization & Sketching
Modern desktop GUI using CustomTkinter and OpenCV
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox


import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image

from core.sketch import pencil_sketch, color_pencil_sketch
from core.cartoon import cartoonize, cartoonize_kmeans, oil_painting_effect
from core.utils import resize_keep_aspect
from comparison import open_effect_comparison

# =========================================================
# APPLICATION THEME
# =========================================================

ctk.set_appearance_mode("dark")

BG = "#151515"
SIDEBAR = "#1C1C1C"
PANEL = "#222222"
PANEL_LIGHT = "#292929"
CANVAS = "#101010"

CORAL = "#FF704F"
CORAL_HOVER = "#E85D40"
LIME = "#C7E56B"

TEXT = "#F4EFE6"
MUTED = "#999999"
BORDER = "#373737"


EFFECTS = [
    "Pencil Sketch",
    "Color Pencil Sketch",
    "Classic Cartoon",
    "K-Means Cartoon",
    "Oil Painting",
]


class CartoonSketchApp:

    def __init__(self, root):

        self.root = root
        self.root.title("CGIP | Creative Image Studio")
        self.root.geometry("1500x900")
        self.root.minsize(1150, 720)
        self.root.configure(fg_color=BG)

        self.original = None
        self.processed = None

        self.image_name = "No image selected"
        self.current_mode = tk.StringVar(value="Pencil Sketch")
        self.show_comparison = tk.BooleanVar(value=False)

        self.blur_ksize = tk.IntVar(value=21)
        self.bilateral_d = tk.IntVar(value=9)
        self.sigma_color = tk.IntVar(value=75)
        self.block_size = tk.IntVar(value=9)
        self.k_clusters = tk.IntVar(value=8)

        self.slider_widgets = {}
        self.gallery_images = []
        self.preview_image = None

        self._build_ui()
        self._build_effect_gallery()

        self.root.bind("<Configure>", self._on_resize)

    # =====================================================
    # MAIN LAYOUT
    # =====================================================

    def _build_ui(self):

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_area()

    # =====================================================
    # SIDEBAR
    # =====================================================

    def _build_sidebar(self):

        self.sidebar = ctk.CTkFrame(
            self.root, width=205, corner_radius=0, fg_color=SIDEBAR
        )

        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.sidebar.grid_propagate(False)

        ctk.CTkLabel(
            self.sidebar, text="CGIP.", font=("Segoe UI", 34, "bold"), text_color=CORAL
        ).pack(anchor="w", padx=25, pady=(32, 0))

        ctk.CTkLabel(
            self.sidebar,
            text="CREATIVE IMAGE STUDIO",
            font=("Segoe UI", 9, "bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=27, pady=(0, 40))

        self._section_label(self.sidebar, "WORKSPACE")

        self._nav_item("?", "Image Studio", True)

        self._nav_item(
            "?", "Sketch Effects", command=lambda: self.show_category("Sketch")
        )

        self._nav_item(
            "?", "Cartoon Effects", command=lambda: self.show_category("Cartoon")
        )

        self._nav_item("?", "Compare Studio", command=self.open_compare_studio)

        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).pack(
            fill="x", padx=22, pady=25
        )

        self._section_label(self.sidebar, "YOUR PROJECT")

        self._nav_item("▧", "Open Image", command=self.open_image)
        self._nav_item("↓", "Export Result", command=self.save_image)

        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).pack(
            fill="x", padx=22, pady=25
        )

        self._section_label(self.sidebar, "ABOUT")

        ctk.CTkLabel(
            self.sidebar,
            text="Image Processing\nComputer Graphics & Image Processing",
            font=("Segoe UI", 10),
            text_color=MUTED,
            justify="left",
        ).pack(anchor="w", padx=25)

        footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=22, pady=25)

        ctk.CTkLabel(
            footer, text="CGIP PROJECT", font=("Segoe UI", 10, "bold"), text_color=LIME
        ).pack(anchor="w")

        ctk.CTkLabel(
            footer,
            text="Creative image transformation",
            font=("Segoe UI", 9),
            text_color=MUTED,
        ).pack(anchor="w", pady=(3, 0))

    def open_compare_studio(self):
        if self.original is None:
            messagebox.showwarning(
                "No Image", "Please open an image before comparing effects."
            )
            return

        effect_name = self.current_mode.get()

        open_effect_comparison(
            self.root,
            self.original.copy(),
            lambda mode, img: self._apply_effect(mode, img),
            effect_name,
        )


    def _section_label(self, parent, text):

        ctk.CTkLabel(
            parent, text=text, font=("Segoe UI", 10, "bold"), text_color="#777777"
        ).pack(anchor="w", padx=25, pady=(5, 12))

    def _nav_item(self, icon, text, active=False, command=None):

        frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="#35352D" if active else "transparent",
            corner_radius=9,
        )

        frame.pack(fill="x", padx=13, pady=4)

        button = ctk.CTkButton(
            frame,
            text=f"{icon}     {text}",
            anchor="w",
            height=42,
            fg_color="transparent",
            hover_color="#34342E",
            text_color=TEXT if active else "#B9B9B9",
            font=("Segoe UI", 12, "bold" if active else "normal"),
            command=command if command else lambda: None,
        )

        button.pack(fill="x", padx=3, pady=2)

    # =====================================================
    # MAIN AREA
    # =====================================================

    def _build_main_area(self):

        self.main_area = ctk.CTkFrame(self.root, fg_color=BG, corner_radius=0)

        self.main_area.grid(row=0, column=1, sticky="nsew", padx=24, pady=20)

        self.main_area.grid_columnconfigure(0, weight=1)
        self.main_area.grid_rowconfigure(2, weight=1)

        self._build_topbar()
        self._build_gallery()
        self._build_editor()
        self._build_status()

    # =====================================================
    # TOP BAR
    # =====================================================

    def _build_topbar(self):

        topbar = ctk.CTkFrame(self.main_area, fg_color="transparent")

        topbar.grid(row=0, column=0, sticky="ew", pady=(0, 22))

        title_area = ctk.CTkFrame(topbar, fg_color="transparent")

        title_area.pack(side="left")

        ctk.CTkLabel(
            title_area,
            text="Make something different.",
            font=("Segoe UI", 26, "bold"),
            text_color=TEXT,
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_area,
            text="Transform ordinary photographs into creative artwork.",
            font=("Segoe UI", 12),
            text_color=MUTED,
        ).pack(anchor="w", pady=(4, 0))

        actions = ctk.CTkFrame(topbar, fg_color="transparent")

        actions.pack(side="right")

        ctk.CTkButton(
            actions,
            text="+  Open Image",
            width=125,
            height=42,
            fg_color=PANEL_LIGHT,
            hover_color="#414141",
            text_color=TEXT,
            command=self.open_image,
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            actions,
            text="↓  Export",
            width=115,
            height=42,
            fg_color=CORAL,
            hover_color=CORAL_HOVER,
            text_color="#FFFFFF",
            font=("Segoe UI", 12, "bold"),
            command=self.save_image,
        ).pack(side="left", padx=6)

    # =====================================================
    # EFFECT GALLERY
    # =====================================================

    def _build_gallery(self):

        gallery = ctk.CTkFrame(self.main_area, fg_color="transparent")

        gallery.grid(row=1, column=0, sticky="ew", pady=(0, 18))

        heading = ctk.CTkFrame(gallery, fg_color="transparent")

        heading.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            heading,
            text="EXPLORE EFFECTS",
            font=("Segoe UI", 12, "bold"),
            text_color=TEXT,
        ).pack(side="left")

        ctk.CTkLabel(
            heading,
            text="Choose a style to transform your image",
            font=("Segoe UI", 10),
            text_color=MUTED,
        ).pack(side="right")

        self.gallery_frame = ctk.CTkFrame(gallery, fg_color="transparent")

        self.gallery_frame.pack(fill="x")

        self.effect_cards = {}

        for index, effect in enumerate(EFFECTS):

            card = ctk.CTkFrame(
                self.gallery_frame,
                fg_color=PANEL,
                corner_radius=11,
                border_width=1,
                border_color=BORDER,
            )

            card.grid(row=0, column=index, sticky="nsew", padx=5)

            self.gallery_frame.grid_columnconfigure(index, weight=1)

            thumbnail = ctk.CTkLabel(
                card,
                text=self._effect_symbol(effect),
                font=("Segoe UI", 29, "bold"),
                text_color=CORAL,
                fg_color=CANVAS,
                corner_radius=8,
                height=75,
            )

            thumbnail.pack(fill="x", padx=7, pady=(7, 5))

            title = ctk.CTkLabel(
                card, text=effect, font=("Segoe UI", 10, "bold"), text_color=TEXT
            )

            title.pack(pady=(2, 7))

            button = ctk.CTkButton(
                card,
                text="SELECT",
                height=25,
                font=("Segoe UI", 9, "bold"),
                fg_color="#343434",
                hover_color=CORAL_HOVER,
                text_color=TEXT,
                command=lambda e=effect: self.select_effect(e),
            )

            button.pack(fill="x", padx=7, pady=(0, 8))

            self.effect_cards[effect] = {
                "card": card,
                "thumbnail": thumbnail,
                "button": button,
            }

        self._update_selected_card()

    def _effect_symbol(self, effect):

        symbols = {
            "Pencil Sketch": "✎",
            "Color Pencil Sketch": "✿",
            "Classic Cartoon": "◈",
            "K-Means Cartoon": "▦",
            "Oil Painting": "✺",
        }

        return symbols.get(effect, "✦")

    def _build_effect_gallery(self):
        """
        Generate real effect thumbnails from samples/image.png
        when that file exists.
        """

        sample_path = os.path.join("samples", "image.png")

        if not os.path.exists(sample_path):
            return

        img = cv2.imread(sample_path)

        if img is None:
            return

        img = resize_keep_aspect(img, 160, 110)

        for effect in EFFECTS:

            try:
                result = self._apply_effect(effect, img.copy())

                rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                pil = Image.fromarray(rgb)

                photo = ctk.CTkImage(light_image=pil, dark_image=pil, size=(160, 75))

                self.effect_cards[effect]["thumbnail"].configure(image=photo, text="")

                self.gallery_images.append(photo)

            except Exception as error:
                print(f"Thumbnail error ({effect}): {error}")

    def show_category(self, category):
        if category == "Sketch":
            effects = ["Pencil Sketch", "Color Pencil Sketch"]
        elif category == "Cartoon":
            effects = ["Classic Cartoon", "K-Means Cartoon"]
        else:
            effects = ["Oil Painting"]

        for widget in self.gallery_frame.winfo_children():
            widget.destroy()

        self.effect_cards.clear()

        for index, effect in enumerate(effects):
            card = ctk.CTkFrame(
                self.gallery_frame,
                fg_color=PANEL,
                corner_radius=11,
                border_width=1,
                border_color=BORDER,
            )
            card.grid(row=0, column=index, sticky="nsew", padx=5)
            self.gallery_frame.grid_columnconfigure(index, weight=1)

            thumbnail = ctk.CTkLabel(
                card,
                text=self._effect_symbol(effect),
                font=("Segoe UI", 29, "bold"),
                text_color=CORAL,
                fg_color=CANVAS,
                corner_radius=8,
                height=75,
            )
            thumbnail.pack(fill="x", padx=7, pady=(7, 5))

            ctk.CTkLabel(
                card,
                text=effect,
                font=("Segoe UI", 10, "bold"),
                text_color=TEXT,
            ).pack(pady=(2, 7))

            button = ctk.CTkButton(
                card,
                text="SELECT",
                height=25,
                command=lambda e=effect: self.select_effect(e),
            )
            button.pack(fill="x", padx=7, pady=(0, 8))

            self.effect_cards[effect] = {
                "card": card,
                "thumbnail": thumbnail,
                "button": button,
            }

        self.select_effect(effects[0])

    def select_effect(self, effect):

        self.current_mode.set(effect)
        self._update_selected_card()
        self.selected_effect_label.configure(text=effect)

        if self.original is not None:
            self.process_image()
        else:
            self.status.configure(text=f"Selected {effect} - Open an image")

    def _update_selected_card(self):

        for effect, widgets in self.effect_cards.items():

            selected = effect == self.current_mode.get()

            widgets["card"].configure(
                border_color=CORAL if selected else BORDER,
                border_width=2 if selected else 1,
            )

            widgets["button"].configure(
                fg_color=CORAL if selected else "#343434", text_color="#FFFFFF"
            )

    # =====================================================
    # EDITOR WORKSPACE
    # =====================================================

    def _build_editor(self):

        editor = ctk.CTkFrame(self.main_area, fg_color="transparent")

        editor.grid(row=2, column=0, sticky="nsew")

        editor.grid_columnconfigure(0, weight=1)
        editor.grid_columnconfigure(1, weight=0)
        editor.grid_rowconfigure(0, weight=1)

        self._build_canvas(editor)
        self._build_controls(editor)

    # =====================================================
    # IMAGE CANVAS
    # =====================================================

    def _build_canvas(self, parent):

        canvas_panel = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=13)

        canvas_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        canvas_panel.grid_columnconfigure(0, weight=1)
        canvas_panel.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(canvas_panel, fg_color="transparent")

        header.grid(row=0, column=0, sticky="ew", padx=17, pady=14)

        ctk.CTkLabel(
            header, text="YOUR CANVAS", font=("Segoe UI", 11, "bold"), text_color=TEXT
        ).pack(side="left")

        self.compare_check = ctk.CTkCheckBox(
            header,
            text="Before / After",
            variable=self.show_comparison,
            command=self.display_result,
            font=("Segoe UI", 10),
            text_color=MUTED,
            fg_color=CORAL,
            hover_color=CORAL_HOVER,
            checkmark_color="#FFFFFF",
        )

        self.compare_check.pack(side="right")

        self.image_frame = ctk.CTkFrame(canvas_panel, fg_color=CANVAS, corner_radius=9)

        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)

        self.image_display = ctk.CTkLabel(
            self.image_frame, text="", fg_color="transparent"
        )

        self.image_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.empty_state = ctk.CTkFrame(self.image_frame, fg_color="transparent")

        self.empty_state.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            self.empty_state, text="✦", font=("Segoe UI", 46), text_color=CORAL
        ).pack()

        ctk.CTkLabel(
            self.empty_state,
            text="Your next masterpiece starts here.",
            font=("Segoe UI", 16, "bold"),
            text_color=TEXT,
        ).pack(pady=(8, 5))

        ctk.CTkLabel(
            self.empty_state,
            text="Open an image and choose an effect to begin.",
            font=("Segoe UI", 11),
            text_color=MUTED,
        ).pack()

        ctk.CTkButton(
            self.empty_state,
            text="＋  Choose Image",
            width=145,
            height=38,
            fg_color=CORAL,
            hover_color=CORAL_HOVER,
            command=self.open_image,
        ).pack(pady=17)

    # =====================================================
    # RIGHT CONTROL PANEL
    # =====================================================

    def _build_controls(self, parent):

        controls = ctk.CTkScrollableFrame(
            parent,
            width=275,
            fg_color=PANEL,
            corner_radius=13,
            scrollbar_button_color=CORAL,
            scrollbar_button_hover_color=CORAL_HOVER,
        )

        controls.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(
            controls,
            text="CREATIVE CONTROLS",
            font=("Segoe UI", 13, "bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=18, pady=(20, 5))

        ctk.CTkLabel(
            controls, text="Selected effect", font=("Segoe UI", 10), text_color=MUTED
        ).pack(anchor="w", padx=18, pady=(13, 5))

        self.selected_effect_label = ctk.CTkLabel(
            controls,
            text=self.current_mode.get(),
            font=("Segoe UI", 13, "bold"),
            text_color=CORAL,
            fg_color="#30302B",
            corner_radius=8,
            height=39,
        )

        self.selected_effect_label.pack(fill="x", padx=17)

        self._separator(controls)

        ctk.CTkLabel(
            controls, text="ADJUSTMENTS", font=("Segoe UI", 11, "bold"), text_color=TEXT
        ).pack(anchor="w", padx=18, pady=(0, 8))

        self._add_slider(controls, "Blur Kernel", self.blur_ksize, 3, 51)
        self._add_slider(controls, "Bilateral Filter", self.bilateral_d, 5, 15)
        self._add_slider(controls, "Sigma Color", self.sigma_color, 20, 150)
        self._add_slider(controls, "Block Size", self.block_size, 3, 15)
        self._add_slider(controls, "Color Clusters", self.k_clusters, 4, 16)

        self._separator(controls)

        ctk.CTkButton(
            controls,
            text="✦ APPLY EFFECT",
            height=45,
            fg_color=CORAL,
            hover_color=CORAL_HOVER,
            text_color="#FFFFFF",
            font=("Segoe UI", 12, "bold"),
            command=self.process_image,
        ).pack(fill="x", padx=17, pady=(0, 9))

        ctk.CTkButton(
            controls,
            text="Reset Adjustments",
            height=35,
            fg_color="#343434",
            hover_color="#454545",
            text_color=TEXT,
            command=self.reset_settings,
        ).pack(fill="x", padx=17, pady=(0, 10))

        ctk.CTkLabel(
            controls, text="TIP", font=("Segoe UI", 10, "bold"), text_color=LIME
        ).pack(anchor="w", padx=18, pady=(12, 3))

        ctk.CTkLabel(
            controls,
            text="Experiment with different effects and parameter values to create unique artistic results.",
            font=("Segoe UI", 10),
            text_color=MUTED,
            justify="left",
            wraplength=225,
        ).pack(anchor="w", padx=18, pady=(0, 15))

    def _separator(self, parent):

        ctk.CTkFrame(parent, height=1, fg_color=BORDER).pack(fill="x", padx=18, pady=17)

    # =====================================================
    # ADJUSTMENT SLIDERS
    # =====================================================

    def _add_slider(self, parent, label, variable, minimum, maximum):

        frame = ctk.CTkFrame(parent, fg_color="transparent")

        frame.pack(fill="x", padx=18, pady=7)

        header = ctk.CTkFrame(frame, fg_color="transparent")

        header.pack(fill="x")

        ctk.CTkLabel(
            header, text=label, font=("Segoe UI", 10), text_color="#C5C5C5"
        ).pack(side="left")

        value_label = ctk.CTkLabel(
            header,
            text=str(variable.get()),
            font=("Segoe UI", 10, "bold"),
            text_color=LIME,
        )

        value_label.pack(side="right")

        def update_value(value):

            number = int(float(value))

            if variable in (self.blur_ksize, self.block_size):

                if number % 2 == 0:
                    number += 1

                number = min(number, maximum)

            variable.set(number)
            value_label.configure(text=str(number))

            # Reprocess after the user pauses slider movement.
            if self.original is not None:
                if getattr(self, "_slider_after_id", None) is not None:
                    try:
                        self.root.after_cancel(self._slider_after_id)
                    except Exception:
                        pass

                self._slider_after_id = self.root.after(
                    350, self.process_image
                )

        slider = ctk.CTkSlider(
            frame,
            from_=minimum,
            to=maximum,
            number_of_steps=maximum - minimum,
            height=16,
            progress_color=CORAL,
            button_color=LIME,
            button_hover_color="#D8EF91",
            command=update_value,
        )

        slider.set(variable.get())

        slider.pack(fill="x", pady=(5, 0))

        self.slider_widgets[id(variable)] = {"slider": slider, "label": value_label}

    # =====================================================
    # IMAGE PROCESSING
    # =====================================================

    def _apply_effect(self, mode, img):

        if mode == "Pencil Sketch":

            return pencil_sketch(img, self.blur_ksize.get())

        elif mode == "Color Pencil Sketch":

            return color_pencil_sketch(img, self.blur_ksize.get())

        elif mode == "Classic Cartoon":

            return cartoonize(
                img,
                d=self.bilateral_d.get(),
                sigma_color=self.sigma_color.get(),
                sigma_space=self.sigma_color.get(),
                block_size=self.block_size.get(),
            )

        elif mode == "K-Means Cartoon":

            return cartoonize_kmeans(
                img,
                k=self.k_clusters.get(),
                d=self.bilateral_d.get(),
                sigma=self.sigma_color.get(),
            )

        elif mode == "Oil Painting":

            return oil_painting_effect(img)

        return img

    def process_image(self):

        if self.original is None:

            messagebox.showwarning("No Image", "Please open an image first.")

            return

        try:

            self.status.configure(text="Processing your image...")

            self.root.update_idletasks()

            mode = self.current_mode.get()

            self.processed = self._apply_effect(mode, self.original.copy())

            self._update_selected_card()

            self.selected_effect_label.configure(text=mode)

            self.display_result()

            self.status.configure(text=f"✓  Applied {mode} successfully")

        except Exception as error:

            messagebox.showerror("Processing Error", str(error))

            self.status.configure(text="Processing failed.")

    # =====================================================
    # IMAGE DISPLAY
    # =====================================================

    def _show_image(self, img):

        if len(img.shape) == 2:

            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        else:

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb)

        available_width = max(self.image_frame.winfo_width() - 35, 450)

        available_height = max(self.image_frame.winfo_height() - 35, 300)

        pil.thumbnail((available_width, available_height), Image.Resampling.LANCZOS)

        self.preview_image = ctk.CTkImage(
            light_image=pil, dark_image=pil, size=pil.size
        )

        self.image_display.configure(image=self.preview_image, text="")

    def display_original(self):

        if self.original is None:
            return

        self.empty_state.place_forget()

        self._show_image(self.original)

    def display_result(self):

        if self.original is None:
            return

        self.empty_state.place_forget()

        if self.processed is None:

            self.display_original()
            return

        if self.show_comparison.get():

            original = resize_keep_aspect(self.original, 500, 500)

            processed = resize_keep_aspect(self.processed, 500, 500)

            original = self._to_three_channels(original)
            processed = self._to_three_channels(processed)

            gap = 12

            height = max(original.shape[0], processed.shape[0])

            width = original.shape[1] + processed.shape[1] + gap

            comparison = np.ones((height, width, 3), dtype=np.uint8) * 35

            comparison[: original.shape[0], : original.shape[1]] = original

            comparison[: processed.shape[0], original.shape[1] + gap :] = processed

            self._show_image(comparison)

        else:

            self._show_image(self.processed)

    def _to_three_channels(self, img):

        if len(img.shape) == 2:

            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    # =====================================================
    # OPEN IMAGE
    # =====================================================

    def open_image(self):

        path = filedialog.askopenfilename(
            title="Choose your image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        img = cv2.imread(path)

        if img is None:

            messagebox.showerror("Image Error", "Unable to open this image.")

            return

        self.original = img
        self.processed = None
        self.image_name = os.path.basename(path)

        self.show_comparison.set(False)
        self.compare_check.deselect()

        self.display_original()

        self.status.configure(
            text=f"Loaded: {self.image_name}  |  " f"{img.shape[1]} × {img.shape[0]}"
        )

    # =====================================================
    # RESET
    # =====================================================

    def reset_settings(self):

        defaults = [
            (self.blur_ksize, 21),
            (self.bilateral_d, 9),
            (self.sigma_color, 75),
            (self.block_size, 9),
            (self.k_clusters, 8),
        ]

        for variable, value in defaults:

            variable.set(value)

            widgets = self.slider_widgets[id(variable)]

            widgets["slider"].set(value)

            widgets["label"].configure(text=str(value))

        self.status.configure(
            text="Adjustments reset. Click Apply Effect to process again."
        )

    # =====================================================
    # EXPORT
    # =====================================================

    def save_image(self):

        if self.processed is None:

            messagebox.showwarning(
                "No Result", "Please apply an effect before exporting."
            )

            return

        path = filedialog.asksaveasfilename(
            title="Export your artwork",
            defaultextension=".png",
            initialfile="CGIP_artwork.png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
        )

        if not path:
            return

        success = cv2.imwrite(path, self.processed)

        if success:

            self.status.configure(text=f"✓  Exported: {os.path.basename(path)}")

            messagebox.showinfo(
                "Export Complete", f"Your artwork has been saved!\n\n{path}"
            )

        else:

            messagebox.showerror("Export Error", "Could not save the image.")

    # =====================================================
    # STATUS AND WINDOW
    # =====================================================

    def _build_status(self):

        self.status = ctk.CTkLabel(
            self.main_area,
            text="Ready  •  Open an image to begin",
            font=("Segoe UI", 10),
            text_color=MUTED,
            anchor="w",
        )

        self.status.grid(row=3, column=0, sticky="ew", pady=(12, 0))

    def _on_resize(self, event):

        if event.widget == self.root and self.original is not None:

            self.root.after(150, self.display_result)


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================


def main():

    root = ctk.CTk()

    app = CartoonSketchApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()
