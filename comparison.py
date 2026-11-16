import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import customtkinter as ctk
from PIL import Image

MODEL_PATH = Path(__file__).parent / "models" / "AnimeGANv2_Hayao.onnx"

_model_session = None


def get_model():
    global _model_session

    if _model_session is None:
        _model_session = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )

    return _model_session


def run_ai_cartoon(image):
    session = get_model()
    input_name = session.get_inputs()[0].name

    height, width = image.shape[:2]

    # Resize to a manageable size, preserving aspect ratio.
    scale = min(512 / width, 512 / height)

    new_width = max(32, round(width * scale / 32) * 32)
    new_height = max(32, round(height * scale / 32) * 32)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

    tensor = resized.astype(np.float32) / 127.5 - 1.0
    tensor = np.expand_dims(tensor, axis=0)

    result = session.run(None, {input_name: tensor})[0]

    result = np.clip((result[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def create_image_card(parent, title, image, elapsed, max_size=(330, 390)):
    height, width = image.shape[:2]
    card = ctk.CTkFrame(parent, fg_color="#222222", corner_radius=12)

    card.grid_columnconfigure(0, weight=1)
    card.grid_rowconfigure(1, weight=1)

    ctk.CTkLabel(
        card, text=title, font=("Segoe UI", 16, "bold"), text_color="#F4EFE6"
    ).grid(row=0, column=0, pady=(14, 8))

    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(rgb)
    pil.thumbnail(max_size, Image.Resampling.LANCZOS)

    photo = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)

    image_label = ctk.CTkLabel(card, text="", image=photo)
    image_label.grid(row=1, column=0, padx=10, pady=10)

    # Keep a reference so Tkinter does not discard the image.
    image_label.image = photo

    height, width = image.shape[:2]

    ctk.CTkLabel(
        card,
        text=f"{elapsed:.2f} seconds  |  {width} × {height}",
        font=("Segoe UI", 10),
        text_color="#C7E56B",
    ).grid(row=2, column=0, pady=(5, 14))

    return card


def open_compare_studio(parent, original, effect_runner, effect_names):

    window = ctk.CTkToplevel(parent)
    window.title("CGIP | Effects Comparison Studio")
    window.geometry("1250x850")
    window.minsize(950, 650)
    window.configure(fg_color="#151515")
    window.transient(parent)

    ctk.CTkLabel(
        window,
        text="EFFECTS COMPARISON STUDIO",
        font=("Segoe UI", 27, "bold"),
        text_color="#FF704F",
    ).pack(pady=(20, 4))

    ctk.CTkLabel(
        window,
        text="Compare every transformation using the same original photograph",
        font=("Segoe UI", 12),
        text_color="#999999",
    ).pack(pady=(0, 12))

    status = ctk.CTkLabel(
        window,
        text="Preparing effects...",
        font=("Segoe UI", 12),
        text_color="#C7E56B",
    )
    status.pack(pady=8)

    cards_frame = ctk.CTkScrollableFrame(
        window,
        fg_color="transparent",
        corner_radius=0,
    )
    cards_frame.pack(fill="both", expand=True, padx=15, pady=10)

    for column in range(3):
        cards_frame.grid_columnconfigure(column, weight=1)

    results = [("Original Image", original.copy(), 0.0)]

    def show_next_effect(index):

        if index < len(effect_names):
            mode = effect_names[index]

            try:
                status.configure(
                    text=f"Processing {mode}... ({index + 1}/{len(effect_names)})"
                )
                window.update_idletasks()

                start = time.perf_counter()
                output = effect_runner(mode, original.copy())
                elapsed = time.perf_counter() - start

                results.append((mode, output, elapsed))

                window.after(50, lambda: show_next_effect(index + 1))

            except Exception as error:
                status.configure(
                    text=f"{mode} failed: {error}",
                    text_color="#FF704F",
                )
            return

        try:
            status.configure(text="Running AI AnimeGANv2...")
            window.update_idletasks()

            start = time.perf_counter()
            ai_result = run_ai_cartoon(original.copy())
            ai_time = time.perf_counter() - start

            results.append(("AI AnimeGANv2", ai_result, ai_time))

            for index, (title, image, elapsed) in enumerate(results):
                card = create_image_card(cards_frame, title, image, elapsed)

                row = index // 3
                column = index % 3

                card.grid(
                    row=row,
                    column=column,
                    sticky="nsew",
                    padx=8,
                    pady=8,
                )

            status.configure(
                text=f"All {len(results)} comparisons completed!",
                text_color="#C7E56B",
            )

        except Exception as error:
            status.configure(
                text=f"AI comparison failed: {error}",
                text_color="#FF704F",
            )

    window.after(150, lambda: show_next_effect(0))


def open_effect_comparison(parent, original, effect_runner, effect_name):
    """Compare one selected effect: Original, OpenCV and available ONNX."""

    from core.onnx_engine import ONNXImageEngine

    window = ctk.CTkToplevel(parent)
    window.title("CGIP | Original vs OpenCV vs ONNX")
    window.geometry("1350x780")
    window.minsize(1000, 650)
    window.configure(fg_color="#151515")
    window.transient(parent)

    ctk.CTkLabel(
        window,
        text=f"{effect_name.upper()} | THREE-WAY COMPARISON",
        font=("Segoe UI", 23, "bold"),
        text_color="#FF704F",
    ).pack(pady=(18, 5))

    status = ctk.CTkLabel(
        window,
        text="Processing the same input image with each available approach...",
        text_color="#C7E56B",
    )
    status.pack(pady=8)

    cards = ctk.CTkFrame(window, fg_color="transparent")
    cards.pack(fill="both", expand=True, padx=12, pady=12)

    for col in range(3):
        cards.grid_columnconfigure(col, weight=1)
    cards.grid_rowconfigure(0, weight=1)

    def show_image_card(column, title, image, elapsed, note=""):
        card = ctk.CTkFrame(cards, fg_color="#222222", corner_radius=12)
        card.grid(row=0, column=column, sticky="nsew", padx=7, pady=7)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            card, text=title, font=("Segoe UI", 16, "bold"),
            text_color="#F4EFE6"
        ).grid(row=0, column=0, pady=(14, 8))

        if image is not None:
            if image.ndim == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            pil = Image.fromarray(rgb)
            pil.thumbnail((390, 510), Image.Resampling.LANCZOS)
            photo = ctk.CTkImage(
                light_image=pil, dark_image=pil, size=pil.size
            )
            label = ctk.CTkLabel(card, text="", image=photo)
            label.image = photo
            label.grid(row=1, column=0, padx=8, pady=8)

            detail = f"{elapsed:.3f} seconds"
        else:
            ctk.CTkLabel(
                card,
                text="ONNX MODEL\nNOT AVAILABLE\n\n"
                     "A compatible trained model\nis required for this style.",
                font=("Segoe UI", 15),
                text_color="#FFB39F",
                justify="center",
            ).grid(row=1, column=0, padx=8, pady=8)
            detail = note or "Not evaluated"

        ctk.CTkLabel(
            card, text=detail, font=("Segoe UI", 10),
            text_color="#C7E56B"
        ).grid(row=2, column=0, pady=(5, 14))

    try:
        window.update_idletasks()

        show_image_card(0, "ORIGINAL", original, 0.0)

        start = time.perf_counter()
        cv_result = effect_runner(effect_name, original.copy())
        cv_elapsed = time.perf_counter() - start
        show_image_card(1, "OPENCV", cv_result, cv_elapsed)

        # Only use a verified cartoon model for cartoon comparisons.
        model_path = None
        model_label = "ONNX"

        if effect_name in ("Classic Cartoon", "K-Means Cartoon"):
            project_dir = Path(__file__).resolve().parent
            candidate = project_dir / "models" / "CartoonGAN_Hayao.onnx"
            if candidate.exists():
                model_path = candidate
                model_label = "ONNX CartoonGAN Hayao"

        if model_path is not None:
            status.configure(text="Running ONNX inference...")
            window.update_idletasks()
            engine = ONNXImageEngine(model_path)
            ai_result, ai_elapsed = engine.run(original.copy())
            show_image_card(2, model_label, ai_result, ai_elapsed)
        else:
            show_image_card(
                2, "ONNX UNAVAILABLE", None, 0.0,
                "No verified ONNX model for this effect"
            )

        status.configure(
            text="Comparison complete. Timings measure processing/inference only.",
            text_color="#C7E56B",
        )

    except Exception as error:
        status.configure(
            text=f"Comparison failed: {error}",
            text_color="#FF704F",
        )
