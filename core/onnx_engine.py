import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class ONNXImageEngine:
    """Reusable ONNX image inference engine."""

    def __init__(self, model_path, providers=None):
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            self.model_path = project_root / self.model_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}"
            )

        self.providers = providers or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self.providers,
        )

        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]

        self.input_name = self.input_info.name
        self.output_name = self.output_info.name

        if self.input_info.type != "tensor(float)":
            raise TypeError("This engine currently supports float32 inputs.")

        shape = self.input_info.shape

        if len(shape) != 4:
            raise ValueError(
                f"Expected a 4D image input, received: {shape}"
            )

        if shape[-1] == 3:
            self.layout = "NHWC"
        elif shape[1] == 3:
            self.layout = "NCHW"
        else:
            raise ValueError(
                f"Cannot determine RGB channel layout: {shape}"
            )

    def preprocess(self, image, max_side=512):
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        height, width = image.shape[:2]

        scale = min(max_side / width, max_side / height, 1.0)

        new_width = max(32, round(width * scale / 32) * 32)
        new_height = max(32, round(height * scale / 32) * 32)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(
            rgb,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )

        tensor = resized.astype(np.float32) / 127.5 - 1.0

        if self.layout == "NCHW":
            tensor = np.transpose(tensor, (2, 0, 1))

        return np.expand_dims(tensor, axis=0)

    def postprocess(self, output):
        result = np.asarray(output[0], dtype=np.float32)

        if result.ndim != 3:
            raise ValueError(
                f"Expected 3D image output, received {result.shape}"
            )

        if result.shape[-1] == 3:
            pass
        elif result.shape[0] == 3:
            result = np.transpose(result, (1, 2, 0))
        else:
            raise ValueError(
                f"Cannot determine output channel layout: {result.shape}"
            )

        result = np.clip((result + 1.0) * 127.5, 0, 255)
        result = result.astype(np.uint8)

        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def run(self, image):
        tensor = self.preprocess(image)

        start = time.perf_counter()

        output = self.session.run(
            [self.output_name],
            {self.input_name: tensor},
        )[0]

        elapsed = time.perf_counter() - start

        result = self.postprocess(output)

        return result, elapsed
