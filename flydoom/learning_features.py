"""Versioned contrast/motion input adapter; deliberately not a biological retina."""

import numpy as np

from flydoom.bridge import NeuralController, encode_frame


class ContrastMotionController(NeuralController):
    """Preserve the calibrated input bound while exposing contrast and motion."""

    def __init__(self, source, mix=0.2):
        self.__dict__.update(source.__dict__)
        if not np.isfinite(mix) or not 0 <= mix <= .4:
            raise ValueError("Encoder mix must be in [0, 0.4]")
        self.encoder_mix = float(mix)
        self.previous_image = None

    def reset(self):
        super().reset()
        self.previous_image = None

    def transform(self, frame):
        rgb = np.asarray(frame)
        encode_frame(rgb)  # Validate before changing temporal state.
        gray = rgb.astype(np.float32).mean(2) / 255
        contrast = np.abs(gray - gray.mean())
        motion = np.zeros_like(gray) if self.previous_image is None else np.abs(gray - self.previous_image)
        self.previous_image = gray.copy()
        signal = np.clip(.5 + contrast + motion, 0, 1)
        mixed = (1 - self.encoder_mix) * gray + self.encoder_mix * signal
        return np.repeat(np.rint(mixed * 255).astype(np.uint8)[..., None], 3, axis=2)

    def decide(self, frame):
        # Zero is an exact legacy path, including integer RGB rounding behavior.
        if self.encoder_mix == 0:
            return super().decide(frame)
        return super().decide(self.transform(frame))
