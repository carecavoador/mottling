import cv2
import numpy as np
from pathlib import Path


def load_grayscale_image(path: Path) -> np.ndarray:
    """Loads the image in grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Erro ao carregar imagem: {path}")
    return img.astype(np.float32)


def apply_gaussian_blur(img: np.ndarray, kernel: tuple, sigma: float) -> np.ndarray:
    """Applies Gaussian blur to the image."""
    return cv2.GaussianBlur(img, kernel, sigma)


def calculate_mottling_index(img: np.ndarray) -> tuple[float, float, float]:
    """
    Calculates the mottling index based on the mean and standard deviation of the image.
    """
    mean = np.mean(img)
    std = np.std(img)
    mi = std / mean * 100
    return float(mean), float(std), float(mi)


def mi_from_image(
    path: Path, kernel: tuple, sigma: float
) -> tuple[float, float, float]:
    """Calculates the mottling index directly from an image path."""
    img = load_grayscale_image(path)
    filtered = apply_gaussian_blur(img, kernel, sigma)
    return calculate_mottling_index(filtered)
