import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft2, fftshift

# ============================
# PARÂMETROS GLOBAIS
# ============================

GAUSSIAN_KERNEL = (21, 21)
GAUSSIAN_SIGMA = 5

# Tamanho físico da imagem (mm)
# IMPORTANTE: ajuste conforme sua resolução de captura
PIXEL_SIZE_MM = 0.02  # ex: 0,02 mm/pixel

ROI = None  # (x, y, w, h)

# ============================
# FUNÇÕES BÁSICAS
# ============================

def load_image_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Erro ao carregar imagem: {path}")
    return img.astype(np.float32)


def apply_roi(img, roi):
    if roi is None:
        return img
    x, y, w, h = roi
    return img[y:y+h, x:x+w]


def low_pass_filter(img, kernel, sigma):
    return cv2.GaussianBlur(img, kernel, sigma)


def mottling_metrics(img):
    mean = np.mean(img)
    std = np.std(img)
    mi = std / mean * 100
    return mean, std, mi

# ============================
# FFT 2D
# ============================

def fft_analysis(img, pixel_size_mm):
    """
    Retorna:
    - espectro FFT normalizado
    - curva radial média
    - vetor de frequências espaciais (ciclos/mm)
    """

    img = img - np.mean(img)

    F = fftshift(fft2(img))
    magnitude = np.abs(F)

    h, w = img.shape
    cy, cx = h // 2, w // 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r.astype(np.int32)

    radial_sum = np.bincount(r.ravel(), magnitude.ravel())
    radial_count = np.bincount(r.ravel())
    radial_profile = radial_sum / radial_count

    freq_pixel = np.arange(len(radial_profile)) / (max(h, w) * pixel_size_mm)

    return magnitude, radial_profile, freq_pixel

# ============================
# DIAGNÓSTICO FÍSICO
# ============================

def dominant_wavelength(freq, spectrum):
    idx = np.argmax(spectrum[1:]) + 1  # ignora DC
    f_dom = freq[idx]
    wavelength = 1 / f_dom if f_dom > 0 else np.inf
    return f_dom, wavelength

# ============================
# VISUALIZAÇÃO
# ============================

def plot_fft_results(img, fft_mag, freq, spectrum, mi, wavelength_mm, name):

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.title("Imagem Filtrada")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Espectro FFT (log)")
    plt.imshow(np.log(fft_mag + 1), cmap='inferno')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Energia espectral radial")
    plt.plot(freq, spectrum)
    plt.xlabel("Frequência espacial (ciclos/mm)")
    plt.ylabel("Magnitude média")
    plt.grid(True)

    plt.suptitle(
        f"{name} | MI = {mi:.2f}% | λ dominante ≈ {wavelength_mm:.2f} mm",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()

# ============================
# PIPELINE PRINCIPAL
# ============================

def process_folder(folder):
    results = []

    for file in os.listdir(folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            path = os.path.join(folder, file)

            img = load_image_gray(path)
            img = apply_roi(img, ROI)
            filtered = low_pass_filter(img, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)

            mean, std, mi = mottling_metrics(filtered)

            fft_mag, spectrum, freq = fft_analysis(filtered, PIXEL_SIZE_MM)
            f_dom, wavelength = dominant_wavelength(freq, spectrum)

            results.append((file, mi, wavelength))

            print(f"{file:30s} | MI = {mi:.2f}% | λ dominante ≈ {wavelength:.2f} mm")

            plot_fft_results(filtered, fft_mag, freq, spectrum, mi, wavelength, file)

    return results

# ============================
# EXECUÇÃO
# ============================

if __name__ == "__main__":
    folder_path = r"C:\Users\erodr\src\mottling\Imagens\Testar"
    results = process_folder(folder_path)
