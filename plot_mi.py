import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================
# PARÂMETROS DO MÉTODO
# ============================

# Tamanho do filtro passa-baixa (remove ruído fino)
GAUSSIAN_KERNEL = (21, 21)
GAUSSIAN_SIGMA = 5

# Região de interesse (ROI) – usar None para imagem inteira
ROI = None  # Exemplo: (x, y, w, h)

# ============================
# FUNÇÕES PRINCIPAIS
# ============================

def load_image_gray(path):
    """Carrega imagem e converte para escala de cinza"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Erro ao carregar imagem: {path}")
    return img.astype(np.float32)


def apply_roi(img, roi):
    """Aplica região de interesse"""
    if roi is None:
        return img
    x, y, w, h = roi
    return img[y:y+h, x:x+w]


def low_pass_filter(img, kernel, sigma):
    """Filtro passa-baixa para eliminar ruído fino"""
    return cv2.GaussianBlur(img, kernel, sigma)


def mottling_metrics(img):
    """
    Calcula métricas quantitativas de mottling:
    - Média da luminância
    - Desvio-padrão
    - Mottling Index (coeficiente de variação)
    """
    mean = np.mean(img)
    std = np.std(img)
    mi = std / mean * 100  # percentual
    return mean, std, mi


def plot_results(original, filtered, name, mi):
    """Exibe resultados visuais"""
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Imagem Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # plt.title("Imagem Filtrada")
    # plt.imshow(filtered, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    diff = cv2.normalize(abs(filtered - np.mean(filtered)),
                          None, 0, 255, cv2.NORM_MINMAX)
    plt.title("Mapa de Não-Uniformidade")
    plt.imshow(diff, cmap='inferno')
    plt.axis('off')

    plt.suptitle(f"{name}  |  Mottling Index = {mi:.2f}%")
    plt.tight_layout()
    plt.show()


# ============================
# PROCESSAMENTO EM LOTE
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

            results.append((file, mean, std, mi))

            print(f"{file:30s} | Média: {mean:.2f} | Std: {std:.2f} | MI: {mi:.2f}%")

            plot_results(cv2.imread(path, cv2.IMREAD_COLOR_RGB), filtered, file, mi)

    return results


# ============================
# EXECUÇÃO
# ============================

if __name__ == "__main__":
    folder_path = "Imagens/Testar"  # pasta com as imagens
    results = process_folder(folder_path)
