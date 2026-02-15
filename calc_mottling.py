from pathlib import Path

from mottling import mi_from_image

# Gaussian blur parameters
GAUSSIAN_KERNEL = (21, 21)
GAUSSIAN_SIGMA = 5


def process_folder(folder):
    results = []

    folder_path = Path(folder)
    if not folder_path.exists():
        return results

    for path in folder_path.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".bmp"):
            mean, std, mi = mi_from_image(path, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
            results.append((path.name, mean, std, mi))

            print(
                f"{path.name:30s} | Média: {mean:.2f} | Std: {std:.2f} | MI: {mi:.2f}%"
            )

    return results


if __name__ == "__main__":
    folder_path = "Imagens/Testar"  # pasta com as imagens
    results = process_folder(folder_path)
