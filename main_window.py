import sys
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog, QMessageBox
import os
import cv2
import numpy as np
from main import load_image_gray, apply_roi, low_pass_filter, mottling_metrics

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'main_window.ui'), self)
        self.buttonBrowse.clicked.connect(self.select_folder)
        self.buttonProcess.clicked.connect(self.process_images)
        self.lineEditFolder.setText(os.path.join(os.path.dirname(__file__), 'Imagens', 'Testar'))
        self.tableResults.setColumnCount(4)
        self.tableResults.setHorizontalHeaderLabels(['Arquivo', 'Média', 'Std', 'MI (%)'])
        self.imagePreview.clear()
        self.tableResults.itemSelectionChanged.connect(self.display_selected_images)
        self._image_cache = {}

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Selecionar pasta de imagens')
        if folder:
            self.lineEditFolder.setText(folder)

    def process_images(self):
        folder = self.lineEditFolder.text()
        kernel_x = self.spinBoxKernelX.value()
        kernel_y = self.spinBoxKernelY.value()
        sigma = self.doubleSpinBoxSigma.value()
        kernel = (kernel_x, kernel_y)
        results = []
        self.tableResults.setRowCount(0)
        for file in os.listdir(folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                path = os.path.join(folder, file)
                try:
                    img = load_image_gray(path)
                    img_roi = apply_roi(img, None)
                    filtered = low_pass_filter(img_roi, kernel, sigma)
                    mean, std, mi = mottling_metrics(filtered)
                    results.append((file, mean, std, mi))
                    row = self.tableResults.rowCount()
                    self.tableResults.insertRow(row)
                    self.tableResults.setItem(row, 0, QtWidgets.QTableWidgetItem(file))
                    self.tableResults.setItem(row, 1, QtWidgets.QTableWidgetItem(f'{mean:.2f}'))
                    self.tableResults.setItem(row, 2, QtWidgets.QTableWidgetItem(f'{std:.2f}'))
                    self.tableResults.setItem(row, 3, QtWidgets.QTableWidgetItem(f'{mi:.2f}'))
                    # Cache imagens para exibição posterior
                    diff = cv2.normalize(abs(filtered - np.mean(filtered)), None, 0, 255, cv2.NORM_MINMAX)
                    self._image_cache[file] = {
                        'original': img,
                        'filtered': filtered,
                        'diff': diff
                    }
                    self.show_image_preview(filtered)
                except Exception as e:
                    QMessageBox.critical(self, 'Erro', f'Erro ao processar {file}: {str(e)}')
        # ...existing code...
    def display_selected_images(self):
        selected = self.tableResults.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        file_item = self.tableResults.item(row, 0)
        if not file_item:
            return
        file = file_item.text()
        images = self._image_cache.get(file)
        if not images:
            return
        # Exibe as três imagens em sequência
        self.show_multiple_previews(images['original'], images['filtered'], images['diff'])

    def show_multiple_previews(self, original, filtered, diff):
        import matplotlib.pyplot as plt
        import io
        from PyQt6.QtGui import QPixmap
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(filtered, cmap='gray')
        axes[1].set_title('Filtrada')
        axes[1].axis('off')
        axes[2].imshow(diff, cmap='inferno')
        axes[2].set_title('Mapa de Não-Uniformidade')
        axes[2].axis('off')
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read(), 'PNG')
        self.imagePreview.setPixmap(pixmap)
        plt.close(fig)

    def show_image_preview(self, img):
        # Converte para QPixmap
        import matplotlib.pyplot as plt
        import io
        from PyQt6.QtGui import QPixmap
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read(), 'PNG')
        self.imagePreview.setPixmap(pixmap)
        plt.close(fig)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
