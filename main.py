from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt
import sys
import numpy as np
import io
import base64
from PIL import Image, ImageQt
from predictor import Predictor

class Canvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 300)
        self.setStyleSheet("background: white; border: 2px solid #bbb; border-radius: 16px; margin: 12px;")
        self.pixmap = QPixmap(self.size())
        self.pixmap.fill(Qt.white)
        self.setPixmap(self.pixmap)
        self.drawing = False
        self.last_point = None
        self.setAlignment(Qt.AlignCenter)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing and self.last_point is not None:
            painter = QPainter(self.pixmap)
            pen = QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)  # Thicker pen
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.setPixmap(self.pixmap)

    def mouseReleaseEvent(self, event):
        self.drawing = False
        self.last_point = None

    def clear(self):
        self.pixmap.fill(Qt.white)
        self.setPixmap(self.pixmap)

    def get_image(self):
        return self.pixmap.toImage()

class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        loadUi('main_window.ui', self)
        self.predictor = Predictor()
        self.setup_ui()

    def setup_ui(self):
        # Replace canvasLabel with a real drawing canvas
        canvas_label = self.findChild(QLabel, 'canvasLabel')
        if canvas_label is not None:
            parent_layout = canvas_label.parent().layout()
            if parent_layout is not None:
                # Remove the old QLabel
                parent_layout.removeWidget(canvas_label)
                canvas_label.hide()
                # Add the custom Canvas widget
                self.canvas = Canvas(self)
                # Center the canvas in the layout
                canvas_container = QWidget(self)
                hbox = QHBoxLayout(canvas_container)
                hbox.addStretch(1)
                hbox.addWidget(self.canvas, alignment=Qt.AlignCenter)
                hbox.addStretch(1)
                hbox.setContentsMargins(0, 0, 0, 0)
                parent_layout.insertWidget(3, canvas_container)
            else:
                self.canvas = Canvas(self)
        else:
            self.canvas = Canvas(self)
        self.clearButton.clicked.connect(self.clear_canvas)
        self.predictButton.clicked.connect(self.predict_canvas)
        self.uploadButton.clicked.connect(self.upload_and_predict)
        # Prettify main window and buttons
        self.setStyleSheet("""
            QMainWindow { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f6f7fb, stop:1 #e3e9f7); border-radius: 18px; }
            QWidget { border-radius: 18px; }
            QPushButton { font-size: 28px; border-radius: 32px; padding: 22px 60px; background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #43b04a, stop:1 #388e3c); color: white; font-weight: bold; box-shadow: 0 4px 16px #0002; }
            QPushButton:hover { background: #2e7d32; }
            QLabel#titleLabel { font-size: 36px; font-weight: bold; color: #222; margin: 24px; }
            QLabel#instructionsLabel { background: #f8f9fa; border-radius: 16px; padding: 16px; font-size: 18px; color: #333; }
            QLabel#resultLabel { font-size: 40px; color: #222; margin-top: 24px; }
        """)
        self.centralwidget.setStyleSheet("QWidget { border-radius: 18px; }")

    def clear_canvas(self):
        self.canvas.clear()
        self.resultLabel.setText("")

    def predict_canvas(self):
        image = self.canvas.get_image()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(image.height(), image.width(), 4)
        img = Image.fromarray(arr[..., :3]).convert('L').resize((28, 28))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        result = self.predictor.predict(img_b64)
        if isinstance(result, tuple):
            pred, conf = result
        else:
            pred, conf = result, None
        if pred is None or pred == "None":
            self.resultLabel.setText("<span style='font-size:32px; color:#d32f2f;'><b>Prediction failed</b></span>")
        else:
            conf_str = f"<br><span style='font-size:20px; color:#888;'>Confidence: {conf:.2%}</span>" if conf is not None else ""
            self.resultLabel.setText(f"<span style='font-size:40px; color:#222;'>Prediction: <b>{pred}</b></span>{conf_str}")

    def upload_and_predict(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            img = Image.open(file_path).convert('L').resize((28, 28))
            # Display uploaded image on canvas
            img_rgb = img.convert('RGB').resize((300, 300))
            qimg = QPixmap.fromImage(ImageQt(img_rgb))
            self.canvas.pixmap = qimg
            self.canvas.setPixmap(qimg)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_bytes = buf.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            result = self.predictor.predict(img_b64)
            if isinstance(result, tuple):
                pred, conf = result
            else:
                pred, conf = result, None
            if pred is None or pred == "None":
                self.resultLabel.setText("<span style='font-size:32px; color:#d32f2f;'><b>Prediction failed</b></span>")
            else:
                conf_str = f"<br><span style='font-size:20px; color:#888;'>Confidence: {conf:.2%}</span>" if conf is not None else ""
                self.resultLabel.setText(f"<span style='font-size:40px; color:#222;'>Prediction: <b>{pred}</b></span>{conf_str}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())