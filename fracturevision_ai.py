import sys
import os
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QSizePolicy,
    QSlider, QGroupBox, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QCoreApplication

try:
    from ultralytics import YOLO
except ImportError:
    app = QApplication(sys.argv)
    QMessageBox.critical(None, "Required Library Missing",
                         "The Ultralytics YOLOv8 library is not installed.\nTo install: pip install ultralytics")
    sys.exit(1)


class PhalanxDetector:
    """Bone Fracture Detection System"""

    def __init__(self, model_path=None, status_callback=None):
        self.status_callback = status_callback
        self.class_names = ['fracture']
        self.kirik_class_id = 0
        self.class_colors = {
            'fracture': (0, 0, 255),
        }
        self.model_path = model_path or self.find_latest_model()
        self.model = None
        self.load_model()

    def _update_status(self, message):
        if self.status_callback:
            self.status_callback(message)

    def find_latest_model(self):
        # Search in standard relative locations
        search_paths = [
            Path("weights/best.pt"),
            Path("models/best.pt"),
            Path("runs/detect/train/weights/best.pt"),
            Path("best.pt"),
        ]

        for model_path in search_paths:
            if model_path.exists():
                self._update_status(f"Model found: {model_path}")
                return str(model_path)

        # Fallback: find any .pt file in the current directory
        for pt_file in Path('.').glob('*.pt'):
            self._update_status(f"Model found: {pt_file}")
            return str(pt_file)

        self._update_status("Model file not found!")
        return None

    def load_model(self):
        try:
            if self.model_path is None:
                raise FileNotFoundError("Model path not found")
            self._update_status(f"Loading model: {os.path.basename(self.model_path)}")
            self.model = YOLO(self.model_path)
            self.model.model.names = self.class_names
            self._update_status("Model loaded successfully")
        except Exception as e:
            self._update_status(f"Failed to load model: {e}")
            self.model = None

    def detect(self, image_path_or_frame, confidence_threshold=0.25):
        if self.model is None:
            return {"error": "Model not loaded", "success": False}

        frame = cv2.imread(image_path_or_frame) if isinstance(image_path_or_frame, str) else image_path_or_frame.copy()

        if frame is None:
            return {"error": f"Could not read image: {image_path_or_frame}", "success": False}

        self._update_status(f"Running detection (Threshold: {confidence_threshold:.2f})")
        QCoreApplication.processEvents()

        results = self.model.predict(frame, conf=confidence_threshold, iou=0.45, verbose=False)

        if not results or len(results) == 0:
            return {"error": "No results produced", "success": False}

        result = results[0]
        self._update_status("Performing analysis")
        detections = self._boxes_to_numpy(result)
        analysis = self.analyze_detections(detections)

        return {
            "success": True,
            "original_frame": frame,
            "analysis": analysis,
            "raw_detections": detections
        }

    def _boxes_to_numpy(self, result):
        boxes = []
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                boxes.append({
                    'xmin': xyxy[0],
                    'ymin': xyxy[1],
                    'xmax': xyxy[2],
                    'ymax': xyxy[3],
                    'confidence': conf,
                    'class': cls
                })
        return boxes

    def analyze_detections(self, detections):
        analysis = {
            "total_detections": len(detections),
            "findings": [],
            "primary_result": "No issue detected",
            "priority_level": "NORMAL",
            "medical_alert": False
        }

        if not detections:
            return analysis

        kirik_detections = [d for d in detections if d['class'] == self.kirik_class_id]

        if kirik_detections:
            analysis["primary_result"] = "FRACTURE DETECTED"
            analysis["priority_level"] = "CRITICAL"
            analysis["medical_alert"] = True
            for d in kirik_detections:
                analysis["findings"].append({
                    "type": "fracture",
                    "confidence": d['confidence']
                })

        return analysis

    def visualize_detections(self, frame, detections):
        annotated_frame = frame.copy()

        if not detections:
            return annotated_frame

        for detection in detections:
            x1 = int(detection['xmin'])
            y1 = int(detection['ymin'])
            x2 = int(detection['xmax'])
            y2 = int(detection['ymax'])
            confidence = detection['confidence']
            class_id = int(detection['class'])

            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
                color = self.class_colors.get(class_name, (0, 255, 0))
                is_critical = class_name == 'fracture'
                thickness = 3 if is_critical else 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                label = f"{class_name}: {confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame


class BoneFractureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bone Fracture and Dislocation Detection System")
        self.setGeometry(100, 100, 1280, 800)

        self.image_path = None
        self.original_frame = None
        self.last_pixmap = None

        self.setAcceptDrops(True)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QHBoxLayout(main_widget)

        self.init_ui()
        self.detector = PhalanxDetector(status_callback=self.update_status_bar)

    def init_ui(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(400)

        control_group = QGroupBox("Control Panel")
        control_group.setFont(QFont("Arial", 11, QFont.Bold))
        group_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.select_image)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setFont(QFont("Arial", 10, QFont.Bold))
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setEnabled(False)

        btn_layout.addWidget(self.select_button)
        btn_layout.addWidget(self.analyze_button)
        group_layout.addLayout(btn_layout)

        conf_layout = QHBoxLayout()
        self.conf_label = QLabel("Confidence Threshold: 25%")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(99)
        self.conf_slider.setValue(25)
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.confidence_slider_changed)
        conf_layout.addWidget(self.conf_label)
        conf_layout.addWidget(self.conf_slider)
        group_layout.addLayout(conf_layout)

        control_group.setLayout(group_layout)
        left_layout.addWidget(control_group)

        report_group = QGroupBox("Analysis Report")
        report_group.setFont(QFont("Arial", 11, QFont.Bold))
        report_layout = QVBoxLayout()
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("Courier New", 10))
        report_layout.addWidget(self.report_text)
        report_group.setLayout(report_layout)
        left_layout.addWidget(report_group)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.image_label = QLabel("Select an image file for analysis\nor drag and drop it here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont("Arial", 16))
        self.image_label.setStyleSheet(
            "color: grey; border: 2px dashed #aaa; border-radius: 10px; padding: 10px;"
        )
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        right_layout.addWidget(self.image_label)

        self.main_layout.addWidget(left_panel)
        self.main_layout.addWidget(right_panel, 1)

        self.statusBar().setFont(QFont("Arial", 10))
        self.update_status_bar("Ready")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.load_image_from_path(path)
            else:
                self.update_status_bar("Invalid file type")

    def load_image_from_path(self, path):
        self.image_path = path
        self.original_frame = cv2.imread(self.image_path)
        if self.original_frame is None:
            self.update_status_bar(f"Could not load image: {path}")
            return
        self.display_image(self.original_frame)
        self.analyze_button.setEnabled(True)
        self.report_text.clear()
        self.update_status_bar(f"Selected image: {os.path.basename(self.image_path)}")

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.load_image_from_path(path)

    def confidence_slider_changed(self, value):
        self.conf_label.setText(f"Confidence Threshold: {value}%")

    def analyze_image(self):
        if not self.image_path or self.original_frame is None:
            return

        self.analyze_button.setEnabled(False)
        QApplication.processEvents()

        try:
            confidence = self.conf_slider.value() / 100.0
            result = self.detector.detect(self.original_frame, confidence_threshold=confidence)

            if not result["success"]:
                self.update_status_bar(f"Error: {result['error']}")
                return

            analysis_report = self.format_report(result["analysis"])
            self.report_text.setHtml(analysis_report)

            annotated_frame = self.detector.visualize_detections(
                self.original_frame, result["raw_detections"]
            )
            self.display_image(annotated_frame)
            self.update_status_bar("Analysis complete")

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"An error occurred during analysis:\n\n{str(e)}"
            )
            self.update_status_bar("Analysis failed")
        finally:
            self.analyze_button.setEnabled(True)

    def display_image(self, cv_img):
        pixmap = self.convert_cv_qt(cv_img)
        self.last_pixmap = pixmap
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

    def format_report(self, analysis):
        color = "red" if analysis["medical_alert"] else "green"
        report = f"<h2 style='color:{color}'>Result: {analysis['primary_result']}</h2>"
        report += f"<b>Priority:</b> {analysis['priority_level']}<br>"
        report += f"<b>Detection Count:</b> {analysis['total_detections']}<br>"

        if not analysis["findings"]:
            report += "<b>Details:</b> No findings"
            return report

        report += "<br><b>Findings:</b><br><ul>"
        for finding in analysis["findings"]:
            finding_type = finding['type']
            confidence_pct = finding['confidence'] * 100
            if finding_type == 'fracture':
                report += f"<li><b>FRACTURE: {confidence_pct:.1f}%</b></li>"
        report += "</ul>"
        return report

    def update_status_bar(self, message):
        self.statusBar().showMessage(message, 5000)

    def resizeEvent(self, event):
        if self.last_pixmap:
            self.image_label.setPixmap(
                self.last_pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = BoneFractureApp()
    main_window.show()
    sys.exit(app.exec_())
