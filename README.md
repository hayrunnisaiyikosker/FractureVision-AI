# FractureVision-AI  
### Deep Learning–Based X-Ray Fracture Detection System (YOLOv8)

---

## Overview

FractureVision-AI is an AI-powered medical imaging project designed to automatically detect bone fractures from X-ray images using object detection techniques.

The model was trained over 100 epochs on an augmented dataset exceeding 40,000 X-ray images. The dataset was expanded using preprocessing and image enhancement techniques such as brightness and contrast adjustments to improve generalization performance. The system achieved an overall detection accuracy of approximately 80–90%.

The system integrates a trained YOLOv8 model with a PyQt5 desktop application, enabling users to upload X-ray images and receive real-time fracture localization with bounding box visualization and confidence scores.

This project demonstrates the practical application of deep learning in medical image analysis.

---

## Model & Training Details

- **Architecture:** YOLOv8 (Object Detection)  
- **Framework:** PyTorch  
- **Dataset Size:** 40,000+ augmented X-ray images  
- **Training Epochs:** 100  
- **Detection Accuracy:** ~80–90%  
- **Output:** Bounding box localization with confidence scores  

The model detects fracture regions directly on X-ray images. During inference, detected areas are highlighted with bounding boxes and corresponding confidence values.

---

## System Architecture

FractureVision-AI consists of two main components:

### 1. Deep Learning Detection Model
- Processes X-ray images  
- Detects fracture regions  
- Outputs bounding box coordinates  
- Provides confidence scores  

### 2. Desktop Application (PyQt5 GUI)
- Image upload functionality  
- Real-time fracture detection  
- Visualized prediction results  
- User-friendly interface  

---

## Workflow

1. The user uploads an X-ray image via the GUI.  
2. The trained YOLOv8 model processes the image.  
3. The system detects potential fracture regions.  
4. Bounding boxes are drawn on detected areas.  
5. The processed image with predictions is displayed to the user.  

---

## Technologies Used

- Python  
- PyTorch  
- YOLOv8 (Ultralytics)  
- OpenCV  
- PyQt5  
- NumPy  

---

## Performance Highlights

- Trained on 40K+ augmented labeled X-ray images  
- 100-epoch training process  
- Strong fracture detection performance (~80–90% accuracy)  
- Optimized for fast inference  
- Practical real-world AI healthcare application  

---

## Disclaimer

This project is developed for educational and research purposes only. It is not intended for clinical diagnosis.
