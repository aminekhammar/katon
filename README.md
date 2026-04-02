# katon
Intelligent Forest Fire Detection System


[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-orange.svg)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Google Colab](https://img.shields.io/badge/Training-Google%20Colab-yellow.svg)](https://colab.research.google.com/)

---

##  Project Overview
**Katon** is an AI-powered surveillance dashboard developed for the **I-ACE Competition (Intellect Scientific Club)**. Forest fires represent a global environmental crisis, often escalating due to delayed detection. Our system provides a real-time monitoring solution using a distributed network of cameras and advanced computer vision to identify smoke and flames instantly, enabling rapid intervention before disasters escalate.

---

##  Technical Stack
* **AI Engine:** YOLOv8 (Nano & Large architectures) via `ultralytics`.
* **Web Framework:** `Streamlit` (Custom Dark-Mode Dashboard).
* **Computer Vision:** `OpenCV (cv2)` for real-time frame processing.
* **Concurrency:** Python `threading` for simultaneous handling of multiple camera feeds.
* **Infrastructure:** Trained on `Google Colab` with `Google Drive` for persistence.

---

##  External Resources
- **Model Weights (best.pt):** [Download from Google Drive](https://drive.google.com/drive/folders/1qr_Ry6aleuUPK7E87-3g91JV7Tx5ENpS?usp=sharing)
- **Trained Dataset:** [Access on Google Drive](https://drive.google.com/drive/folders/1oFuvkJTmqkime_pzTYMSk-07yCOcPo8t?usp=sharing)
- **Source:** [University of Jeddah Dataset](https://universe.roboflow.com/university-of-jeddah-fslnf/forest-fire-obkwo) via Roboflow.

---

##  Cloud Training Workflow (Google Colab)
After placing the dataset in Google Drive, switch to Google Colab for the GPU and follow the steps

```python
# 1. Install Ultralytics
!pip install ultralytics
from ultralytics import YOLO

# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#3. Model testing
!yolo task=detect mode=predict \
  model=yolov8n.pt \    #yolov8n_for_nano_and_yolov8l_for_large
  source="https://ultralytics.com/images/bus.jpg"

#4. Training Model
!yolo task=detect mode=train \
  model=yolov8n.pt \   #yolov8n_for_nano_and_yolov8l_for_large
  data=/content/drive/MyDrive/Datasets/ForestFireDetection/data.yaml \
  epochs=120 imgsz=640 batch=16 \ 
  project=/content/drive/MyDrive/Results \
  name=fire_detection
```
---


