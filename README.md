# ⛑️ Hard Hat Detection with YOLOv8-OBB

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-OBB-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Real-time safety helmet compliance detection for construction sites using YOLOv8 Oriented Bounding Boxes**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Model Performance](#-model-performance) • [Team](#-team)

</div>

---

## 📋 Overview

Construction sites face a critical safety challenge: workers without helmets are at risk of serious head injuries. Manual monitoring is slow, unreliable, and labor-intensive. This project provides an **AI-powered solution** using YOLOv8 with Oriented Bounding Boxes (OBB) to automatically detect:

- 🧠 **Heads** — Unprotected/bare heads (safety violations)
- ⛑️ **Helmets** — Workers wearing safety hard hats (compliance)

The system can process images in real-time and is deployable via a user-friendly **Streamlit web application**.

---

## ✨ Features

- **Real-time Detection**: Fast inference (~25ms per image on GPU)
- **Oriented Bounding Boxes**: Better handling of rotated objects common in construction scenes
- **High Accuracy**: 93.1% mAP50 on the validation set
- **Web Interface**: Easy-to-use Streamlit app for image upload and analysis
- **Configurable Thresholds**: Adjust confidence and IoU thresholds on-the-fly
- **Detection Summary**: Detailed breakdown of detected objects with confidence scores
- **Download Results**: Export annotated images with detection overlays

---

## 🎯 Demo

Upload an image to the Streamlit app and get instant detection results:

| Original Image | Detection Result |
|:--------------:|:----------------:|
| Input image of workers | Annotated output with bounding boxes |

The app displays:
- Side-by-side comparison of original and annotated images
- Detection summary with counts per class
- Per-detection confidence scores with visual progress bars
- Inference time metrics

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster inference)

### Clone the Repository

```bash
git clone https://github.com/Nawaf-Alorabi/Tw_Hard_hat_Detection.git
cd Tw_Hard_hat_Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install ultralytics streamlit opencv-python pillow numpy
```

### Download the Model

1. Run the training notebook in Google Colab to generate the model, or
2. Download the pre-trained `hat_detection_tuned_best.pt` from the releases
3. Place the model file in the project root directory

---

## 💻 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the App

1. **Configure Settings** (Sidebar):
   - Set the model path (default: `hat_detection_tuned_best.pt`)
   - Adjust confidence threshold (default: 0.25)
   - Adjust IoU threshold for NMS (default: 0.45)
   - Toggle label and confidence display

2. **Upload an Image**:
   - Supported formats: JPG, JPEG, PNG, BMP, WEBP
   - Drag & drop or click to browse

3. **View Results**:
   - See original vs. annotated image comparison
   - Review detection summary with class breakdown
   - Download the annotated image

### Training Your Own Model

Open the Jupyter notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nawaf-Alorabi/Tw_Hard_hat_Detection/blob/main/Hat_Detection_YOLOv8.ipynb)

The notebook includes:
- Dataset download from Roboflow
- Baseline model training (YOLOv8n-OBB)
- Hyperparameter tuning (YOLOv8s-OBB)
- Model evaluation and visualization
- Model export

---

## 📊 Model Performance

### Tuned Model Metrics

| Metric | Overall | Head | Helmet |
|--------|---------|------|--------|
| **Precision** | 0.942 | 0.946 | 0.939 |
| **Recall** | 0.876 | 0.910 | 0.841 |
| **mAP50** | 0.931 | 0.963 | 0.899 |
| **mAP50-95** | 0.758 | — | — |

### Training Configuration

**Baseline Model:**
- Architecture: YOLOv8n-OBB (nano)
- Epochs: 20
- Image Size: 640×640

**Tuned Model:**
- Architecture: YOLOv8s-OBB (small)
- Epochs: 25
- Batch Size: 8
- Optimizer: AdamW (lr=0.001)
- Image Size: 640×640
- Augmentations:
  - Horizontal flip (p=0.5)
  - HSV adjustments (h=0.015, s=0.7, v=0.4)
  - Rotation (±10°)
  - Translation (10%)
  - Scale (20%)

### Dataset

- **Source**: Roboflow (Hat Detection Dataset)
- **Classes**: 2 (head, helmet)
- **Training Images**: 298
- **Validation Images**: 29
- **Annotations**: 666 heads, 384 helmets

---

## 📁 Project Structure

```
Tw_Hard_hat_Detection/
├── app.py                           # Streamlit web application
├── Hat_Detection_YOLOv8.ipynb       # Training notebook
├── hat_detection_tuned_best.pt      # Trained model weights
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── assets/                          # Demo images and screenshots
```

---

## 🔧 Requirements

```txt
ultralytics>=8.0.0
streamlit>=1.0.0
opencv-python>=4.6.0
Pillow>=7.1.2
numpy>=1.18.5
```

---

## 📈 Business Applications

| Segment | Use Case |
|---------|----------|
| **Construction** | Automated helmet compliance monitoring |
| **Manufacturing** | Factory floor safety enforcement |
| **Oil & Gas** | Remote site safety inspection |
| **Government** | Regulatory compliance auditing |

### Revenue Model

- SaaS subscription per monitored site
- Pay-per-report API usage
- Enterprise licensing for on-premise deployment

---

## 🔮 Future Improvements

- [ ] Video stream support (RTSP, webcam)
- [ ] Multi-class PPE detection (vests, goggles, gloves)
- [ ] Alert system integration (email, SMS, webhook)
- [ ] Edge deployment (Jetson Nano, Raspberry Pi)
- [ ] Dashboard with historical analytics
- [ ] Multi-camera support

---

## 👥 Team

**Tuwaiq Academy — April 2025**

| Name | Role |
|------|------|
| Nawaf | Project Lead |
| Faisal | Developer |
| Saad | Developer |
| Muadh | Developer |
| Yasser | Developer |
| Abdulrahman | Developer |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://docs.ultralytics.com) for YOLOv8
- [Roboflow](https://roboflow.com) for dataset hosting and annotation tools
- [Streamlit](https://streamlit.io) for the web framework
- [Tuwaiq Academy](https://tuwaiq.edu.sa) for project support

---

<div align="center">

**Built with ❤️ for workplace safety**

⭐ Star this repo if you find it useful!

</div>
