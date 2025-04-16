# 👁️ Eye Blink Detection with CNN & MediaPipe

This project demonstrates a real-time **eye blink detection system** using a custom-trained **Convolutional Neural Network (CNN)** combined with **MediaPipe** for accurate eye region localization. The application tracks blinks, detects whether eyes are open or closed, and measures the total time eyes remain closed.

## 🎬 Demo

https://user-images.githubusercontent.com/your-username/demo.mp4

<details>
<summary>If the video does not play inline, click the link below 👇</summary>

[🎥 Click to watch demo.mp4](demo.mp4)

</details>

---

## 🧠 Model Overview

The model is a lightweight CNN trained to classify grayscale images (24x24 pixels) of either left or right eyes into two categories: `open` or `closed`.

### Dataset

The dataset is organized as follows:

dataset/ ├── closedLeftEyes/ ├── closedRightEyes/ ├── openLeftEyes/ └── openRightEyes/

yaml
Kopiuj
Edytuj

Each folder contains grayscale `.jpg` images of individual eye states.

### Training

- Model defined in [`model.py`](model.py)
- Training done in [`train.ipynb`](train.ipynb)
- Trained model saved as [`eye_blink_cnn.pth`](eye_blink_cnn.pth)

---

## 📂 Project Structure

. ├── train.ipynb # Jupyter notebook for training the CNN ├── model.py # CNN model architecture definition ├── inference.py # Real-time eye blink detection using webcam ├── eye_blink_cnn.pth # Trained CNN model weights ├── demo.mp4 # Video demo of real-time detection └── README.md # Project description and usage

---

## 🖥️ Real-Time Inference

The live detection is performed via webcam using [`inference.py`](inference.py), which:

- Utilizes **MediaPipe Face Mesh** for eye landmark detection.
- Extracts and crops the eye region based on facial landmarks.
- Preprocesses the eye image and feeds it into the CNN model.
- Detects blinks only when **both eyes are simultaneously closed**.
- Displays:
  - Eye states (open or closed) for both eyes.
  - Blink count.
  - Total duration (in seconds) eyes remained closed.

### 👇 Example Output

- **Green Text**: Eye open
- **Red Text**: Eye closed
- **Blink Count** and **Total Eye Closed Time** shown on the frame

---

## 📦 Requirements

Make sure to install all required packages:

```bash
pip install -r requirements.txt

