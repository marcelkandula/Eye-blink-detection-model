# Eye Blink Detection Model

A lightweight and effective eye blink detection system built using PyTorch, OpenCV, and mediapipe. This project detects eye blinks in real-time from camera and video streams.

---

##  Demo
 
Click the link below to view the demo video:

[🎬 **demo.mp4**](demo.mp4)


---

##  How It Works

- Uses mediapipe's facial landmark tool to collect regions of eyes of multiple faces.
- Predicts with pretrained model whether and eye is open or closed.
- Shows outputs such as: number of blinks per face and saves video with bounding boxes to output.mp4 (only if source == video)

---


###  Installation

1. Clone this repository:

```bash
git clone https://github.com/marcelkandula/Eye-blink-detection-model.git
cd Eye-blink-detection-model
pip install -r requirements.txt
python src/inference.py
```

2. Run inference:

```bash
python src/inference.py --source video --path path_to_video.mp4
```

or when you want to use your default camera


```bash
python src/inference.py
```

Citation

Special thanks to creators of CEW dataset:

F. Song, X. Tan, X. Liu and S. Chen,
"Eyes Closeness Detection from Still Images with Multi-scale Histograms of Principal Oriented Gradients",
Pattern Recognition, 2014.

