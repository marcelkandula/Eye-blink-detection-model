# Eye Blink Detection Model

A lightweight and effective eye blink detection system built using PyTorch, OpenCV, and mediapipe. This project detects eye blinks in real-time from video streams.

---

##  Demo
 
Click the link below to view the demo video:

[🎬 **demo.mp4**](demo.mp4)


---

##  How It Works

- Uses mediapipe's facial landmark tool to collect regions of eyes.
- Predicts with pretrained model whether and eye is open or closed.
- Shows outputs such as: number of blinks, total time of closed eyes and average time per blink

---


###  Installation

1. Clone this repository:

```bash
git clone https://github.com/marcelkandula/Eye-blink-detection-model.git
cd Eye-blink-detection-model
pip install -r requirements.txt
python inference.py
```

Citation

Special thanks to creators of CEW dataset:

F. Song, X. Tan, X. Liu and S. Chen,
"Eyes Closeness Detection from Still Images with Multi-scale Histograms of Principal Oriented Gradients",
Pattern Recognition, 2014.

