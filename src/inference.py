import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from model import EyeBlinkCNN  

import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Blink detection - from camera or video file"
    )
    p.add_argument(
        "--source",
        choices=["camera", "video"],
        default="camera",
        help="Video source",
    )
    p.add_argument(
        "--path",
        type=str,
        default="video.mp4",
        help="Path to file (only if --source video)",
    )
    p.add_argument(
        "--max-faces",
        type=int,
        default=5,
        help="maximum number of faces to detect (default: 5)",
    )
    return p.parse_args()


# Model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EyeBlinkCNN().to(device)
model.load_state_dict(torch.load("weights/eye_blink_detection.pth", map_location=device))
model.eval()

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((24, 24)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


@torch.no_grad()
def predict_eye(eye_img) -> str:
    if eye_img.size == 0:
        return "closed"  
    tensor = transform(eye_img).unsqueeze(0).to(device)
    out = model(tensor)
    _, pred = torch.max(out, 1)
    return "open" if pred.item() == 1 else "closed"


# face data structure
@dataclass
class FaceData:
    blink_counter: int = 0
    total_closed_time: float = 0.0
    both_closed: bool = False
    closed_start: float | None = None
    box_color: tuple[int, int, int] = (0, 255, 0)

# main loop

def main() -> None:
    args = parse_args()
    # open video source
    if args.source == "camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("cant open camera")
        out = None
    else:
        if not args.path or not Path(args.path).is_file():
            raise FileNotFoundError("path file doesnt exist")
        cap = cv2.VideoCapture(args.path)
        # prepare video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=args.max_faces,
        refine_landmarks=True,
    )
    LEFT_EYE_IDXS = [33, 133]
    RIGHT_EYE_IDXS = [362, 263]

    faces: dict[int, FaceData] = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for idx, landmarks in enumerate(results.multi_face_landmarks):
                data = faces.setdefault(idx, FaceData())

                # eye left + right detection
                states = {}
                for side, (inner_idx, outside_idx) in zip(("left", "right"), (LEFT_EYE_IDXS, RIGHT_EYE_IDXS)):
                    inner_landmark, outside_landmark = landmarks.landmark[inner_idx], landmarks.landmark[outside_idx]
                    inner_pixel_X, inner_pixel_Y = int(inner_landmark.x * w), int(inner_landmark.y * h)
                    outside_pixel_X, outside_pixel_Y = int(outside_landmark.x * w), int(outside_landmark.y * h)

                    margin = 15
                    x_min = max(0, min(inner_pixel_X, outside_pixel_X) - margin)
                    x_max = min(w, max(inner_pixel_X, outside_pixel_X) + margin)
                    y_min = max(0, min(inner_pixel_Y, outside_pixel_Y) - margin)
                    y_max = min(h, max(inner_pixel_Y, outside_pixel_Y) + margin)

                    eye_img = frame[y_min:y_max, x_min:x_max]
                    state = predict_eye(eye_img)
                    states[side] = state

                    color = (0, 255, 0) if state == "open" else (0, 0, 255)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(frame, f"{side.capitalize()}:{state}", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color,1, cv2.LINE_AA,)

                # blink logic
                now_closed = states["left"] == states["right"] == "closed"
                now_open = states["left"] == states["right"] == "open"

                if now_closed and not data.both_closed:
                    data.both_closed = True
                    data.closed_start = time.perf_counter()
                elif now_open and data.both_closed:
                    if data.closed_start is not None:
                        dur = time.perf_counter() - data.closed_start
                        data.total_closed_time += dur
                    data.blink_counter += 1
                    data.both_closed = False
                    data.closed_start = None

                # draw face bounding box and blink count
                xs = [int(l.x * w) for l in landmarks.landmark]
                ys = [int(l.y * h) for l in landmarks.landmark]
                x_min_f, x_max_f = max(0, min(xs)), min(w, max(xs))
                y_min_f, y_max_f = max(0, min(ys)), min(h, max(ys))
                cv2.rectangle(frame, (x_min_f, y_min_f), (x_max_f, y_max_f), (255, 255, 0), 1)

                info = f"Face {idx}: blinks {data.blink_counter}"
                cv2.putText(
                    frame,
                    info,
                    (x_min_f, y_min_f - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 69, 0),
                    2,
                    cv2.LINE_AA,
                )

        # show and write frame
        cv2.imshow("Eye‑Blink Detection", frame)
        if out is not None:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF in (27, ord("q")):  # ESC / q
            break

    cap.release()
    if out is not None:
        out.release()
        print("Saved video as output.mp4")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
