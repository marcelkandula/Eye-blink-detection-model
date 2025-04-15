from model import EyeBlinkCNN
import torch
import torchvision.transforms as transforms
import cv2
import mediapipe as mp
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EyeBlinkCNN().to(device)
model.load_state_dict(torch.load('eye_blink_cnn.pth', map_location=device))
model.eval()

# transform crooped eye frame into, ready to inference image 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((24, 24)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def predict_eye(eye_img):
    with torch.no_grad():
        input_tensor = transform(eye_img).unsqueeze(0).to(device)
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        return 'open' if pred.item() == 1 else 'closed'


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
LEFT_EYE_IDXS = [33, 133]
RIGHT_EYE_IDXS = [362, 263]

blink_counter = 0
both_were_closed = False

closed_start_time = None
total_closed_time = 0.0 

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    left_state, right_state = "open", "open"

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            for side, (p1_idx, p2_idx) in zip(["left", "right"], [LEFT_EYE_IDXS, RIGHT_EYE_IDXS]):
                p1 = face_landmarks.landmark[p1_idx]
                p2 = face_landmarks.landmark[p2_idx]

                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)

                margin = 15
                x_min = max(0, min(x1, x2) - margin)
                x_max = min(w, max(x1, x2) + margin)
                y_min = max(0, min(y1, y2) - margin)
                y_max = min(h, max(y1, y2) + margin)

                eye_img = frame[y_min:y_max, x_min:x_max]
                if eye_img.size == 0:
                    continue

                state = predict_eye(eye_img)
                if side == "left":
                    left_state = state
                else:
                    right_state = state

                color = (0, 255, 0) if state == "open" else (0, 0, 255)
                label = f"{side.capitalize()} eye: {state}"
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # detect blink only if eyes went from open -> closed -> open
        if left_state == "closed" and right_state == "closed":
            if not both_were_closed:
                closed_start_time = time.perf_counter()  # start count
                both_were_closed = True
        elif left_state == "open" and right_state == "open": 
            if both_were_closed and closed_start_time is not None: # if eyes opened after blink, stop time and count blink
                closed_duration = time.perf_counter() - closed_start_time
                total_closed_time += closed_duration
                blink_counter += 1
                both_were_closed = False
                closed_start_time = None

    cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,69,0), 2)
    cv2.putText(frame, f"Eyes closed time: {total_closed_time:.3f} s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,	(255,69,0), 2)
    cv2.putText(frame, f"Average time for each blink: {total_closed_time / blink_counter if blink_counter is not 0 else 0:.3f} s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,69,0), 2)

    cv2.imshow("Eye Blink Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()