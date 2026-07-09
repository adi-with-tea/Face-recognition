import cv2
from retinaface import RetinaFace
import time
import numpy as np

print("RetinaFace Face Detector running... Press 'q' to quit.")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

fps_list = []
inference_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Measure inference time
    start = time.time()
    results = RetinaFace.detect_faces(frame)
    end = time.time()

    inference_time = end - start
    inference_times.append(inference_time)
    fps = 1.0 / inference_time if inference_time > 0 else 0
    fps_list.append(fps)

    # Draw results
    if isinstance(results, dict):
        for face_id, face_data in results.items():
            # Bounding box
            x1, y1, x2, y2 = face_data['facial_area']
            confidence = face_data['score']

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw confidence score
            label = f"Face: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw landmarks
            landmarks = face_data['landmarks']
            for key, point in landmarks.items():
                px, py = int(point[0]), int(point[1])
                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

    # Display FPS
    avg_fps = np.mean(fps_list[-30:]) if fps_list else 0
    face_count = len(results) if isinstance(results, dict) else 0
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Faces: {face_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, "Model: RetinaFace", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("RetinaFace Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print performance summary
print("\n===== RetinaFace Performance Summary =====")
print(f"Average FPS: {np.mean(fps_list):.2f}")
print(f"Average Inference Time: {np.mean(inference_times)*1000:.2f} ms")
print(f"Min Inference Time: {np.min(inference_times)*1000:.2f} ms")
print(f"Max Inference Time: {np.max(inference_times)*1000:.2f} ms")