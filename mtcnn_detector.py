import cv2
from mtcnn import MTCNN
import time
import numpy as np

# Initialize detector
detector = MTCNN()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("MTCNN Face Detector running... Press 'q' to quit.")

fps_list = []
inference_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB (MTCNN expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Measure inference time
    start = time.time()
    results = detector.detect_faces(rgb_frame)
    end = time.time()

    inference_time = end - start
    inference_times.append(inference_time)
    fps = 1.0 / inference_time if inference_time > 0 else 0
    fps_list.append(fps)

    # Draw bounding boxes and landmarks
    for result in results:
        x, y, w, h = result['box']
        confidence = result['confidence']
        keypoints = result['keypoints']

        # Fix negative coordinates
        x, y = max(0, x), max(0, y)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw confidence score
        label = f"Face: {confidence:.2f}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw landmarks
        for key, point in keypoints.items():
            cv2.circle(frame, point, 3, (0, 0, 255), -1)

    # Display FPS
    avg_fps = np.mean(fps_list[-30:]) if fps_list else 0
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Faces: {len(results)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, "Model: MTCNN", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("MTCNN Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print performance summary
print("\n===== MTCNN Performance Summary =====")
print(f"Average FPS: {np.mean(fps_list):.2f}")
print(f"Average Inference Time: {np.mean(inference_times)*1000:.2f} ms")
print(f"Min Inference Time: {np.min(inference_times)*1000:.2f} ms")
print(f"Max Inference Time: {np.max(inference_times)*1000:.2f} ms")