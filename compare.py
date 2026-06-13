import cv2
import numpy as np
from mtcnn import MTCNN
from retinaface import RetinaFace
import time
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# Robustness Analysis
# Tests both models on challenging conditions using your webcam
# ─────────────────────────────────────────

mtcnn_detector = MTCNN()


def detect_mtcnn(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    start = time.time()
    results = mtcnn_detector.detect_faces(rgb)
    elapsed = time.time() - start
    return len(results), elapsed, results


def detect_retinaface(frame):
    start = time.time()
    results = RetinaFace.detect_faces(frame)
    elapsed = time.time() - start
    count = len(results) if isinstance(results, dict) else 0
    return count, elapsed, results


def draw_detections_mtcnn(frame, results, color=(0, 255, 0)):
    for r in results:
        x, y, w, h = r['box']
        x, y = max(0, x), max(0, y)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        for pt in r['keypoints'].values():
            cv2.circle(frame, pt, 3, (0, 0, 255), -1)
    return frame


def draw_detections_retina(frame, results, color=(255, 128, 0)):
    if isinstance(results, dict):
        for face in results.values():
            x1, y1, x2, y2 = face['facial_area']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            for pt in face['landmarks'].values():
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)
    return frame


def run_side_by_side():
    """Run both detectors side by side from webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("\nSide-by-side comparison running...")
    print("Press 'q' to quit | Press 's' to save screenshot")

    mtcnn_fps_list, retina_fps_list = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        left = frame.copy()
        right = frame.copy()

        # MTCNN
        m_count, m_time, m_results = detect_mtcnn(frame)
        left = draw_detections_mtcnn(left, m_results)
        m_fps = 1.0 / m_time if m_time > 0 else 0
        mtcnn_fps_list.append(m_fps)

        # RetinaFace
        r_count, r_time, r_results = detect_retinaface(frame)
        right = draw_detections_retina(right, r_results)
        r_fps = 1.0 / r_time if r_time > 0 else 0
        retina_fps_list.append(r_fps)

        # Labels
        avg_m_fps = np.mean(mtcnn_fps_list[-20:])
        avg_r_fps = np.mean(retina_fps_list[-20:])

        cv2.putText(left, f"MTCNN | FPS: {avg_m_fps:.1f} | Faces: {m_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(right, f"RetinaFace | FPS: {avg_r_fps:.1f} | Faces: {r_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 128, 0), 2)

        # Combine
        combined = np.hstack([left, right])
        cv2.imshow("MTCNN (left) vs RetinaFace (right) — Press Q to quit", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('comparison_screenshot.png', combined)
            print("Screenshot saved as comparison_screenshot.png")

    cap.release()
    cv2.destroyAllWindows()

    # Final summary
    print("\n" + "="*50)
    print("     REAL-TIME PERFORMANCE SUMMARY")
    print("="*50)
    print(f"{'Model':<15} {'Avg FPS':>10} {'Min FPS':>10} {'Max FPS':>10}")
    print("-"*50)
    if mtcnn_fps_list:
        print(f"{'MTCNN':<15} {np.mean(mtcnn_fps_list):>10.2f} {np.min(mtcnn_fps_list):>10.2f} {np.max(mtcnn_fps_list):>10.2f}")
    if retina_fps_list:
        print(f"{'RetinaFace':<15} {np.mean(retina_fps_list):>10.2f} {np.min(retina_fps_list):>10.2f} {np.max(retina_fps_list):>10.2f}")
    print("="*50)

    # Plot FPS over time
    plt.figure(figsize=(10, 4))
    plt.plot(mtcnn_fps_list, label='MTCNN', color='steelblue', alpha=0.7)
    plt.plot(retina_fps_list, label='RetinaFace', color='coral', alpha=0.7)
    plt.title('FPS Over Time: MTCNN vs RetinaFace')
    plt.xlabel('Frame')
    plt.ylabel('FPS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fps_comparison.png', dpi=150)
    plt.show()
    print("FPS chart saved as fps_comparison.png")


if __name__ == "__main__":
    run_side_by_side()