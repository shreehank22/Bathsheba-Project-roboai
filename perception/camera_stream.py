# camera_stream.py
import threading
import time
import cv2
import queue

_frame_queue = queue.Queue(maxsize=2)

def _display_loop(stop_event):
    while not stop_event.is_set():
        try:
            fixed_bgr, wrist_bgr = _frame_queue.get(timeout=0.1)
            cv2.imshow('Fixed Camera', fixed_bgr)
            cv2.imshow('Wrist Camera', wrist_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        except queue.Empty:
            continue

def start_stream(stop_event):
    t = threading.Thread(target=_display_loop, args=(stop_event,), daemon=True)
    t.start()

def push_frames(fixed_img, wrist_img):
    import cv2
    fixed_bgr = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
    if not _frame_queue.full():
        _frame_queue.put((fixed_bgr, wrist_bgr))

def stop_stream(stop_event):
    stop_event.set()
    cv2.destroyAllWindows()