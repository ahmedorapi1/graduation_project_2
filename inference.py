import threading
import queue
import time

import cv2
import numpy as np
import onnxruntime as ort


class OnnxInferenceWorker:
    def __init__(self, model_path, input_size=(320, 320), providers=None, callback=None):
        self.model_path = model_path
        self.input_w, self.input_h = input_size
        self.providers = providers or ["CPUExecutionProvider"]
        self.callback = callback

        self.frame_queue = queue.Queue(maxsize=2)
        self._running = False
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._init_session()

    def _init_session(self):
        print(f"[INFO] Loading ONNX model: {self.model_path}")
        self.session = ort.InferenceSession(
            self.model_path,
            providers=self.providers
        )

        input_tensor = self.session.get_inputs()[0]
        self.input_name = input_tensor.name
        self.input_shape = input_tensor.shape

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join()
        print("[INFO] Inference thread stopped.")

    def push_frame(self, frame):
        if self.frame_queue.full():
            try:
                _ = self.frame_queue.get_nowait()
            except queue.Empty:
                pass

        self.frame_queue.put(frame)

    def _preprocess(self, frame):
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def _run(self):
        print("[INFO] Inference loop running...")
        while self._running:
            frame = self.frame_queue.get()
            if frame is None:
                break

            if frame is None or frame.size == 0:
                continue

            input_tensor = self._preprocess(frame)

            outputs = self.session.run(None, {self.input_name: input_tensor})
            if self.callback is not None:
                try:
                    self.callback(frame, outputs)
                except Exception as e:
                    print(f"[WARNING] Callback error: {e}")

        print("[INFO] Inference loop exited.")



def callback(frame, outputs):
    pass


def main():
    model_path = "YOLOv10n_gestures_int8_cpu.onnx"
    input_size = (320, 320)

    worker = OnnxInferenceWorker(
        model_path=model_path,
        input_size=input_size,
        providers=["CPUExecutionProvider"],
        callback=callback
    )

    worker.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        worker.push_frame(frame)

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
