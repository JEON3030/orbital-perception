#!/usr/bin/env python3
"""실시간 CCTV(RTSP/HLS/HTTP) 객체탐지 — 온보드 저지연 라이브 데모.

파일 처리(perception.run_video)와 실시간 스트림은 다르다. 스트림은 프레임이 계속
들어오는데 추론이 조금이라도 느리면 **버퍼가 쌓여 화면이 몇 분씩 밀린다**. 그래서:
  1) 최신프레임 스레드 캡처: 백그라운드 스레드가 계속 read() 하며 **가장 최신 프레임만**
     들고 있고, 추론 루프는 항상 그 최신 프레임만 가져간다(오래된 프레임은 버림) → 저지연.
  2) 자동 재접속: CCTV 는 끊긴다. read 실패가 쌓이면 release 후 재연결.
  3) RTSP 는 TCP 전송이 안정적(OpenCV FFMPEG 옵션, VideoCapture 열기 전에 설정).

출력은 MJPEG(multipart/x-mixed-replace) 로 HTTP 제공 → 브라우저 <img> 로 저지연 재생.
탐지는 이 프로젝트의 TRT 엔진(젯슨 GPU)·torchvision NMS 스텁을 그대로 재사용한다.

사용:
  ./live_cctv.sh --source "rtsp://user:pass@ip:554/stream1"
  ./live_cctv.sh --source "http://.../cctv.m3u8" --track
  → 브라우저에서 http://<보드IP>:7861/  (Tailscale/LAN)
"""
import argparse
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

# RTSP: TCP 전송이 UDP 보다 끊김·깨짐이 적다. VideoCapture 열기 전에 잡아야 먹는다.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# perception import = torchvision NMS 스텁 + ultralytics 로드 (webviz 와 동일 재사용)
import perception
from types import SimpleNamespace

YOLO = perception.YOLO
infer_once = perception.infer_once
infer_track = perception.infer_track


# ── 최신프레임 스레드 캡처 ──────────────────────────────────────────────────
class StreamGrabber:
    def __init__(self, url, reconnect=True):
        self.url = url
        self.reconnect = reconnect
        self.cap = None
        self._frame = None
        self._ts = 0.0
        self._lock = threading.Lock()
        self.running = False
        self.connected = False
        self.grabbed = 0          # 캡처한 총 프레임(스트림 fps 추정용)

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 내부 버퍼 최소화(밀림 방지)
        except Exception:
            pass
        return cap

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        fails = 0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open()
                if not self.cap.isOpened():
                    self.connected = False
                    time.sleep(2.0)
                    continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                fails += 1
                self.connected = False
                if fails > 30 and self.reconnect:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    fails = 0
                time.sleep(0.05)
                continue
            fails = 0
            self.connected = True
            with self._lock:
                self._frame = frame
                self._ts = time.time()
                self.grabbed += 1

    def latest(self):
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame, self._ts     # 읽는 쪽에서 필요시 copy

    def stop(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


# ── 라이브 탐지기: 최신프레임 → TRT 탐지 → 주석 프레임(JPEG) ────────────────
class LiveDetector:
    def __init__(self, source, task="detect", conf=0.25, imgsz=640,
                 classes=None, track=False, jpeg_quality=80):
        import device
        fmt = "engine" if device.is_cuda() else "pt"
        base = "yolo11n" if task == "detect" else "yolo11n-seg"
        weight = f"{base}.engine" if fmt == "engine" else f"{base}.pt"
        root = os.path.dirname(os.path.abspath(__file__))
        wp = os.path.join(root, weight)
        if not os.path.exists(wp):
            raise FileNotFoundError(f"가중치 없음: {weight} (엔진은 export_trt.sh 로 생성)")
        self.model = YOLO(wp)
        self.fmt = fmt
        self.task = task
        self.track = track
        self.jpeg_quality = jpeg_quality
        self.args = SimpleNamespace(task=task, conf=float(conf), imgsz=int(imgsz),
                                    device=(0 if device.is_cuda() else "cpu"),
                                    classes=classes, track=track,
                                    tracker="bytetrack.yaml")
        self.grabber = StreamGrabber(source)
        self._jpeg = None
        self._jlock = threading.Lock()
        self.running = False
        self.infer_fps = 0.0
        self.last_latency_ms = 0.0
        self.det_summary = ""

    def start(self):
        self.grabber.start()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _encode(self, frame):
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            with self._jlock:
                self._jpeg = buf.tobytes()

    def _placeholder(self, text):
        import numpy as np
        img = np.zeros((360, 640, 3), dtype="uint8")
        cv2.putText(img, text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 200, 255), 2)
        self._encode(img)

    def _loop(self):
        from collections import Counter
        first_track = True
        last_ts = 0.0
        ema = None
        self._placeholder("connecting to stream...")
        while self.running:
            frame, ts = self.grabber.latest()
            if frame is None or ts == last_ts:
                if not self.grabber.connected:
                    self._placeholder("waiting/reconnecting to CCTV...")
                time.sleep(0.005)
                continue
            last_ts = ts
            frame = frame.copy()
            t0 = time.time()
            if self.track:
                res = infer_track(self.model, frame, self.args, persist=not first_track)
                first_track = False
            else:
                res = infer_once(self.model, frame, self.args)
            ann = res.plot()
            dt = time.time() - t0
            ema = dt if ema is None else 0.9 * ema + 0.1 * dt
            self.infer_fps = 1.0 / ema if ema else 0.0
            self.last_latency_ms = (time.time() - ts) * 1000.0   # 캡처→표시 지연
            if res.boxes is not None and len(res.boxes):
                c = Counter(int(i) for i in res.boxes.cls.tolist())
                self.det_summary = ", ".join(f"{self.model.names[k]}×{v}"
                                             for k, v in c.most_common(6))
            else:
                self.det_summary = "(객체 없음)"
            hud = (f"{self.task}|{self.fmt}  infer {self.infer_fps:4.1f}fps  "
                   f"lat {self.last_latency_ms:4.0f}ms  {self.det_summary}")
            cv2.putText(ann, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(ann, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            self._encode(ann)

    def jpeg(self):
        with self._jlock:
            return self._jpeg

    def stop(self):
        self.running = False
        self.grabber.stop()


# ── MJPEG HTTP 서버 ────────────────────────────────────────────────────────
_INDEX = b"""<!doctype html><meta charset=utf-8>
<title>Live CCTV \xeb\x8f\x84\xea\xb5\xac</title>
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center">
<h3 style="padding:8px">\xf0\x9f\x94\xb4 \xec\x8b\xa4\xec\x8b\x9c\xea\xb0\x84 CCTV \xea\xb0\x9d\xec\xb2\xb4\xed\x83\x90\xec\xa7\x80</h3>
<img src="/mjpeg" style="max-width:100%;height:auto"/>
</body>"""


def make_handler(detector):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(_INDEX)
                return
            if self.path.startswith("/mjpeg"):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpg = detector.jpeg()
                        if jpg:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.04)     # ~25fps 상한(표시)
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self.send_response(404)
                self.end_headers()
    return H


def main():
    ap = argparse.ArgumentParser(description="실시간 CCTV 객체탐지 (MJPEG 라이브)")
    ap.add_argument("--source", required=True, help="RTSP/HLS/HTTP 스트림 URL")
    ap.add_argument("--task", choices=["detect", "segment"], default="detect")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes", nargs="*", default=None,
                    help="필터 클래스 id(예: 0 2 = person car)")
    ap.add_argument("--track", action="store_true", help="ByteTrack 시간축 추적")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()

    import device
    device.set_mode("auto")
    print(f"장치: {device.summary()}")
    classes = [int(c) for c in args.classes] if args.classes else None
    det = LiveDetector(args.source, task=args.task, conf=args.conf, imgsz=args.imgsz,
                       classes=classes, track=args.track).start()
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(det))
    print(f"라이브 뷰:  http://<보드IP>:{args.port}/   (Ctrl-C 로 중지)")
    print(f"소스: {args.source}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        det.stop()


if __name__ == "__main__":
    main()
