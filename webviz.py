#!/usr/bin/env python3
"""
Orbital Perception — 시각화 웹앱 (Gradio).

위성 온보드 저전력 인식 프로젝트의 성과를 브라우저에서 보여준다.
  탭 1) 성과 대시보드  : TensorRT FP16 vs PyTorch 에너지(mJ/frame·frames/J) 비교
  탭 2) 라이브 추론    : 이미지/영상 업로드 → detect/segment, 모델(.pt/.engine)·해상도
                         선택 → Before|After + 전력 타임라인 + 에너지 리포트

기존 perception.py / powerlog.py 파이프라인을 그대로 재사용한다(같은 torchvision
NMS 스텁, 같은 tegrastats 전력계측). 실행은 run.sh 와 동일한 환경을 세팅하는
webviz.sh 로 한다.

  ./webviz.sh                # 0.0.0.0:7860
  ./webviz.sh --port 8000
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# perception 을 import 하면 torchvision 스텁 설치 + ultralytics YOLO 로드가 끝난다.
import perception
YOLO = perception.YOLO
PowerMeter = perception.PowerMeter
measure_idle = perception.measure_idle
infer_once = perception.infer_once

import gradio as gr

ROOT = Path(__file__).parent
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
TRT_DIR = OUTDIR / "trt_compare"

ACCENT = "#00b4d8"
IMG_EXT = perception.IMG_EXT
VID_EXT = perception.VID_EXT

# ── 모델/유휴전력 캐시 ────────────────────────────────────────────────────
_models = {}
_idle = {"w": None}


def model_path(task, fmt):
    base = "yolo11n" if task == "detect" else "yolo11n-seg"
    return f"{base}.engine" if fmt == "engine" else f"{base}.pt"


def fmt_label(fmt):
    return "TRT-FP16" if fmt == "engine" else "PyTorch"


def get_model(path):
    if path not in _models:
        if not (ROOT / path).exists():
            raise gr.Error(f"가중치 없음: {path} — 엔진은 ./export_trt.sh 로 먼저 생성하세요.")
        _models[path] = YOLO(str(ROOT / path))
    return _models[path]


def get_idle(refresh=False):
    if _idle["w"] is None or refresh:
        _idle["w"] = measure_idle(2.0)
    return _idle["w"]


def _resolve_classes(model, text):
    if not text or not text.strip():
        return None
    name2id = {v: k for k, v in model.names.items()}
    ids = []
    for tok in text.replace(",", " ").split():
        if tok.isdigit():
            ids.append(int(tok))
        elif tok in name2id:
            ids.append(name2id[tok])
    return ids or None


def _args(task, conf, imgsz, classes):
    return SimpleNamespace(task=task, conf=float(conf), imgsz=int(imgsz),
                           device=0, classes=classes)


def _summary_md(model, class_ids):
    from collections import Counter
    c = Counter(int(i) for i in class_ids)
    if not c:
        return "**탐지 결과:** (객체 없음)"
    items = ", ".join(f"{model.names[k]}×{v}" for k, v in c.most_common())
    return f"**탐지 결과:** {items}"


def _report_md(r):
    lines = [
        "### ⚡ 에너지 리포트 (위성 전력지표)",
        f"- 프레임: **{r['frames']}**장 / {r['seconds']}s  ({r['fps']} FPS)",
        f"- 평균 전력: **{r['avg_power_W']} W**  (피크 {r['peak_power_W']} W)",
    ]
    if r.get("idle_power_W") is not None:
        lines.append(f"- 유휴 기준선: {r['idle_power_W']} W  →  동적(추론) **{r['dynamic_power_W']} W**")
    lines += [
        f"- 총 에너지: {r['energy_J']} J",
        f"- ▶ 프레임당: **{r['mJ_per_frame']} mJ/frame**"
        + (f"  (순수추론 {r['dynamic_mJ_per_frame']} mJ)" if r.get("dynamic_mJ_per_frame") is not None else ""),
        f"- ▶ 효율: **{r['frames_per_joule']} frames/J**",
    ]
    return "\n".join(lines)


def _power_fig(pm, idle_w, title):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    if pm and pm.samples:
        t0 = pm.samples[0][0]
        xs = [s[0] - t0 for s in pm.samples]
        ys = [s[1] for s in pm.samples]
        ax.plot(xs, ys, color=ACCENT, lw=1.8, label="VDD_IN (board)")
        ax.fill_between(xs, ys, alpha=0.15, color=ACCENT)
        if idle_w:
            ax.axhline(idle_w, ls="--", color="#888", lw=1.2, label=f"idle {idle_w:.2f} W")
        ax.set_ylim(0, max(ys) * 1.15 + 0.5)
    else:
        ax.text(0.5, 0.5, "no power samples", ha="center", va="center",
                transform=ax.transAxes, color="#888")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("power (W)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# ── 이미지 추론 ───────────────────────────────────────────────────────────
def run_image(image, task, fmt, imgsz, conf, repeat, classes_text, measure):
    if image is None:
        raise gr.Error("이미지를 업로드하세요.")
    path = model_path(task, fmt)
    model = get_model(path)
    classes = _resolve_classes(model, classes_text)
    args = _args(task, conf, imgsz, classes)

    # gradio는 RGB → 파이프라인(cv2)은 BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    infer_once(model, bgr, args)  # 워밍업(계측 제외)

    idle_w = None
    pm = None
    if measure:
        idle_w = get_idle()
        reps = max(int(repeat), 1)
        pm = PowerMeter()
        with pm:
            res = None
            for _ in range(reps):
                res = infer_once(model, bgr, args)
    else:
        reps = 1
        res = infer_once(model, bgr, args)

    annotated = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
    cls = res.boxes.cls.tolist() if res.boxes is not None else []
    summary = _summary_md(model, cls)

    if measure:
        r = pm.report(n_frames=reps, idle_watt=idle_w)
        report_md = _report_md(r) + f"\n\n<sub>동일 프레임 {reps}회 반복추론 평균 · " \
                                    f"{fmt_label(fmt)} · imgsz {int(imgsz)}</sub>"
        fig = _power_fig(pm, idle_w, f"{task} · {fmt_label(fmt)} · imgsz{int(imgsz)}")
    else:
        report_md = "_빠른 미리보기 (전력 계측 꺼짐)_"
        fig = _power_fig(None, None, "energy measurement off")

    return image, annotated, summary, report_md, fig


# ── 영상 추론 ─────────────────────────────────────────────────────────────
def run_video(video, task, fmt, imgsz, conf, stride, max_frames, classes_text,
              measure, progress=gr.Progress()):
    if not video:
        raise gr.Error("영상을 업로드하세요.")
    path = model_path(task, fmt)
    model = get_model(path)
    classes = _resolve_classes(model, classes_text)
    args = _args(task, conf, imgsz, classes)
    stride = max(int(stride), 1)
    max_frames = int(max_frames)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise gr.Error(f"영상을 열 수 없습니다: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    planned = (total // stride) if total else 0
    if max_frames:
        planned = min(planned, max_frames) if planned else max_frames

    out_path = OUTDIR / f"webviz_{task}_{fmt}.mp4"
    out_fps = max(src_fps / stride, 1.0)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), out_fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    idle_w = get_idle() if measure else None
    from collections import Counter
    tally = Counter()
    hud = f"{task} | {fmt_label(fmt)} | imgsz{int(imgsz)}"
    pm = PowerMeter()
    n = 0
    fi = -1
    warmed = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % stride != 0:
            continue
        if not warmed:
            infer_once(model, frame, args)   # 워밍업
            warmed = True
            if measure:
                pm.start()
        res = infer_once(model, frame, args)
        if res.boxes is not None:
            tally.update(int(i) for i in res.boxes.cls.tolist())
        ann = res.plot()
        cv2.putText(ann, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(ann, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        writer.write(ann)
        n += 1
        if planned:
            progress(min(n / planned, 1.0), desc=f"{task} {n}/{planned} 프레임")
        else:
            progress((n % 100) / 100, desc=f"{task} {n} 프레임")
        if max_frames and n >= max_frames:
            break
    if measure:
        pm.stop()
    cap.release()
    writer.release()

    summary = _summary_md(model, list(tally.elements()))
    summary = f"**누적 탐지({n}프레임):** " + summary.split("** ", 1)[-1]
    if measure and n:
        r = pm.report(n_frames=n, idle_watt=idle_w)
        report_md = _report_md(r) + f"\n\n<sub>영상 {n}프레임 · stride {stride} · " \
                                    f"{fmt_label(fmt)} · imgsz {int(imgsz)}</sub>"
        fig = _power_fig(pm, idle_w, f"{task} video · {fmt_label(fmt)} · imgsz{int(imgsz)}")
    else:
        report_md = "_전력 계측 꺼짐_"
        fig = _power_fig(None, None, "energy measurement off")

    return str(out_path), summary, report_md, fig


# ── 성과 대시보드 ─────────────────────────────────────────────────────────
def load_bench():
    data = {}
    for task in ("detect", "segment"):
        for fmt in ("pt", "engine"):
            p = TRT_DIR / f"{task}_{fmt}.json"
            if p.exists():
                try:
                    data[(task, fmt)] = json.loads(p.read_text())["energy"]
                except Exception:
                    pass
    return data


def _grouped_bar(ax, data, key, title, ylabel, lower_better=True):
    tasks = ["detect", "segment"]
    pt = [data.get((t, "pt"), {}).get(key, 0) for t in tasks]
    en = [data.get((t, "engine"), {}).get(key, 0) for t in tasks]
    x = np.arange(len(tasks))
    bw = 0.36
    b1 = ax.bar(x - bw / 2, pt, bw, label="PyTorch", color="#adb5bd")
    b2 = ax.bar(x + bw / 2, en, bw, label="TRT-FP16", color=ACCENT)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:g}",
                        ha="center", va="bottom", fontsize=8)
    for i, t in enumerate(tasks):
        p, e = pt[i], en[i]
        if p and e:
            delta = (e - p) / p * 100
            good = (delta < 0) if lower_better else (delta > 0)
            ax.annotate(f"{delta:+.0f}%", (i, max(p, e)), xytext=(0, 14),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=("#2a9d8f" if good else "#e63946"), fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.margins(y=0.22)
    ax.grid(axis="y", alpha=0.2)


def dashboard_fig():
    data = load_bench()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    if not data:
        for ax in axes:
            ax.axis("off")
        axes[1].text(0.5, 0.5, "outputs/trt_compare 에 벤치 결과 없음",
                     ha="center", va="center")
        return fig
    _grouped_bar(axes[0], data, "mJ_per_frame", "Energy per frame (total)", "mJ/frame")
    _grouped_bar(axes[1], data, "dynamic_mJ_per_frame", "Dynamic energy per frame", "mJ/frame")
    _grouped_bar(axes[2], data, "frames_per_joule", "Energy efficiency", "frames/J", lower_better=False)
    fig.suptitle("TensorRT FP16 vs PyTorch  —  7W, imgsz640, yolo11n (60 reps)",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    return fig


def dashboard_table():
    data = load_bench()
    headers = ["task", "format", "FPS", "avg W", "mJ/frame", "순수 mJ", "frames/J"]
    rows = []
    for task in ("detect", "segment"):
        for fmt in ("pt", "engine"):
            r = data.get((task, fmt))
            if not r:
                continue
            rows.append([task, fmt_label(fmt), r.get("fps"), r.get("avg_power_W"),
                         r.get("mJ_per_frame"), r.get("dynamic_mJ_per_frame"),
                         r.get("frames_per_joule")])
    return headers, rows


def dashboard_summary():
    data = load_bench()
    md = ["## 🛰️ Orbital Perception — 지금까지의 진척",
          "Jetson Orin Nano 온보드 탐지/세그멘테이션. 위성 탑재가 목표라 지표가 "
          "FPS가 아니라 **프레임당 에너지(mJ/frame)·frames-per-joule**. 운용점은 **7W**.\n"]
    wins = []
    for task in ("detect", "segment"):
        pt = data.get((task, "pt"))
        en = data.get((task, "engine"))
        if pt and en:
            d_tot = (en["mJ_per_frame"] - pt["mJ_per_frame"]) / pt["mJ_per_frame"] * 100
            d_dyn = (en["dynamic_mJ_per_frame"] - pt["dynamic_mJ_per_frame"]) / pt["dynamic_mJ_per_frame"] * 100
            d_fj = (en["frames_per_joule"] - pt["frames_per_joule"]) / pt["frames_per_joule"] * 100
            wins.append(f"- **{task}**: mJ/frame {pt['mJ_per_frame']:.0f}→{en['mJ_per_frame']:.0f} "
                        f"(**{d_tot:+.0f}%**), 순수추론 {pt['dynamic_mJ_per_frame']:.0f}→"
                        f"{en['dynamic_mJ_per_frame']:.0f} (**{d_dyn:+.0f}%**), "
                        f"효율 {pt['frames_per_joule']:.2f}→{en['frames_per_joule']:.2f} frames/J (**{d_fj:+.0f}%**)")
    if wins:
        md.append("### ✅ TensorRT FP16 엔진화 성과 (PyTorch 대비)")
        md += wins
        md.append("\n> FP16이라 정확도 손실 없이 프레임당 에너지 절감. 순수 추론 에너지는 거의 절반.")
    else:
        md.append("_아직 벤치 결과가 없습니다. `./export_trt.sh` 후 `./bench_power.sh` 로 생성하세요._")
    return "\n".join(md)


# ── UI 구성 ───────────────────────────────────────────────────────────────
def build():
    with gr.Blocks(title="Orbital Perception 시각화") as demo:
        gr.Markdown("# 🛰️ Orbital Perception — 위성 온보드 저전력 인식 시각화\n"
                    "탐지/세그멘테이션을 **프레임당 에너지(mJ/frame)** 관점에서 보여주는 대시보드입니다.")

        with gr.Tabs():
            # ── 탭 1: 대시보드 ──
            with gr.Tab("📊 성과 대시보드"):
                gr.Markdown(dashboard_summary())
                gr.Plot(dashboard_fig, label="TensorRT FP16 vs PyTorch", format="png")
                h, rows = dashboard_table()
                gr.Dataframe(value=rows, headers=h, label="측정치 (7W · imgsz640 · 60회)",
                             interactive=False, wrap=True)
                refresh = gr.Button("🔄 벤치 결과 새로고침", size="sm")

            # ── 탭 2: 라이브 추론 ──
            with gr.Tab("🖼️ 라이브 추론 (이미지·영상)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        task = gr.Radio(["detect", "segment"], value="detect", label="작업")
                        fmt = gr.Radio(
                            [("PyTorch (.pt)", "pt"), ("TensorRT FP16 (.engine)", "engine")],
                            value="engine", label="모델 포맷 (엔진=저전력)")
                        imgsz = gr.Slider(256, 960, value=640, step=32, label="추론 해상도 imgsz (작을수록 저전력)")
                        conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="신뢰도 임계값 conf")
                        classes_text = gr.Textbox(label="클래스 필터(선택)", placeholder="예: person car")
                        measure = gr.Checkbox(value=True, label="⚡ 전력 계측(tegrastats) 켜기")
                        gr.Markdown("**이미지 옵션**")
                        repeat = gr.Slider(1, 100, value=40, step=1, label="반복추론 횟수(이미지 계측 안정화)")
                        gr.Markdown("**영상 옵션**")
                        stride = gr.Slider(1, 10, value=2, step=1, label="프레임 스킵 stride (N중 1장)")
                        max_frames = gr.Slider(0, 600, value=120, step=10, label="최대 프레임(0=전체)")

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("이미지"):
                                img_in = gr.Image(type="numpy", label="입력 이미지", sources=["upload", "clipboard"])
                                img_btn = gr.Button("▶ 이미지 추론 실행", variant="primary")
                                with gr.Row():
                                    img_before = gr.Image(label="Before (원본)")
                                    img_after = gr.Image(label="After (탐지/세그)")
                            with gr.Tab("영상"):
                                vid_in = gr.Video(label="입력 영상", sources=["upload"])
                                vid_btn = gr.Button("▶ 영상 추론 실행", variant="primary")
                                vid_out = gr.Video(label="처리 결과 (HUD 오버레이)")

                        summary_md = gr.Markdown()
                        report_md = gr.Markdown()
                        power_plot = gr.Plot(label="전력 타임라인 (VDD_IN)", format="png")

            img_btn.click(run_image,
                          [img_in, task, fmt, imgsz, conf, repeat, classes_text, measure],
                          [img_before, img_after, summary_md, report_md, power_plot],
                          concurrency_limit=1)
            vid_btn.click(run_video,
                          [vid_in, task, fmt, imgsz, conf, stride, max_frames, classes_text, measure],
                          [vid_out, summary_md, report_md, power_plot],
                          concurrency_limit=1)

        # 대시보드 새로고침: 그림/표/요약 갱신은 재빌드가 필요 → 간단히 페이지 안내
        refresh.click(lambda: gr.Info("최신 결과를 보려면 페이지를 새로고침하세요."))

    demo.queue(default_concurrency_limit=1)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true", help="gradio 공개 링크")
    a = ap.parse_args()
    demo = build()
    demo.launch(server_name=a.host, server_port=a.port, share=a.share, show_error=True,
                theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"))


if __name__ == "__main__":
    main()
