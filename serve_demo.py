#!/usr/bin/env python3
"""merge 웹 데모 — 위성 취득 → 재해 분할 → seg-파생 탐지를 브라우저에서 눈으로.

의존성 없음(표준 http.server + numpy/matplotlib/PIL, 이미 설치됨). CLI인 merge에
얇은 시각화 웹을 얹는다. 파이프라인 로직은 merge를 그대로 호출 — 새 모델/규칙 없음.

  python3 serve_demo.py                # 0.0.0.0:8060
  브라우저:  http://<젯슨IP>:8060
"""
from __future__ import annotations

import io
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly

from merge import contract, train_seg, vectorize

HERE = Path(__file__).parent
STATIC = HERE / "_webdemo"
STATIC.mkdir(exist_ok=True)
MODEL = HERE / "runs" / "sat_seg" / "best.pt"

# 재해 클래스 색 (contract 탐지표와 맞춤): fire=빨강, water=파랑
FIRE_RGB = (250, 40, 40)
WATER_RGB = (30, 110, 210)

STATE = {"status": "idle", "msg": "", "stats": None, "scene": None}
LOCK = threading.Lock()


# ── 렌더 ────────────────────────────────────────────────────────────────
def _stretch(band: np.ndarray, lo=2, hi=98) -> np.ndarray:
    a, b = np.percentile(band, [lo, hi])
    if b <= a:
        b = a + 1e-6
    return np.clip((band - a) / (b - a), 0, 1)


def _rgb(cube: np.ndarray) -> np.ndarray:
    """(H,W,6) 반사율 → 자연색 RGB uint8 (red,green,blue = 밴드 2,1,0)."""
    r, g, b = (_stretch(cube[..., i]) for i in (2, 1, 0))
    return (np.dstack([r, g, b]) * 255).astype(np.uint8)


def _save_png(fig, path: Path):
    fig.savefig(path, dpi=90, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render(cube, seg, fire_doc, water_doc, tag: str):
    rgb = _rgb(cube)
    h, w = seg.shape

    # 1) 자연색
    fig, ax = plt.subplots(figsize=(6, 6 * h / w))
    ax.imshow(rgb); ax.set_title("(1) Satellite true-color (S2 20m)"); ax.axis("off")
    _save_png(fig, STATIC / "rgb.png")

    # 2) 분할 오버레이
    over = rgb.copy().astype(float)
    for cid, col in ((1, FIRE_RGB), (2, WATER_RGB)):
        m = seg == cid
        over[m] = 0.45 * over[m] + 0.55 * np.array(col)
    fig, ax = plt.subplots(figsize=(6, 6 * h / w))
    ax.imshow(over.astype(np.uint8))
    ax.set_title("(2) Disaster segmentation  (red=fire, blue=water)"); ax.axis("off")
    _save_png(fig, STATIC / "seg.png")

    # 3) 탐지 폴리곤 (윤곽)
    fig, ax = plt.subplots(figsize=(6, 6 * h / w))
    ax.imshow(rgb); ax.axis("off")
    for doc, col in ((fire_doc, "#fa2828"), (water_doc, "#1e6ed2")):
        for d in doc["detections"]:
            ring = d.get("polygon") or d.get("geometry")
            if not ring:
                continue
            ax.add_patch(MplPoly(np.array(ring), closed=True, fill=False,
                                 edgecolor=col, linewidth=1.4))
    nf, nw = len(fire_doc["detections"]), len(water_doc["detections"])
    ax.set_title(f"(3) seg-derived detection  fire={nf}, water={nw}")
    _save_png(fig, STATIC / "det.png")


# ── 파이프라인 (merge 호출) ──────────────────────────────────────────────
def run_pipeline(bbox, start, end, cloud):
    from merge import acquire
    with LOCK:
        STATE.update(status="running", msg="① 위성 장면 취득 중 (STAC)…")
    try:
        scene = STATIC / "scene.npy"
        meta = acquire.acquire_scene(tuple(bbox), start, end, cloud_max=cloud,
                                     out_npy=str(scene))
        with LOCK:
            STATE["msg"] = "② 재해 분할 추론 중…"
        cube = np.load(scene)
        dev = train_seg.resolve_device("auto")     # GPU 우선
        with LOCK:
            STATE["msg"] = f"② 재해 분할 추론 중 [{dev}]…"
        seg, classes = train_seg.predict_mask(cube, str(MODEL), device=str(dev))
        with LOCK:
            STATE["msg"] = "③ seg-파생 탐지 벡터화 중…"
        fire_doc = vectorize.vectorize_mask(seg, 12, min_area_px=25, label_value=1)
        water_doc = vectorize.vectorize_mask(seg, 13, min_area_px=25, label_value=2)
        render(cube, seg, fire_doc, water_doc, meta["scene_id"])

        frac = {classes[c]: round(float((seg == c).mean()) * 100, 2)
                for c in range(len(classes))}
        fkm = sum(d.get("area_m2", 0) for d in fire_doc["detections"]) / 1e6
        wkm = sum(d.get("area_m2", 0) for d in water_doc["detections"]) / 1e6
        stats = {
            "scene_id": meta["scene_id"], "datetime": meta["datetime"],
            "cloud": meta["cloud_cover"], "shape": list(meta["shape"]),
            "gsd": meta["gsd_m"], "epsg": meta["proj_epsg"],
            "device": str(dev),
            "frac": frac,
            "fire": {"n": len(fire_doc["detections"]), "km2": round(fkm, 3)},
            "water": {"n": len(water_doc["detections"]), "km2": round(wkm, 3)},
        }
        (STATIC / "stats.json").write_text(json.dumps(stats, ensure_ascii=False))
        with LOCK:
            STATE.update(status="done", msg="완료", stats=stats, scene=meta["scene_id"])
    except Exception as e:
        with LOCK:
            STATE.update(status="error", msg=f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ── HTML ────────────────────────────────────────────────────────────────
PRESETS = [
    ("울진 산불 (2022-04)", "129.28,36.95,129.42,37.05", "2022-03-01", "2022-04-30", 20),
    ("소양호 (2022 여름)", "127.75,37.90,127.90,38.02", "2022-07-01", "2022-09-30", 20),
]


def page() -> bytes:
    with LOCK:
        st, msg, s = STATE["status"], STATE["msg"], STATE["stats"]
    opts = "".join(
        f'<option value="{bb}|{a}|{b}|{c}">{name}</option>'
        for name, bb, a, b, c in PRESETS)

    if s:
        f = "".join(f"<tr><td>{k}</td><td>{v}%</td></tr>" for k, v in s["frac"].items())
        panel = f"""
        <div class=meta>
          <b>{s['scene_id']}</b> · {s['datetime'][:10]} · 구름 {s['cloud']}% ·
          격자 {s['shape'][0]}×{s['shape'][1]} @ {s['gsd']}m · UTM {s['epsg']} ·
          추론 <b style="color:#3fb950">{s.get('device','?').upper()}</b>
        </div>
        <div class=cards>
          <div class="card fire"><div class=big>{s['fire']['km2']}</div>km² 산불 · {s['fire']['n']}건</div>
          <div class="card water"><div class=big>{s['water']['km2']}</div>km² 물 · {s['water']['n']}건</div>
        </div>
        <div class=imgs>
          <figure><img src="/_webdemo/rgb.png?t={s['scene_id']}"></figure>
          <figure><img src="/_webdemo/seg.png?t={s['scene_id']}"></figure>
          <figure><img src="/_webdemo/det.png?t={s['scene_id']}"></figure>
        </div>
        <table class=frac><caption>분할 클래스 비율</caption>{f}</table>
        """
    else:
        panel = "<p class=hint>아래에서 지역을 골라 <b>실행</b>을 누르면 취득→분할→탐지가 돕니다.</p>"

    busy = "running" == st
    banner = ""
    if busy:
        banner = f'<div class="banner run">⏳ {msg}</div><script>setTimeout(()=>location.reload(),3000)</script>'
    elif st == "error":
        banner = f'<div class="banner err">✗ {msg}</div>'

    return f"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>merge — 위성 탐지·분할 데모</title><style>
*{{box-sizing:border-box}}body{{font:15px/1.5 system-ui,'Noto Sans KR',sans-serif;
margin:0;background:#0d1117;color:#e6edf3}}
header{{padding:18px 22px;background:#161b22;border-bottom:1px solid #30363d}}
h1{{margin:0;font-size:19px}}h1 span{{color:#58a6ff}}
.sub{{color:#8b949e;font-size:13px;margin-top:3px}}
main{{max-width:1180px;margin:0 auto;padding:22px}}
form{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;background:#161b22;
padding:14px;border-radius:10px;border:1px solid #30363d}}
select,button{{font:inherit;padding:9px 12px;border-radius:8px;border:1px solid #30363d;
background:#0d1117;color:#e6edf3}}
button{{background:#238636;border-color:#238636;font-weight:600;cursor:pointer}}
button:disabled{{opacity:.5;cursor:wait}}
.banner{{margin:14px 0;padding:11px 14px;border-radius:8px}}
.banner.run{{background:#1f3a5f;border:1px solid #388bfd}}
.banner.err{{background:#4b1e22;border:1px solid #f85149}}
.meta{{margin:16px 0 8px;color:#8b949e;font-size:13px}}
.cards{{display:flex;gap:14px;margin:12px 0}}
.card{{flex:1;padding:16px;border-radius:10px;text-align:center}}
.card.fire{{background:#3a1414;border:1px solid #fa2828}}
.card.water{{background:#0f2338;border:1px solid #1e6ed2}}
.big{{font-size:30px;font-weight:700}}
.imgs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:16px}}
figure{{margin:0}}img{{width:100%;border-radius:10px;border:1px solid #30363d}}
table.frac{{margin-top:18px;border-collapse:collapse;font-size:14px}}
table.frac caption{{text-align:left;color:#8b949e;margin-bottom:6px}}
table.frac td{{padding:4px 18px 4px 0;border-bottom:1px solid #21262d}}
.hint{{color:#8b949e}}code{{color:#79c0ff}}
</style></head><body>
<header><h1><span>merge</span> · 위성 탐지·분할 데모</h1>
<div class=sub>Sentinel-2 20m 6밴드 → 재해 분할(UNet) → seg-파생 탐지 · 파이프라인은 <code>python -m merge</code> 그대로</div>
</header><main>
<form method=POST action=/run>
  <label>지역&nbsp;<select name=preset>{opts}</select></label>
  <button type=submit {'disabled' if busy else ''}>▶ 실행 (취득→분할→탐지)</button>
  <span class=sub>실측 1~2분 소요 (실제 STAC 취득)</span>
</form>
{banner}
{panel}
</main></body></html>""".encode()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, page())
        elif self.path.startswith("/_webdemo/"):
            p = STATIC / self.path.split("?")[0].split("/_webdemo/")[1]
            if p.exists() and p.suffix == ".png":
                self._send(200, p.read_bytes(), "image/png")
            else:
                self._send(404, b"no")
        else:
            self._send(404, b"no")

    def do_POST(self):
        if self.path != "/run":
            return self._send(404, b"no")
        n = int(self.headers.get("Content-Length", 0))
        q = parse_qs(self.rfile.read(n).decode())
        preset = q.get("preset", [""])[0]
        try:
            bb, a, b, c = preset.split("|")
            bbox = [float(x) for x in bb.split(",")]
            with LOCK:
                if STATE["status"] == "running":
                    raise RuntimeError("이미 실행 중")
            threading.Thread(target=run_pipeline,
                             args=(bbox, a, b, float(c)), daemon=True).start()
        except Exception as e:
            with LOCK:
                STATE.update(status="error", msg=str(e))
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8060
    # 첫 화면에 기존 결과가 보이도록 캐시된 stats 복원
    sp = STATIC / "stats.json"
    if sp.exists():
        STATE.update(status="done", stats=json.loads(sp.read_text()))
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    print(f"■ merge 웹 데모 → http://0.0.0.0:{port}  (Ctrl-C 종료)")
    srv.serve_forever()
