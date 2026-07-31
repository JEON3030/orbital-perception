#!/usr/bin/env bash
# 전력모드 벤치마크: 같은 워크로드를 현재 nvpmodel 모드에서 돌려
# frames-per-joule / mJ-per-frame 을 뽑는다.
#
# 7W vs 15W 비교 방법(전력모드 전환엔 sudo 필요):
#   sudo nvpmodel -m 1 && ./bench_power.sh 7W       # 7W
#   sudo nvpmodel -m 0 && ./bench_power.sh 15W      # 15W
# 그러면 outputs/bench_<라벨>.json 두 개가 생기고, 아래 compare로 표를 본다:
#   ./bench_power.sh --compare
set -e
cd "$HOME/orbital-perception"

if [ "$1" = "--compare" ]; then
  export PYTHONNOUSERSITE=1
  "$HOME/wildfire-seg/.venv/bin/python" - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("outputs/bench_*.json")):
    d = json.load(open(f))
    e = d["energy"]
    rows.append((os.path.basename(f).replace("bench_","").replace(".json",""),
                 d.get("nvpmodel","?"), e["avg_power_W"], e["fps"],
                 e["mJ_per_frame"], e["frames_per_joule"]))
if not rows:
    print("bench 결과 없음. 먼저 ./bench_power.sh <라벨> 실행."); raise SystemExit
w = "{:<10} {:>8} {:>10} {:>8} {:>14} {:>16}"
print(w.format("label","mode","avgW","FPS","mJ/frame","frames/J"))
print("-"*70)
for r in rows:
    print(w.format(r[0], str(r[1]), f"{r[2]:.2f}", f"{r[3]:.1f}",
                   f"{r[4]:.1f}", f"{r[5]:.3f}"))
print("\n▶ 위성 관점: mJ/frame 낮을수록·frames/J 높을수록 좋음(에너지 효율).")
PY
  exit 0
fi

LABEL="${1:-run}"
IMG="${2:-input/sample.jpg}"
MODE=$(nvpmodel -q 2>/dev/null | grep -i "power mode" | awk '{print $NF}')

export LD_LIBRARY_PATH="$HOME/wildfire-seg/libs:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$HOME/orbital-perception/.ultra"
PY="$HOME/wildfire-seg/.venv/bin/python"

echo "[bench] label=$LABEL  mode=${MODE}  img=$IMG  (60회 반복추론)"
$PY -u perception.py "$IMG" --task detect --repeat 60 --outdir outputs >/dev/null

# 방금 만든 에너지 json을 bench_<label>.json 으로 저장 + nvpmodel 기록
STEM=$(basename "$IMG"); STEM="${STEM%.*}"
$PY - "$STEM" "$LABEL" "$MODE" <<'PY'
import json, sys
stem, label, mode = sys.argv[1], sys.argv[2], sys.argv[3]
src = f"outputs/{stem}_detect_energy.json"
d = json.load(open(src)); d["nvpmodel"] = mode
json.dump(d, open(f"outputs/bench_{label}.json","w"), ensure_ascii=False, indent=2)
print(f"  → outputs/bench_{label}.json  ({d['energy']['mJ_per_frame']} mJ/frame, "
      f"{d['energy']['frames_per_joule']} frames/J)")
PY
