#!/usr/bin/env python3
"""한국 국가교통정보센터(ITS) 공개 교통 CCTV 조회 → 실시간 탐지 연결.

ITS 오픈API(openapi.its.go.kr) 는 지정한 위경도 박스 안의 교통 CCTV 목록과 각 CCTV 의
**실시간 HLS 스트림 URL(cctvurl)** 을 준다(cctvType=1 = 실시간 스트리밍). 이 URL 을
live_cctv.py 에 넣으면 젯슨에서 그대로 실시간 객체탐지가 된다.

무료 API 키: https://www.its.go.kr → 오픈API → 인증키 신청(즉시 발급). 키는 환경변수
ITS_API_KEY 또는 --key 로 준다.

사용:
  ./its_cctv.sh --key <KEY> --region seoul --list          # 근처 CCTV 목록
  ./its_cctv.sh --key <KEY> --region seoul --index 0 --run  # 0번 CCTV 실시간 탐지
  ./its_cctv.sh --key <KEY> --bbox 126.9 127.1 37.5 37.6 --list
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

ENDPOINT = "https://openapi.its.go.kr:9443/cctvInfo"

# 대략적 위경도 박스(minX maxX minY maxY) — 필요시 --bbox 로 덮어씀
REGIONS = {
    "seoul":   (126.85, 127.10, 37.48, 37.62),
    "busan":   (128.95, 129.15, 35.08, 35.24),
    "daegu":   (128.50, 128.68, 35.80, 35.92),
    "incheon": (126.60, 126.75, 37.40, 37.52),
    "gyeongbu": (127.00, 127.20, 37.20, 37.50),   # 경부고속 일부
}


def fetch(key, bbox, road_type="ex", cctv_type=1):
    minx, maxx, miny, maxy = bbox
    q = urllib.parse.urlencode({
        "apiKey": key, "type": road_type, "cctvType": cctv_type,
        "minX": minx, "maxX": maxx, "minY": miny, "maxY": maxy,
        "getType": "json"})
    url = f"{ENDPOINT}?{q}"
    with urllib.request.urlopen(url, timeout=15) as r:
        raw = r.read().decode("utf-8", "replace")
    data = json.loads(raw)
    # 응답 래퍼가 버전마다 다름(response.data / body / data) → 견고하게 훑는다
    items = None
    if isinstance(data, dict):
        for path in (("response", "data"), ("body", "items"), ("data",), ("body",)):
            cur = data
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok and isinstance(cur, list):
                items = cur
                break
    if items is None:
        raise SystemExit(f"[ITS] 예상외 응답(키/파라미터 확인):\n{raw[:500]}")
    out = []
    for it in items:
        out.append({
            "name": it.get("cctvname") or it.get("cctvName") or "?",
            "url": it.get("cctvurl") or it.get("cctvUrl") or "",
            "x": it.get("coordx"), "y": it.get("coordy"),
            "fmt": it.get("cctvformat") or it.get("cctvFormat") or "",
        })
    return [o for o in out if o["url"]]


def main():
    ap = argparse.ArgumentParser(description="ITS 공개 교통 CCTV 조회/실행")
    ap.add_argument("--key", default=os.environ.get("ITS_API_KEY"),
                    help="ITS 인증키(또는 환경변수 ITS_API_KEY)")
    ap.add_argument("--region", choices=list(REGIONS), default="seoul")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("minX", "maxX", "minY", "maxY"),
                    help="위경도 박스로 지역 직접 지정")
    ap.add_argument("--road", choices=["ex", "its"], default="ex",
                    help="ex=고속도로, its=국도")
    ap.add_argument("--list", action="store_true", help="CCTV 목록만 출력")
    ap.add_argument("--index", type=int, default=None, help="실행할 CCTV 번호")
    ap.add_argument("--run", action="store_true", help="--index CCTV 로 실시간 탐지 실행")
    ap.add_argument("--track", action="store_true", help="ByteTrack 추적")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()

    if not args.key:
        raise SystemExit("ITS 인증키 필요: --key <KEY> 또는 ITS_API_KEY 환경변수 "
                         "(무료 발급: https://www.its.go.kr → 오픈API)")

    bbox = tuple(args.bbox) if args.bbox else REGIONS[args.region]
    cams = fetch(args.key, bbox, road_type=args.road)
    print(f"[ITS] {args.road} · {args.region if not args.bbox else args.bbox} "
          f"→ CCTV {len(cams)}개")
    for i, c in enumerate(cams):
        mark = " ◀" if args.index == i else ""
        print(f"  [{i:2d}] {c['name']}  ({c['fmt']}){mark}")

    if args.run:
        if args.index is None or not (0 <= args.index < len(cams)):
            raise SystemExit("--run 에는 유효한 --index 필요")
        target = cams[args.index]
        print(f"\n▶ 실시간 탐지: {target['name']}\n  {target['url']}\n")
        # live_cctv 를 같은 프로세스로 구동(모델·엔진 재사용)
        sys.argv = ["live_cctv.py", "--source", target["url"], "--port", str(args.port)]
        if args.track:
            sys.argv.append("--track")
        import live_cctv
        live_cctv.main()


if __name__ == "__main__":
    main()
