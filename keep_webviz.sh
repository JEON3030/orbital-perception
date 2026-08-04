#!/usr/bin/env bash
# webviz 를 세션과 분리해 "안 닫히게" 상시 구동하는 keeper.
#  - setsid+nohup 으로 띄우면 이 스크립트를 실행한 셸/세션이 죽어도 살아남는다.
#  - webviz 가 어떤 이유로 종료돼도 while 루프가 5초 뒤 자동 재기동한다.
#  - LAN URL(http://<보드IP>:7860)은 프로세스가 살아있는 한 안 바뀜(가장 안정적).
#  - --share 의 gradio.live 링크는 원래 임시라 재기동 때마다 새로 발급된다.
#
# 사용:
#   setsid nohup ./keep_webviz.sh >/dev/null 2>&1 &   # 세션 분리 상시 구동
#   grep -aoE 'https://[a-z0-9]+\.gradio\.live' webviz_keeper.log | tail -1   # 현재 공유링크
#   pkill -f keep_webviz.sh; pkill -f webviz.py                                # 중지

cd "$HOME/orbital-perception" || exit 1
LOG="$HOME/orbital-perception/webviz_keeper.log"

# 이미 떠 있는 webviz 있으면 정리(포트 7860 충돌 방지)
pkill -f "webviz.py" 2>/dev/null
sleep 1

while true; do
  echo "[keeper $(date '+%F %T')] starting webviz --share" >> "$LOG"
  ./webviz.sh --share >> "$LOG" 2>&1
  code=$?
  echo "[keeper $(date '+%F %T')] webviz exited (code $code) -> restart in 5s" >> "$LOG"
  sleep 5
done
