#!/usr/bin/env python3
"""
Jetson 전력 계측기 (tegrastats 기반).

위성 탑재가 목표이므로 "속도"가 아니라 "프레임당 에너지(mJ/frame)"가 핵심 지표다.
tegrastats가 내보내는 VDD_IN(보드 총 입력전력, mW)을 백그라운드 스레드로 샘플링해
어떤 작업 구간의 평균/피크 전력과 소비 에너지(J)를 적분한다.

사용:
    from powerlog import PowerMeter
    with PowerMeter() as pm:
        ...무거운 작업...
    print(pm.report(n_frames=120))

INA3221 sysfs를 직접 읽는 폴백도 포함(tegrastats 미동작 환경 대비).
"""
import re
import subprocess
import threading
import time
from pathlib import Path

# tegrastats 라인에서 "VDD_IN 4976mW/4976mW" 형태의 순시 전력(첫 숫자)을 뽑는다.
_VDD_IN_RE = re.compile(r"VDD_IN\s+(\d+)mW")
# 세부 레일(있으면 참고용): CPU+GPU+CV, SOC
_CPU_GPU_RE = re.compile(r"VDD_CPU_GPU_CV\s+(\d+)mW")
_SOC_RE = re.compile(r"VDD_SOC\s+(\d+)mW")


class PowerMeter:
    """tegrastats를 백그라운드로 돌리며 전력 샘플을 모으는 컨텍스트 매니저."""

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        # 각 샘플: (t_epoch, vdd_in_W, cpu_gpu_W, soc_W)
        self.samples = []
        self._t_start = None
        self._t_end = None

    # ── 백그라운드 수집 ──────────────────────────────────────────────
    def _reader(self):
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = _VDD_IN_RE.search(line)
            if not m:
                continue
            vdd_in = int(m.group(1)) / 1000.0  # mW → W
            cg = _CPU_GPU_RE.search(line)
            so = _SOC_RE.search(line)
            cpu_gpu = int(cg.group(1)) / 1000.0 if cg else float("nan")
            soc = int(so.group(1)) / 1000.0 if so else float("nan")
            self.samples.append((time.time(), vdd_in, cpu_gpu, soc))

    def start(self):
        self._t_start = time.time()
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        except FileNotFoundError:
            # tegrastats 없음 → sysfs 폴백
            self._proc = None
            self._thread = threading.Thread(target=self._sysfs_reader, daemon=True)
            self._thread.start()
        return self

    def _sysfs_reader(self):
        """INA3221 hwmon 폴백: in7(VDD_IN 근사)_input(mV)·curr로 전력 추정이 어려워
        power*_input(µW)이 있으면 그걸 쓰고, 없으면 in*_input(mW로 노출되는 보드도 있음)."""
        rails = list(Path("/sys/class/hwmon").glob("hwmon*/"))
        while not self._stop.is_set():
            total_w = 0.0
            for h in rails:
                for p in h.glob("power*_input"):  # µW
                    try:
                        total_w += int(p.read_text()) / 1e6
                    except Exception:
                        pass
            if total_w > 0:
                self.samples.append((time.time(), total_w, float("nan"), float("nan")))
            time.sleep(self.interval_ms / 1000.0)

    def stop(self):
        self._t_end = time.time()
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # ── 집계 ────────────────────────────────────────────────────────
    def _powers(self):
        return [s[1] for s in self.samples]

    def elapsed(self) -> float:
        if self._t_start is None:
            return 0.0
        end = self._t_end or time.time()
        return end - self._t_start

    def energy_joules(self) -> float:
        """전력 샘플을 사다리꼴 적분해 소비 에너지(J)를 구한다."""
        s = self.samples
        if len(s) < 2:
            # 샘플이 부족하면 평균전력×경과시간으로 근사
            p = self._powers()
            avg = sum(p) / len(p) if p else 0.0
            return avg * self.elapsed()
        e = 0.0
        for (t0, p0, *_), (t1, p1, *_) in zip(s, s[1:]):
            e += (p0 + p1) / 2.0 * (t1 - t0)
        return e

    def avg_power(self) -> float:
        p = self._powers()
        return sum(p) / len(p) if p else 0.0

    def peak_power(self) -> float:
        p = self._powers()
        return max(p) if p else 0.0

    def report(self, n_frames: int, idle_watt: float | None = None) -> dict:
        """작업 리포트. idle_watt를 주면 유휴전력을 뺀 '순수 추론 에너지'도 계산."""
        dt = self.elapsed()
        energy = self.energy_joules()
        avg_w = self.avg_power()
        fps = n_frames / dt if dt > 0 else 0.0
        r = {
            "frames": n_frames,
            "seconds": round(dt, 3),
            "fps": round(fps, 2),
            "avg_power_W": round(avg_w, 3),
            "peak_power_W": round(self.peak_power(), 3),
            "energy_J": round(energy, 3),
            "mJ_per_frame": round(energy / n_frames * 1000.0, 1) if n_frames else None,
            "frames_per_joule": round(n_frames / energy, 3) if energy > 0 else None,
            "samples": len(self.samples),
        }
        if idle_watt is not None:
            dyn_w = max(avg_w - idle_watt, 0.0)
            dyn_e = dyn_w * dt
            r["idle_power_W"] = round(idle_watt, 3)
            r["dynamic_power_W"] = round(dyn_w, 3)
            r["dynamic_mJ_per_frame"] = round(dyn_e / n_frames * 1000.0, 1) if n_frames else None
        return r


def measure_idle(seconds: float = 2.0, interval_ms: int = 100) -> float:
    """작업 전 유휴 평균전력(W)을 측정한다."""
    pm = PowerMeter(interval_ms=interval_ms).start()
    time.sleep(seconds)
    pm.stop()
    return pm.avg_power()


if __name__ == "__main__":
    # 단독 실행: 3초 유휴 전력 측정
    import sys
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    print(f"유휴 전력 {secs}s 측정 중...")
    pm = PowerMeter().start()
    time.sleep(secs)
    pm.stop()
    print(f"평균 {pm.avg_power():.2f} W  피크 {pm.peak_power():.2f} W  "
          f"({pm.elapsed():.1f}s, 샘플 {len(pm.samples)}개)")
