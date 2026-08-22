import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import add_industry_density, add_transit_accessibility  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULT_TXT = Path(__file__).resolve().parent / "add_features_result.txt"

log_lines = []


def log(msg: str) -> None:
    log_lines.append(msg)


def process(label: str, path_in: Path, path_out: Path, active_mask_fn, bus, subway) -> None:
    t0 = time.time()
    df = pd.read_csv(path_in, encoding="utf-8-sig")
    log(f"[{label}] 로드: {len(df)}행, {time.time() - t0:.1f}s")

    t0 = time.time()
    df = add_industry_density(df, radius_m=300.0, active_mask=active_mask_fn(df))
    log(f"[{label}] 밀집도 계산: {time.time() - t0:.1f}s")

    t0 = time.time()
    df = add_transit_accessibility(df, bus, subway)
    log(f"[{label}] 대중교통 접근성 계산: {time.time() - t0:.1f}s")

    df.to_csv(path_out, index=False, encoding="utf-8-sig")
    log(f"[{label}] 저장: {path_out.name}")
    log(f"  업종밀집도 기술통계:\n{df['업종밀집도'].describe().to_string()}")
    log(f"  최근접버스정류장_거리m 기술통계:\n{df['최근접버스정류장_거리m'].describe().to_string()}")
    log(f"  최근접지하철역_거리m 기술통계:\n{df['최근접지하철역_거리m'].describe().to_string()}")
    log("")


bus = pd.read_csv(RAW / "bus_stops_nationwide.csv", encoding="cp949")
subway = pd.read_excel(RAW / "subway_stations_nationwide.xlsx")
log(f"버스정류장 {len(bus)}개, 지하철역 {len(subway)}개 로드\n")

process(
    "all_industries",
    PROCESSED / "all_industries_clean.csv",
    PROCESSED / "all_industries_features.csv",
    lambda df: df["영업상태"] == "영업/정상",
    bus,
    subway,
)
process(
    "retail_seoul",
    PROCESSED / "retail_seoul_clean.csv",
    PROCESSED / "retail_seoul_features.csv",
    lambda df: df["y"] == 0,
    bus,
    subway,
)

with open(RESULT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print("done")
