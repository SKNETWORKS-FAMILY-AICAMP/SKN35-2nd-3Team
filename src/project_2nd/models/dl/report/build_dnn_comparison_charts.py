"""저장된 DNN 3회 실행 결과로 발표용 비교 그래프를 생성한다."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


REPORT_DIR = Path(__file__).resolve().parent
DL_DIR = REPORT_DIR.parent

RUNS = (
    {
        "key": "run1",
        "label": "1차 · 7E/P2",
        "short": "1차\n7E/P2",
        "path": DL_DIR
        / "saved"
        / "runs"
        / "DNN_20260828_004659_cc977ca_pjw_official",
    },
    {
        "key": "run2",
        "label": "2차 · 7E/P5",
        "short": "2차\n7E/P5",
        "path": DL_DIR
        / "26.08.28-dl-0920"
        / "DNN_20260828_092220_87442d5_pjw_official",
    },
    {
        "key": "run3",
        "label": "3차 · 100E/P5",
        "short": "3차\n100E/P5",
        "path": DL_DIR
        / "26.08.28-dl-e100-p5"
        / "DNN_20260828_094006_7e84a4a_pjw_official",
    },
)

COLORS = ("#315A7D", "#67A9CF", "#E07A5F")
GRID_COLOR = "#D6DCE2"
TEXT_COLOR = "#17212B"
MUTED_COLOR = "#52606D"


def load_json(path: Path) -> dict[str, object] | list[dict[str, object]]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_runs() -> list[dict[str, object]]:
    loaded: list[dict[str, object]] = []
    for definition in RUNS:
        run_dir = Path(definition["path"])
        if not run_dir.is_dir():
            raise FileNotFoundError(f"DNN 실행 폴더가 없습니다: {run_dir}")
        metrics = load_json(run_dir / "metrics" / "metrics.json")
        history = load_json(run_dir / "metrics" / "history.json")
        manifest = load_json(run_dir / "run_manifest.json")
        shap_importance = pd.read_csv(run_dir / "shap" / "shap_global_importance.csv")
        loaded.append(
            {
                **definition,
                "metrics": metrics,
                "history": history,
                "manifest": manifest,
                "shap": shap_importance,
            }
        )
    return loaded


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Malgun Gothic",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": MUTED_COLOR,
            "ytick.color": MUTED_COLOR,
            "text.color": TEXT_COLOR,
            "axes.titleweight": "bold",
            "axes.titlesize": 20,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


def save_figure(figure: plt.Figure, filename: str) -> None:
    figure.savefig(
        REPORT_DIR / filename,
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def plot_test_metrics(runs: list[dict[str, object]]) -> None:
    metric_specs = (
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("accuracy", "Accuracy"),
    )
    x = np.arange(len(metric_specs), dtype=np.float64)
    width = 0.24
    figure, axis = plt.subplots(figsize=(16, 9))

    for run_index, (run, color) in enumerate(zip(runs, COLORS)):
        test_metrics = run["metrics"]["test"]
        values = [float(test_metrics[key]) for key, _ in metric_specs]
        positions = x + (run_index - 1) * width
        bars = axis.bar(
            positions,
            values,
            width=width,
            color=color,
            label=run["label"],
        )
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, fontsize=10)

    axis.set_title("DNN 3회 Test 성능 비교", pad=24)
    axis.set_ylabel("평가지표 값 (0~1)")
    axis.set_xticks(x, [label for _, label in metric_specs])
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    axis.text(
        0.01,
        -0.12,
        "주요 비교 지표: 클래스 불균형을 고려해 PR-AUC를 우선 확인",
        transform=axis.transAxes,
        color=MUTED_COLOR,
        fontsize=12,
    )
    figure.subplots_adjust(top=0.84, bottom=0.16, left=0.08, right=0.98)
    save_figure(figure, "01_test_metrics_comparison.png")


def plot_validation_progress(runs: list[dict[str, object]]) -> None:
    figure, (initial_axis, long_axis) = plt.subplots(1, 2, figsize=(16, 9))

    for run, color, style, marker in zip(
        runs,
        COLORS,
        ("-", "--", "-."),
        ("o", "s", "^"),
    ):
        history = pd.DataFrame(run["history"])
        initial = history.loc[history["epoch"] <= 7]
        initial_axis.plot(
            initial["epoch"],
            initial["validation_pr_auc"],
            color=color,
            linestyle=style,
            marker=marker,
            linewidth=2.4,
            markersize=6,
            label=run["label"],
        )

    initial_axis.set_title("초기 7 Epoch", pad=18)
    initial_axis.set_xlabel("Epoch")
    initial_axis.set_ylabel("Validation PR-AUC")
    initial_axis.set_xticks(np.arange(1, 8))
    initial_axis.set_ylim(0.31, 0.42)
    initial_axis.grid(color=GRID_COLOR, linewidth=0.8)
    initial_axis.legend(loc="lower right", frameon=False)
    initial_axis.text(
        0.04,
        0.93,
        "동일 데이터·시드·구조로 1~7 Epoch 결과가 동일",
        transform=initial_axis.transAxes,
        color=MUTED_COLOR,
        fontsize=11,
        va="top",
    )

    extended = runs[2]
    history = pd.DataFrame(extended["history"])
    metrics = extended["metrics"]
    best_epoch = int(metrics["best_epoch"])
    best_value = float(metrics["best_validation_pr_auc"])
    stop_epoch = int(history["epoch"].max())
    baseline = float(runs[1]["metrics"]["validation"]["pr_auc"])

    long_axis.plot(
        history["epoch"],
        history["validation_pr_auc"],
        color=COLORS[2],
        linewidth=2.6,
        label="3차 Validation PR-AUC",
    )
    long_axis.axhline(
        baseline,
        color=COLORS[1],
        linewidth=1.7,
        linestyle="--",
        label=f"7 Epoch 기준 {baseline:.4f}",
    )
    long_axis.scatter(
        [best_epoch],
        [best_value],
        s=90,
        color=TEXT_COLOR,
        zorder=5,
    )
    long_axis.annotate(
        f"최적 {best_epoch} Epoch\nPR-AUC {best_value:.4f}",
        xy=(best_epoch, best_value),
        xytext=(-135, -58),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": TEXT_COLOR},
        fontsize=12,
    )
    long_axis.axvline(stop_epoch, color=MUTED_COLOR, linestyle=":", linewidth=1.6)
    long_axis.text(
        stop_epoch - 1,
        best_value - 0.012,
        f"조기 종료 {stop_epoch}E",
        ha="right",
        color=MUTED_COLOR,
        fontsize=11,
    )
    long_axis.set_title("100 Epoch·Patience 5 실행", pad=18)
    long_axis.set_xlabel("Epoch")
    long_axis.set_ylabel("Validation PR-AUC")
    long_axis.grid(color=GRID_COLOR, linewidth=0.8)
    long_axis.legend(loc="lower right", frameon=False)

    figure.suptitle("Validation PR-AUC 학습 추이", fontsize=24, fontweight="bold", y=0.98)
    figure.subplots_adjust(top=0.86, bottom=0.11, left=0.07, right=0.98, wspace=0.20)
    save_figure(figure, "02_validation_pr_auc_progress.png")


def plot_extended_training_gain(runs: list[dict[str, object]]) -> None:
    baseline = runs[1]["metrics"]["test"]
    extended = runs[2]["metrics"]["test"]
    specs = (
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("accuracy", "Accuracy"),
    )
    labels = [label for _, label in specs]
    gains = np.array(
        [(float(extended[key]) - float(baseline[key])) * 100 for key, _ in specs]
    )

    figure, axis = plt.subplots(figsize=(16, 9))
    y = np.arange(len(labels))
    bars = axis.barh(y, gains, color=COLORS[2], height=0.58)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("절대 개선 폭 (%p)")
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    axis.set_title("7 Epoch 대비 장기 학습의 Test 성능 개선 폭", pad=24)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlim(0, max(gains) * 1.22)
    for bar, gain in zip(bars, gains):
        axis.text(
            gain + max(gains) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"+{gain:.2f}%p",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
    axis.text(
        0.01,
        -0.10,
        "비교: 2차(7E/P5) → 3차(max 100E/P5, 최적 81E)",
        transform=axis.transAxes,
        color=MUTED_COLOR,
        fontsize=12,
    )
    figure.subplots_adjust(top=0.87, bottom=0.15, left=0.15, right=0.96)
    save_figure(figure, "03_extended_training_gain.png")


def plot_shap_comparison(runs: list[dict[str, object]]) -> None:
    baseline = runs[1]["shap"].set_index("feature")["mean_abs_shap_logit"]
    extended = runs[2]["shap"].set_index("feature")["mean_abs_shap_logit"]
    selected = list(dict.fromkeys([*baseline.head(10).index, *extended.head(10).index]))
    frame = pd.DataFrame(
        {
            "7 Epoch (1·2차 동일)": baseline.reindex(selected).fillna(0.0),
            "100E/P5 (최적 81E)": extended.reindex(selected).fillna(0.0),
        }
    )
    frame["sort"] = frame.max(axis=1)
    frame = frame.sort_values("sort", ascending=True).drop(columns="sort")

    y = np.arange(len(frame))
    height = 0.36
    figure, axis = plt.subplots(figsize=(16, 9))
    axis.barh(
        y - height / 2,
        frame.iloc[:, 0],
        height=height,
        color=COLORS[1],
        label=frame.columns[0],
    )
    axis.barh(
        y + height / 2,
        frame.iloc[:, 1],
        height=height,
        color=COLORS[2],
        label=frame.columns[1],
    )
    axis.set_yticks(y, frame.index)
    axis.set_xlabel("평균 절대 SHAP 값 (모델 logit 기여도)")
    axis.set_title("학습 길이에 따른 주요 피처 중요도 변화", pad=24)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False)
    figure.subplots_adjust(top=0.87, bottom=0.13, left=0.25, right=0.97)
    save_figure(figure, "04_shap_feature_importance_comparison.png")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    runs = load_runs()
    plot_test_metrics(runs)
    plot_validation_progress(runs)
    plot_extended_training_gain(runs)
    plot_shap_comparison(runs)
    print(f"발표용 그래프 생성 완료: {REPORT_DIR}")


if __name__ == "__main__":
    main()
