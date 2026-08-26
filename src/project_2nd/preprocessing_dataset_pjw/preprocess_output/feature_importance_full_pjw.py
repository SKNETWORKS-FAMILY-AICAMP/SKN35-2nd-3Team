"""정제본(refined) 전체 컬럼에 대한 LightGBM feature importance 순위.

compare_baseline_5fold_pjw.py는 top 15까지만 저장하는데, 팀원(딥러닝 담당)이
전체 피처 순위를 참고하고 싶어해서 전체 컬럼 버전을 별도로 저장한다.

importance_type='split' (LightGBM 기본값, 트리 분할에 사용된 횟수 기준) 5-fold 평균.
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
REFINED = REPO_ROOT / "data" / "processed" / "modeling_dataset_refined_pjw.csv"
NON_FEATURE = {"store_id", "fold", "is_closed_next"}

df = pd.read_csv(REFINED)
target = "is_closed_next"
feature_cols = [c for c in df.columns if c not in NON_FEATURE]

for c in feature_cols:
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c] = df[c].astype("category")

importances = []
for k in sorted(df["fold"].unique()):
    train = df[df["fold"] != k]
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(train[feature_cols], train[target])
    importances.append(pd.Series(model.feature_importances_, index=feature_cols))

mean_imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
total = mean_imp.sum()

out = pd.DataFrame({
    "rank": range(1, len(mean_imp) + 1),
    "column": mean_imp.index,
    "mean_split_importance": mean_imp.values.round(1),
    "pct_of_total": (mean_imp.values / total * 100).round(2),
})
# *.csv는 .gitignore 대상이라 다른 검증 결과 파일들과 같은 확장자(.txt)로 저장
out_path = Path(__file__).resolve().parent / "feature_importance_full_result_pjw.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(out.to_string(index=False))
    f.write("\n")
print(f"저장: {out_path}")
print(out.to_string(index=False))
