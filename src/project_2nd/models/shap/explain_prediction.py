"""단일 예측에 대한 SHAP 설명을 DB 스키마(predictions.shap_top_features)에 맞게 뽑아내는 공용 함수.

스키마 출처: src/project_2nd/db/테이블_설명.md
  predictions.shap_top_features (JSON, NULL 허용):
    [{"feature": "store_age_months", "shap_value": -0.12, "feature_value": 6}, ...]

트리 기반 모델(RandomForest/ExtraTrees/LightGBM/XGBoost/CatBoost) 전부 shap.TreeExplainer로
지원되므로, 어떤 팀원이 어떤 트리 모델을 최종으로 쓰든 이 함수 하나로 재사용 가능하다.
"""

from typing import Any

import numpy as np
import pandas as pd
import shap


def explain_prediction(
    model: Any,
    row: pd.Series | pd.DataFrame,
    feature_names: list[str],
    top_k: int = 5,
    explainer: shap.TreeExplainer | None = None,
) -> list[dict]:
    """모델 하나 + 매장 한 건(row)에 대해 shap_top_features 형식의 리스트를 반환.

    Args:
        model: 학습된 트리 기반 분류기(RandomForest/ExtraTrees/LightGBM/XGBoost/CatBoost 등)
        row: 피처 1행(pd.Series) 또는 1행짜리 pd.DataFrame
        feature_names: row의 컬럼 순서와 일치하는 피처 이름 리스트
        top_k: |shap_value| 기준 상위 몇 개까지 반환할지
        explainer: 이미 만들어둔 shap.TreeExplainer가 있으면 재사용(매 호출마다 새로
            만들면 느려짐 — 배치로 여러 건 처리할 땐 explainer를 한 번만 만들어서 넘길 것)

    Returns:
        [{"feature": str, "shap_value": float, "feature_value": Any}, ...]
        (|shap_value| 내림차순, top_k개)
    """
    X = row.to_frame().T if isinstance(row, pd.Series) else row

    if explainer is None:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1])[0]  # 이진분류: 양성 클래스(1)
    elif np.asarray(shap_values).ndim == 3:
        sv = np.asarray(shap_values)[0, :, 1]
    else:
        sv = np.asarray(shap_values)[0]

    return format_shap_row(sv, X.iloc[0], feature_names, top_k)


def _to_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def format_shap_row(sv_row: np.ndarray, X_row: pd.Series, feature_names: list[str], top_k: int = 5) -> list[dict]:
    """이미 계산된 SHAP 값 한 행(sv_row)을 shap_top_features 형식으로 변환.

    shap_values()가 이미 계산돼 있을 때(예: 여러 용도로 재사용) 재계산 없이
    포맷팅만 하고 싶으면 이 함수를 직접 쓰면 된다.
    """
    order = np.argsort(-np.abs(sv_row))[:top_k]
    return [
        {
            "feature": feature_names[idx],
            "shap_value": round(float(sv_row[idx]), 6),
            "feature_value": _to_jsonable(X_row.iloc[idx]),
        }
        for idx in order
    ]


def compute_shap_matrix(model: Any, X: pd.DataFrame) -> np.ndarray:
    """X 전체에 대한 SHAP 값 행렬(n_rows, n_features)을 한 번만 계산해서 반환.

    이 결과를 format_shap_row로 여러 번 포맷팅하거나, np.abs(...).mean(axis=0)으로
    전역 중요도를 뽑는 등 재사용할 수 있다 — TreeExplainer 재호출(제일 비싼 부분)을
    한 번으로 줄이기 위한 함수.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    if np.asarray(shap_values).ndim == 3:
        return np.asarray(shap_values)[:, :, 1]
    return np.asarray(shap_values)


def explain_batch(model: Any, X: pd.DataFrame, top_k: int = 5) -> list[list[dict]]:
    """여러 행을 한 번에 처리(explainer를 한 번만 만들어서 재사용, 배치가 훨씬 빠름)."""
    feature_names = list(X.columns)
    sv_all = compute_shap_matrix(model, X)
    return [format_shap_row(sv_all[i], X.iloc[i], feature_names, top_k) for i in range(len(X))]
