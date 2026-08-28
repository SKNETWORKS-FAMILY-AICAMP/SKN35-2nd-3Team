"""화면 간 공용 UI 컴포넌트 (위험도 배지, 점수 카드 등)

seoul-biz-ui-logic.md(3, 3-2번)에서 확정된 생존점수 등급/색상 규칙과,
팀원 "프로젝트 한계점" 문서 반영 사항(모델링 제외 업종, 표본부족 배지,
업종전환 6개월 caveat, 절대점수 대신 상대순위 병기)을 한 곳에 모아둔다.
화면(pages/*.py, app.py)에서는 여기 함수만 가져다 쓰면 됨.
"""

import streamlit as st

# ---------------------------------------------------------------
# 생존점수 등급 (ui-logic.md 3번)
#   survival_score = round((1 - 폐업확률) * 100)
# ---------------------------------------------------------------
_GRADES = (
    (80, "우수", "good", "#2e7d32"),
    (65, "양호", "warning", "#f9a825"),
    (45, "주의", "serious", "#ef6c00"),
    (0, "위험", "critical", "#c62828"),
)


def proba_to_survival_score(closure_proba: float) -> int:
    """모델이 뱉는 폐업확률(0~1) -> 생존점수(0~100). predictions.score가 폐업확률이라는
    전제(ui-logic.md 3번, 2026-08-26 확인됨)를 그대로 따른다."""
    return round((1 - closure_proba) * 100)


def score_to_grade(survival_score: float) -> dict:
    """0~100 생존점수 -> {'label','key','color'}"""
    for threshold, label, key, color in _GRADES:
        if survival_score >= threshold:
            return {"label": label, "key": key, "color": color}
    return {"label": "위험", "key": "critical", "color": "#c62828"}


def grade_badge(survival_score: float, percentile: float | None = None) -> None:
    """점수 배지. percentile(0~100, 상위 %)이 있으면 함께 표기.
    3-2 문서 결론: 이 모델 성능 수준(ROC-AUC 0.73~0.75대)에서는 절대 점수 단독보다
    상대 순위를 병기하는 게 더 정직한 표현이라 기본으로 같이 넣는다."""
    grade = score_to_grade(survival_score)
    pct_txt = f" · 상위 {percentile:.0f}%" if percentile is not None else ""
    st.markdown(
        f"""<div style="display:inline-block;padding:4px 12px;border-radius:999px;
             background:{grade['color']}22;color:{grade['color']};
             font-weight:600;font-size:0.9rem;">
             {grade['label']} · {survival_score}점{pct_txt}
             </div>""",
        unsafe_allow_html=True,
    )


def confidence_notice() -> None:
    """모델 신뢰도 문구 — 점수가 보이는 곳엔 항상 붙인다(과신 방지, ui-logic.md 3번)."""
    st.caption("이 점수는 예측 참고용이며 실제와 다를 수 있어요.")


def low_sample_badge() -> None:
    """3-2: dong_industry_historical_rate 표본 30건 미만 -> 업종 평균 대체값 사용 시 표시."""
    st.caption("동네 표본 부족 — 업종 평균 기준 추정치예요.")


def short_term_switch_caveat() -> None:
    """3-2: industry_survival_stats는 전환 후 최대 6개월까지만 관측한 단기 지표."""
    st.caption("전환 후 6개월 내 단기 생존율 기준이에요.")


# ---------------------------------------------------------------
# 3-2: 모델링 제외 업종군
#   TODO: 아래 문자열은 industries.custom_group 실제 값과 대조해서 정확히
#   맞출 것 (10개 대분류 그룹명 원본을 팀원한테 확인 필요 — 지금은 한계점
#   문서에 언급된 이름 그대로 적어둔 추정치).
# ---------------------------------------------------------------
EXCLUDED_CUSTOM_GROUPS = ["과학·기술", "부동산", "시설관리·임대"]


def is_excluded_industry(custom_group: str) -> bool:
    return custom_group in EXCLUDED_CUSTOM_GROUPS


def excluded_industry_notice() -> None:
    st.caption("이 업종군은 예측 대상이 아니에요.")


def score_card(title: str, survival_score: float, percentile: float | None = None,
                shap_lines: list[str] | None = None, is_low_sample: bool = False,
                extra_caveat: str | None = None) -> None:
    """업종/가게 하나에 대한 점수 카드: 제목 + 배지 + (선택)SHAP 근거 + 신뢰도 문구."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        grade_badge(survival_score, percentile)
        if is_low_sample:
            low_sample_badge()
        if extra_caveat:
            st.caption(extra_caveat)
        if shap_lines:
            for line in shap_lines[:3]:
                st.caption(f"· {line}")
        confidence_notice()


def login_cta_banner() -> None:
    """GUEST 우측 패널 상단 로그인 유도 배너."""
    st.info("로그인하면 내 가게 맞춤 건강검진과 업종전환 추천을 볼 수 있어요.")


def onboarding_banner() -> None:
    """NEW_MEMBER 우측 패널 상단 온보딩 배너."""
    st.info("가게를 등록하면 내 가게 기준 맞춤 분석을 받을 수 있어요.")