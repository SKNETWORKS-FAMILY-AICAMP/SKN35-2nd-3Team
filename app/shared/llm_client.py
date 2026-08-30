# -*- coding: utf-8 -*-
"""Gemini API 호출 헬퍼. 챗봇 UI(app.py의 모달)에서 사용."""
import os

import google.generativeai as genai

_MODEL_NAME = "gemini-1.5-flash"  # 무료 티어


def _get_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(_MODEL_NAME)


def ask_chatbot(user_message: str, context: dict | None = None) -> str:
    """context에 현재 화면 정보(선택된 동/가게 등)를 넣어주면 그 정보 기준으로만
    답하도록 프롬프트에 고정 — 환각 방지를 위해 자유 대화가 아니라 주어진 데이터
    설명에 집중시킨다."""
    model = _get_model()
    if model is None:
        return "챗봇 설정이 아직 안 돼 있어요(GEMINI_API_KEY 확인 필요)."

    context_str = ""
    if context:
        context_str = f"\n\n[참고 데이터]\n{context}"

    prompt = (
        "너는 '서울 상권 폐업예측' 서비스의 도우미야. "
        "아래 참고 데이터에 있는 내용만 근거로 답하고, 없는 사실은 지어내지 마. "
        "소상공인이 이해하기 쉽게 짧고 친절하게 답해줘."
        f"{context_str}\n\n사용자 질문: {user_message}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"답변 생성 중 오류가 발생했어요: {e}"