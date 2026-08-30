# -*- coding: utf-8 -*-
"""Gemini API 호출 헬퍼. 챗봇 UI(app.py의 모달)에서 사용.

2026-08-30: google-generativeai(구 SDK)가 공식적으로 지원 종료됐다("All support
for the `google.generativeai` package has ended")는 걸 실제로 확인해서
google-genai(신규 SDK)로 교체. 발표 직전에 구 SDK가 갑자기 깨지는 상황을
피하기 위한 선제 조치 — requirements.txt도 google-genai로 갱신 필요.
"""
import os

from google import genai
from google.genai import errors, types

_MODEL_NAME = "gemini-flash-lite-latest"  # 무료 티어

_SYSTEM_INSTRUCTION = (
    "너는 '서울 상권 폐업예측' 서비스의 도우미야. "
    "주어진 참고 데이터에 있는 내용만 근거로 답하고, 없는 사실은 지어내지 마. "
    "소상공인이 이해하기 쉽게 짧고 친절하게 답해줘."
)


def _get_client() -> "genai.Client | None":
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _context_to_text(context: dict | None) -> str:
    """None 값은 "아직 이 정보가 없음"이라는 뜻일 뿐인데 그대로 프롬프트에
    넣으면 토큰 낭비고, 가끔 모델이 "None점이에요"처럼 값을 그대로 읽어버리는
    사고가 나서 None인 항목은 제외하고 넣는다."""
    if not context:
        return ""
    filtered = {k: v for k, v in context.items() if v is not None}
    if not filtered:
        return ""
    lines = "\n".join(f"- {k}: {v}" for k, v in filtered.items())
    return f"\n\n[참고 데이터]\n{lines}"


def _history_to_contents(history: list[tuple[str, str]]) -> list[types.Content]:
    """st.session_state['chatbot_history']((role, msg) 튜플 리스트, role은
    'user'/'assistant')를 genai Content 리스트로 변환. Gemini 쪽 role 이름은
    'model'이라 'assistant' -> 'model'로 바꿔줘야 한다."""
    contents = []
    for role, msg in history:
        genai_role = "model" if role == "assistant" else "user"
        contents.append(types.Content(role=genai_role, parts=[types.Part(text=msg)]))
    return contents


def ask_chatbot(
    user_message: str,
    context: dict | None = None,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """context에 현재 화면 정보(선택된 동/가게 등)를 넣어주면 그 정보 기준으로만
    답하도록 시스템 지침에 고정 — 환각 방지를 위해 자유 대화가 아니라 주어진 데이터
    설명에 집중시킨다.

    2026-08-30 추가: history(이전 대화 turn들, (role, msg) 튜플 리스트)를 받아서
    같이 전달한다. 예전에는 매 호출마다 user_message 한 줄만 보내서 "거기
    유동인구는?" 같은 후속 질문이 "거기"가 뭘 가리키는지 전혀 모른 채 처리됐다
    (화면상으로는 대화가 이어져 보이지만 실제로는 매번 새 대화였음)."""
    client = _get_client()
    if client is None:
        return "챗봇 설정이 아직 안 돼 있어요(GEMINI_API_KEY 확인 필요)."

    contents = _history_to_contents(history or [])
    user_text = user_message + _context_to_text(context)
    contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    try:
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=_SYSTEM_INSTRUCTION),
        )
        return response.text
    except errors.APIError as e:
        # APIError.code는 실제 HTTP 상태 코드 정수라 문자열 매칭보다 안전하다
        # (예전 "429" in str(e) 방식은 에러 메시지 안에 우연히 "429"가 들어간
        # 다른 상황과 헷갈릴 여지가 있었음).
        if e.code == 429:
            return "지금 요청이 몰려서 잠시 답변이 어려워요. 몇 초 뒤에 다시 시도해주세요."
        return f"답변 생성 중 오류가 발생했어요: {e}"
    except Exception as e:
        return f"답변 생성 중 오류가 발생했어요: {e}"