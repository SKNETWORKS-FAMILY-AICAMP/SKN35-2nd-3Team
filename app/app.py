"""
진입점 (Streamlit)

역할: 첫 화면에서 사용자 유형(예비창업자 / 기존점주 / 관리자)을 선택받아
      해당 로그인 흐름으로 라우팅한다.

실행:
    streamlit run app/app.py

실제 화면 로직은 pages/ 아래 각 파일에 있고,
이 파일은 진입/라우팅 역할만 담당한다 (Streamlit 멀티페이지 규칙:
pages/ 폴더의 파일들이 자동으로 사이드바 메뉴에 노출됨).

TODO:
- 사용자 유형 선택 UI
- st.session_state 기반 로그인 상태 관리
- 유형별 pages/ 로 리다이렉트
"""
