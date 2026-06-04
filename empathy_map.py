import streamlit as st
import pandas as pd
from google import genai
import streamlit.components.v1 as components

# ========== [고정 API 키 입력란] ==========
MY_API_KEY = st.secrets["GEMINI_API_KEY"]
# ==========================================

# 페이지 설정
st.set_page_config(page_title="공감맵 자동 생성기", layout="wide")
st.title("📊 설문조사 공감맵 자동 생성기")

# 생성된 HTML 내용을 유지하기 위한 세션 상태 초기화
if 'html_content' not in st.session_state:
    st.session_state.html_content = None

# 메인 화면: 파일 업로드
uploaded_file = st.file_uploader("설문조사 결과 파일 업로드 (CSV 또는 Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.subheader("1. 데이터 확인 및 분석 열 선택")
        st.dataframe(df.head(5))
        
        # 주관식 문항(열) 선택
        target_column = st.selectbox("분석할 주관식 답변이 포함된 열(Column)을 선택하세요.", df.columns)
        
        # 실행 버튼
        if st.button("공감맵 및 대시보드 생성하기"):
            with st.spinner('AI가 데이터를 분석하여 디자인 레이아웃에 맞게 대시보드를 생성 중입니다...'):
                clean_api_key = MY_API_KEY.strip()
                
                # 데이터 전처리
                responses = df[target_column].dropna().astype(str).tolist()
                survey_text = "\n- ".join(responses)
                
                # 최신 구글 API 클라이언트 연결
                client = genai.Client(api_key=clean_api_key)
                
                # HTML/CSS 가이드라인이 명시된 고도화 프롬프트
                prompt = f"""
                다음은 프로그램 참가자들의 주관식 설문조사 응답입니다.
                이 데이터를 철저히 분석하여 아래 4가지 섹션을 도출하고, 제공된 HTML 템플릿의 내용(Text)을 분석 결과로 교체하여 완성해 주세요.
                
                1. 이슈 구조화 (Issue Structuring): 사용자, 요구사항, 목표, 문제점, 행동 추출
                2. 공감 맵 (Empathy Map): Says, Thinks, Does, Feels 4가지 영역 분석
                3. 네트워킹 분석 (Networking Analysis): 핵심 키워드 간의 관계 분석
                4. 문제 정의 (Problem Definition): 핵심 문제 진술
                
                [출력 규칙]
                - 아래 제공된 [HTML 템플릿]의 구조(태그, 클래스명 등)와 CSS 스타일을 단 하나도 수정하거나 삭제하지 말고 100% 그대로 유지하세요.
                - 템플릿 내부의 예시 텍스트만 실제 분석 결과로 교체하세요.
                - 마크다운 기호(```html)는 완전히 제외하고 순수 HTML 텍스트만 리턴해야 합니다.
                
                [설문 응답 데이터]
                {survey_text}

                [HTML 템플릿]
                <!DOCTYPE html>
                <html lang="ko">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>프로그램 참가자 설문조사 분석 대시보드</title>
                    <link href="[https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap](https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap)" rel="stylesheet">
                    <style>
                        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

                        body {{
                            font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
                            background: #f0f2f5;
                            color: #1e2a3a;
                            padding: 36px 20px 60px;
                            font-size: 14px;
                            line-height: 1.7;
                        }}

                        .page {{
                            max-width: 1080px;
                            margin: 0 auto;
                            background: #ffffff;
                            border-radius: 16px;
                            overflow: hidden;
                            box-shadow: 0 2px 20px rgba(0,0,0,0.08);
                        }}

                        /* ── HEADER ── */
                        .header {{
                            background: #1e2a3a;
                            color: #ffffff;
                            padding: 36px 48px 32px;
                            position: relative;
                            overflow: hidden;
                        }}
                        .header::after {{
                            content: '';
                            position: absolute;
                            top: -60px; right: -60px;
                            width: 240px; height: 240px;
                            border-radius: 50%;
                            background: rgba(255,255,255,0.04);
                            pointer-events: none;
                        }}
                        .header-meta {{
                            font-size: 11px;
                            letter-spacing: 0.12em;
                            text-transform: uppercase;
                            color: rgba(255,255,255,0.45);
                            margin-bottom: 10px;
                        }}
                        .header h1 {{
                            font-size: 22px;
                            font-weight: 700;
                            letter-spacing: -0.02em;
                            color: #ffffff;
                        }}
                        .header-sub {{
                            margin-top: 6px;
                            font-size: 13px;
                            color: rgba(255,255,255,0.55);
                            font-weight: 300;
                        }}
                        .header-divider {{
                            margin: 20px 0 0;
                            display: flex;
                            gap: 24px;
                        }}
                        .header-pill {{
                            display: inline-block;
                            font-size: 11px;
                            padding: 4px 14px;
                            border-radius: 20px;
                            background: rgba(255,255,255,0.1);
                            color: rgba(255,255,255,0.7);
                            border: 0.5px solid rgba(255,255,255,0.15);
                        }}

                        /* ── CONTENT WRAPPER ── */
                        .content {{
                            padding: 40px 48px;
                        }}

                        /* ── SECTION LABEL ── */
                        .section-label {{
                            display: flex;
                            align-items: center;
                            gap: 12px;
                            margin-bottom: 20px;
                        }}
                        .section-number {{
                            width: 28px; height: 28px;
                            border-radius: 50%;
                            background: #1e2a3a;
                            color: #fff;
                            font-size: 12px;
                            font-weight: 700;
                            display: flex; align-items: center; justify-content: center;
                            flex-shrink: 0;
                        }}
                        .section-title {{
                            font-size: 15px;
                            font-weight: 700;
                            color: #1e2a3a;
                            letter-spacing: -0.01em;
                        }}
                        .section-title-en {{
                            font-size: 11px;
                            color: #8a95a3;
                            font-weight: 400;
                            letter-spacing: 0.06em;
                            text-transform: uppercase;
                            margin-left: 4px;
                        }}

                        /* ── SECTION 1: ISSUE STRUCTURING ── */
                        .issue-grid {{
                            display: grid;
                            grid-template-columns: 1fr 1fr;
                            gap: 12px;
                            margin-bottom: 40px;
                        }}
                        .issue-card {{
                            background: #f8f9fb;
                            border: 0.5px solid #e3e8ef;
                            border-radius: 10px;
                            padding: 18px 20px;
                            border-left: 3px solid #b0bac8;
                        }}
                        .issue-card.user  {{ border-left-color: #3b82f6; }}
                        .issue-card.need  {{ border-left-color: #10b981; }}
                        .issue-card.goal  {{ border-left-color: #8b5cf6; }}
                        .issue-card.prob  {{ border-left-color: #ef4444; }}
                        .issue-card.action{{ border-left-color: #f59e0b; grid-column: 1 / -1; }}

                        .issue-tag {{
                            display: inline-block;
                            font-size: 10px;
                            font-weight: 700;
                            letter-spacing: 0.08em;
                            text-transform: uppercase;
                            padding: 3px 10px;
                            border-radius: 4px;
                            margin-bottom: 10px;
                        }}
                        .tag-user   {{ background: #eff6ff; color: #1d4ed8; }}
                        .tag-need   {{ background: #ecfdf5; color: #065f46; }}
                        .tag-goal   {{ background: #f5f3ff; color: #5b21b6; }}
