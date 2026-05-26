import streamlit as st
import pandas as pd
from google import genai
import streamlit.components.v1 as components
import re

# 페이지 설정
st.set_page_config(page_title="공감맵 자동 생성기", layout="wide")
st.title("📊 학술 프로그램 설문조사 공감맵 자동 생성기")

# 사이드바: API 키 입력
with st.sidebar:
    st.header("설정")
    st.success("🔒 Gemini API 키가 시스템에 안전하게 연동되었습니다.")

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
            with st.spinner("AI가 데이터를 분석하여 이슈 구조화 및 공감맵을 생성 중입니다..."):
                try:
                    # 시스템 Secrets에 저장한 API 키를 자동으로 가져옵니다.
                    clean_api_key = st.secrets["GEMINI_API_KEY"]
                    
                    # 데이터 전처리
                    responses = df[target_column].dropna().astype(str).tolist()
                    survey_text = "\n- ".join(responses)
                    
                    # 최신 구글 API 클라이언트 연결
                    client = genai.Client(api_key=clean_api_key)
                    
                    prompt = f"""
                    다음은 프로그램 참가자들의 주관식 설문조사 응답입니다.
                    이 데이터를 분석하여 아래 두 가지를 도출해 주세요.
                    
                    1. 이슈 구조화: 사용자, 요구사항, 목표, 문제점, 행동이 포함된 문장으로 서술
                    2. 공감 맵(Empathy Map): Says, Thinks, Does, Feels 4가지 영역으로 분류하여 구체적으로 작성
                    
                    최종 결과물은 한눈에 파악할 수 있는 세련된 대시보드 형태의 HTML/CSS 코드로만 출력해 주세요. (```html 과 같은 마크다운 블록 제거 후 순수 html 코드만 출력할 것)
                    
                    [설문 응답 데이터]
                    - {survey_text}
                    """
                    
                    # 최신 문법으로 텍스트 생성 요청
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                    )
                    
                    # 결과물 정제 및 출력
                    html_content = response.text.strip().removeprefix('```html').removesuffix('```')
                    
                    st.subheader("2. 감정 신호 분석 기반 공감 맵 대시보드")
                    components.html(html_content, height=850, scrolling=True)
                    
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
