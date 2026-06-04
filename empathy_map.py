import streamlit as st
import pandas as pd
from google import genai
import streamlit.components.v1 as components
import re

# ========== [고정 API 키 입력란] ==========
MY_API_KEY = st.secrets["GEMINI_API_KEY"]
# ==========================================

# 페이지 설정
st.set_page_config(page_title="공감맵 자동 생성기", layout="wide")
st.title("📊 설문조사 공감맵 자동 생성기")

# [수정] 생성된 HTML 내용을 유지하기 위한 세션 상태 초기화
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
            if MY_API_KEY == "여기에_발급받은_API_키를_붙여넣으세요" or not MY_API_KEY.strip():
                st.error("코드 8번째 줄(MY_API_KEY)에 API 키가 입력되지 않았습니다. 코드를 수정하고 저장해주세요.")
            else:
                with st.spinner('AI가 데이터를 분석하여 디자인 레이아웃에 맞게 대시보드를 생성 중입니다...'):
                    # API 키 정제
                    clean_api_key = MY_API_KEY.strip()
                    
                    # 데이터 전처리
                    responses = df[target_column].dropna().astype(str).tolist()
                    survey_text = "\n- ".join(responses)
                    
                    # 최신 구글 API 클라이언트 연결
                    client = genai.Client(api_key=clean_api_key)
                    
                    # HTML/CSS 가이드라인이 명시된 프롬프트
                    prompt = f"""
                    다음은 프로그램 참가자들의 주관식 설문조사 응답입니다.
                    이 데이터를 철저히 분석하여 아래 세 가지 섹션을 모두 포함한 대시보드를 도출해 주세요.
                    
                    1. 이슈 구조화 (Issue Structuring): 사용자, 요구사항, 목표, 문제점, 행동이 포함된 문장으로 서술
                    2. 공감 맵 (Empathy Map): Says, Thinks, Does, Feels 4가지 영역으로 분류하여 작성
                    3. 문제 정의 (Problem Definition): 데이터에 기반한 핵심 문제를 명확히 규명하여 서술
                    
                    최종 결과물은 아래의 HTML/CSS 스타일 가이드라인을 엄격히 준수한 완성형 웹 대시보드 코드로만 출력해 주세요. 마크다운 기호(```html)는 완전히 제외하고 순수 HTML 텍스트만 리턴해야 합니다.
                    
                    [HTML/CSS 스타일 및 구조 가이드라인]
                    - 기본 스타일: body {{ font-family: 'Malgun Gothic', sans-serif; background-color: #f8f9fa; color: #333; margin: 0; padding: 20px; }}
                    - 컨테이너: .container {{ max-width: 1100px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
                    - 타이틀: h1 {{ text-align: center; color: #2c3e50; font-size: 1.8em; border-bottom: 2px solid #e9ecef; padding-bottom: 15px; margin-bottom: 30px; }}
                    - 서브헤더: h2 {{ color: #34495e; font-size: 1.4em; margin-bottom: 15px; border-left: 4px solid #3498db; padding-left: 10px; }}
                    
                    - 섹션 1 (이슈 구조화): .issue-section {{ background: #f1f8ff; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
                      내부 문장 스타일: .issue-sentence {{ background: #ffffff; padding: 15px; border: 1px solid #dcebf7; border-radius: 6px; margin-bottom: 15px; font-size: 1.05em; line-height: 1.6; }}
                      강조 텍스트: .highlight {{ font-weight: bold; color: #2980b9; }}
                      
                    - 섹션 2 (공감 맵): .empathy-map {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; position: relative; margin-top: 20px; }}
                      중앙 유저 바: .user-center {{ grid-column: 1 / -1; background: #2c3e50; color: #fff; text-align: center; padding: 15px; border-radius: 8px; font-size: 1.2em; font-weight: bold; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                      4분면 공통: .quadrant {{ padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-top: 5px solid; }}
                      * Says: .says {{ background-color: #fffdf5; border-top-color: #f1c40f; }}
                      * Thinks: .thinks {{ background-color: #f4fcff; border-top-color: #3498db; }}
                      * Does: .does {{ background-color: #fff5f5; border-top-color: #e74c3c; }}
                      * Feels: .feels {{ background-color: #f9f6ff; border-top-color: #9b59b6; }}
                      
                    - 섹션 3 (문제 정의): .problem-section {{ background: #fff5f5; padding: 20px; border-radius: 8px; margin-top: 30px; border: 1px solid #f9d5d5; }}
                      내부 문장 스타일: .problem-sentence {{ background: #ffffff; padding: 15px; border: 1px solid #fadbd8; border-radius: 6px; margin-bottom: 15px; font-size: 1.05em; line-height: 1.6; }}
                      적색 강조: .highlight-red {{ font-weight: bold; color: #c0392b; }}
                    
                    [설문 응답 데이터]
                    - {survey_text}
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                    )
                    
                    # [수정] 결과물을 세션 상태에 저장합니다.
                    st.session_state.html_content = response.text.strip().removeprefix('```html').removesuffix('```')
                    
        # [수정] 생성된 대시보드가 세션 상태에 존재하면 화면에 항상 출력하고 다운로드 버튼을 제공합니다.
        if st.session_state.html_content is not None:
            st.subheader("2. 감정 신호 분석 기반 공감 맵 대시보드")
            
            # 다운로드 버튼 추가 (HTML 파일로 다운로드)
            st.download_button(
                label="📥 공감맵 대시보드 다운로드 (HTML)",
                data=st.session_state.html_content,
                file_name="empathy_map_dashboard.html",
                mime="text/html"
            )
            
            # 화면 표시 유지
            components.html(st.session_state.html_content, height=1000, scrolling=True)
                    
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
