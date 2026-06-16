import streamlit as st
import pandas as pd
from google import genai
import streamlit.components.v1 as components
import re

# ========== [Settings] ==========
MAX_RESPONSES = 300
MAX_SURVEY_CHARS = 12000
ALLOWED_HTML_TAGS = ("<!DOCTYPE html", "<html", "<head", "<body")
# =================================


def get_gemini_api_key():
    """Check the API key only when generation starts."""
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key or not api_key.strip():
        st.error("GEMINI_API_KEY가 Streamlit secrets에 설정되어 있지 않습니다.")
        st.stop()
    return api_key.strip()


def build_survey_text(series: pd.Series) -> tuple[str, int, int, bool]:
    responses = [value.strip() for value in series.dropna().astype(str) if value.strip()]
    original_count = len(responses)
    selected = responses[:MAX_RESPONSES]
    survey_text = "\n- ".join(selected)
    was_truncated = original_count > len(selected) or len(survey_text) > MAX_SURVEY_CHARS
    return survey_text[:MAX_SURVEY_CHARS], original_count, len(selected), was_truncated


def strip_markdown_code_fence(text: str) -> str:
    cleaned = text.strip()
    fenced = re.match(r"^```(?:html)?\s*(.*?)\s*```$", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return fenced.group(1).strip() if fenced else cleaned


def sanitize_generated_html(html: str) -> str:
    html = strip_markdown_code_fence(html)
    if not html.lstrip().lower().startswith(tuple(tag.lower() for tag in ALLOWED_HTML_TAGS)):
        raise ValueError("Gemini 응답이 HTML 문서 형식이 아닙니다.")

    # Remove executable scripts and risky inline handlers before rendering in Streamlit.
    html = re.sub(r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"\s+on\w+\s*=\s*(['\"]).*?\1", "", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"\s+href\s*=\s*(['\"])\s*javascript:.*?\1", ' href="#"', html, flags=re.IGNORECASE | re.DOTALL)
    return html
# 페이지 설정
st.set_page_config(page_title="설문조사 공감맵 생성기", layout="wide")
st.title("📊 설문조사 공감맵 & HMW 대시보드 생성기")

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
        if st.button("대시보드 생성하기 (Pain Point & HMW 포함)"):
            with st.spinner('AI가 데이터를 분석하여 6단계 대시보드를 생성 중입니다. 다소 시간이 걸릴 수 있습니다...'):
                clean_api_key = get_gemini_api_key()
                
                # 데이터 전처리
                survey_text, original_count, selected_count, was_truncated = build_survey_text(df[target_column])
                if not survey_text:
                    st.warning("선택한 컬럼에 분석할 주관식 답변이 없습니다.")
                    st.stop()
                if was_truncated:
                    st.info(f"응답이 많아 {original_count}개 중 {selected_count}개, 최대 {MAX_SURVEY_CHARS:,}자까지만 분석합니다.")
                
                # 최신 구글 API 클라이언트 연결
                client = genai.Client(api_key=clean_api_key)
                
                # HTML/CSS 가이드라인이 명시된 고도화 프롬프트
                prompt = f"""
                다음은 프로그램 참가자들의 주관식 설문조사 응답입니다.
                이 데이터를 철저히 분석하여 아래 6가지 섹션을 도출하고, 제공된 HTML 템플릿의 내용(Text)을 분석 결과로 교체하여 완성해 주세요.
                
                1. 이슈 구조화 (Issue Structuring): 사용자, 요구사항, 목표, 문제점, 행동 추출
                2. 공감 맵 (Empathy Map): Says, Thinks, Does, Feels 4가지 영역 분석
                3. 네트워킹 분석 (Networking Analysis): 핵심 키워드 간의 관계를 서술형으로 분석
                4. Pain Point 식별 (Pain Point Identification): 상호작용 레벨(지원방식 등 물리적 마찰), 사용자 여정 레벨(기획/제작 과정의 어려움), 장기적 관계 레벨(동기부여, 성취감 저하)로 3분류
                5. 문제 재정의 (Problem Redefinition): 핵심 사용자(User)가 어떤 니즈(Needs to)를 가지고 있는지, 그리고 그 근본적인 이유/인사이트(Because)가 무엇인지 문장 형태로 도출
                6. HMW (How Might We) 도출: 재정의된 문제를 바탕으로, 긍정적이고 창의적인 해결책을 촉발할 수 있는 '우리가 어떻게 하면 ~할 수 있을까?' 질문 3가지 작성
                
                [출력 규칙]
