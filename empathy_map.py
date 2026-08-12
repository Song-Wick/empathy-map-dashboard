import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import io
import json
import re
from google import genai
from kiwipiepy import Kiwi

# Enable Matplotlib in non-GUI background
plt.switch_backend('Agg')

# Set page config
st.set_page_config(
    page_title="설문조사 2단계 입체 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set Korean Font for Matplotlib (to avoid broken labels in downloaded PNGs)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ========== [Settings & Consts] ==========
MAX_RESPONSES = 300
MAX_SURVEY_CHARS = 24000

# Default Korean stopwords list for network analysis
STOPWORDS = {
    '것', '수', '등', '및', '데', '이', '그', '저', '적', '때', '바', '의', '에', '을', '를', '은', '는', '이', '가', 
    '저희', '우리', '너희', '당신', '대해', '대해서', '통해', '통해서', '위해', '위해서', '때문', '때문에', '정도', '이후',
    '관련', '경우', '대한', '대해서', '통한', '분석', '의견', '답변', '설문', '결과', '내용', '부분', '사항', '사람', '이용',
    '사용', '생각', '의사', '작성', '도출', '단계', '설정', '기준', '분포', '비율', '빈도', '만족', '만족도', '프로그램',
    '이번', '통해', '통하여', '통해서', '매우', '진짜', '진짜로', '정말', '정말로', '가장', '제일', '매우', '아주', '조금', '약간',
    '그것', '이것', '저것', '에서', '부터', '까지', '으로', '로써', '로서', '하고', '그리고', '하지만', '그런데', '그래서'
}

# Cached Kiwi morphological analyzer
@st.cache_resource
def get_kiwi():
    return Kiwi()

def extract_nouns_kiwi(text: str, exclude_single_char: bool = True, custom_stopwords: set[str] = set()) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    kiwi = get_kiwi()
    nouns = []
    try:
        tokens = kiwi.tokenize(text)
        for t in tokens:
            if t.tag in ('NNG', 'NNP'):  # General and Proper Nouns
                noun = t.form.strip()
                if exclude_single_char and len(noun) <= 1:
                    continue
                if noun in STOPWORDS or noun in custom_stopwords:
                    continue
                nouns.append(noun)
    except Exception:
        pass
    return nouns


HMW_OPPORTUNITY_GUIDE = """
[문제의 기회 요소 발견 7요소 기반 HMW 가이드]
HMW를 도출할 때는 문제를 곧바로 해결책으로 바꾸지 말고, Pain Point와 문제 재정의 안에서
아이디어가 확장될 수 있는 '기회 요소'를 먼저 찾은 뒤 질문으로 전환하세요.

1. 긍정적 요소 강화: 나쁜 것을 없애는 데서 멈추지 말고, 이미 좋게 작동하는 경험이나 기대감을 극대화하는 질문
2. 부정적 요소 제거: 사용자가 겪는 고통, 마찰, 불안, 혼란을 줄이거나 아예 필요 없게 만드는 질문
3. 반대로 뒤집기: 문제의 상황을 반대로 바라보며 사용자가 찾아오게 하거나 경험의 주도권을 바꾸는 질문
4. 가정에 의문 제기하기: 당연하다고 믿는 절차, 장소, 순서, 역할, 업계 관행을 다시 묻는 질문
5. 형용사 사용: 불안한 경험을 안정적인 경험으로 바꾸듯 사용자가 기억할 감정과 상태를 바꾸는 질문
6. 예상치 못한 자원 활용: 팀, 현장, 데이터, 기록, 동료, 피드백처럼 가지고 있지만 쓰지 않던 자원을 활용하는 질문
7. 유추와 비유: 놀이공원, 공항, 다른 산업의 사례처럼 익숙한 경험을 끌어와 새로운 방향을 여는 질문

좋은 HMW 조건:
- 사용자 중심으로 작성하고 기술, 비용, 운영자 편의 중심으로 좁히지 마세요.
- 특정 솔루션(앱, 챗봇, 영상, 체크리스트 등)을 질문 안에 미리 넣지 마세요.
- 범위는 너무 넓거나 좁지 않게, 구체적이지만 다양한 아이디어가 나올 만큼 열어 두세요.
- 부정적 표현보다 가능성, 기대, 자신감, 즐거움, 성장을 여는 긍정 표현을 사용하세요.
- 각 HMW는 '사용자 + 의도하는 행동/경험 + 원하는 결과'가 드러나야 합니다.
- HMW 질문은 브레인스토밍과 아이디에이션을 촉발하는 방아쇠 역할을 해야 하므로 한 문장으로 짧고 직관적으로 작성하세요.
- 세부 행동지표, 조건, 절차, 예시는 질문 안에 나열하지 말고 상위 개념으로 묶어 압축하세요.
- 압축하더라도 핵심 Pain Point, 사용자 의도, 기대 결과의 의미가 훼손되지 않아야 합니다.
- 문장은 35~55자 내외를 권장하며, 길어도 70자를 넘기지 마세요.
- 최종 HMW는 최소 5개, 최대 7개로 작성하세요.
- 가능한 한 7요소를 고르게 검토하되, 설문 데이터에서 근거가 약한 요소는 제외해도 됩니다.
- 각 HMW는 서로 다른 기회 요소를 적용해 중복되지 않게 작성하세요.
"""

DEMOGRAPHIC_KEYWORDS = (
    "성별", "연령", "나이", "학년", "학과", "전공", "직무", "직군", "부서", "소속",
    "지역", "거주", "경력", "직업", "학교", "학부", "학위", "신분", "직급", "구분",
    "gender", "age", "grade", "major", "department", "region", "career", "job", 
    "role", "status", "class"
)
LIKERT_KEYWORDS = (
    "만족", "점수", "평점", "효과", "이해", "난이도", "추천", "동의", "필요", "도움",
    "satisfaction", "score", "rating", "effect", "recommend", "agree",
)

# ========== [Custom CSS Injection] ==========
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
        
        /* Global font adjustment */
        html, body, [class*="css"] {
            font-family: 'Noto Sans KR', sans-serif;
        }
        
        /* Premium Gradient Header */
        .premium-header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: #ffffff;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            position: relative;
            overflow: hidden;
        }
        .premium-header::after {
            content: '';
            position: absolute;
            top: -50px; right: -50px;
            width: 150px; height: 150px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.03);
            pointer-events: none;
        }
        .premium-header h1 {
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            margin: 0 0 8px 0;
            letter-spacing: -0.02em;
        }
        .premium-header p {
            font-size: 14px;
            color: #94a3b8;
            margin: 0;
            font-weight: 300;
        }
        
        /* Empathy Map Styles */
        .empathy-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 15px;
        }
        .quadrant {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            min-height: 250px;
        }
        .quadrant.says { border-top: 5px solid #f59e0b; }
        .quadrant.thinks { border-top: 5px solid #3b82f6; }
        .quadrant.does { border-top: 5px solid #ef4444; }
        .quadrant.feels { border-top: 5px solid #8b5cf6; }
        
        .q-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 15px;
            font-weight: 700;
            font-size: 16px;
            color: #0f172a;
        }
        .q-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        .says .q-dot { background-color: #f59e0b; }
        .thinks .q-dot { background-color: #3b82f6; }
        .does .q-dot { background-color: #ef4444; }
        .feels .q-dot { background-color: #8b5cf6; }
        
        .quadrant ul {
            padding-left: 15px;
            margin: 0;
        }
        .quadrant ul li {
            font-size: 13.5px;
            color: #334155;
            line-height: 1.6;
            margin-bottom: 8px;
        }
        
        .feel-badge {
            display: inline-block;
            font-size: 11px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 4px;
            margin-right: 5px;
        }
        .feel-pos { background: #ecfdf5; color: #065f46; }
        .feel-neg { background: #fef2f2; color: #991b1b; }
        .feel-neu { background: #f1f5f9; color: #475569; }
        
        /* KPI Cards */
        .kpi-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 15px 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
            text-align: center;
        }
        .kpi-value {
            font-size: 24px;
            font-weight: 700;
            color: #1e293b;
        }
        .kpi-label {
            font-size: 12px;
            color: #64748b;
            margin-top: 4px;
        }
        
        /* Guide Box Styling */
        .guide-box {
            background-color: #f8fafc;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            border-radius: 4px 8px 8px 4px;
            margin-bottom: 20px;
            font-size: 13.5px;
            color: #334155;
            line-height: 1.6;
        }
        .guide-title {
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        </style>
    """, unsafe_allow_html=True)

# ========== [Helper Functions] ==========
def get_gemini_client():
    """Get genai.Client using key from secrets or session state."""
    # Prioritize secrets for security and to prevent overriding by users if already set
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.session_state.get("gemini_api_key", "")
    if api_key.strip():
        return genai.Client(api_key=api_key.strip())
    return None

def is_likely_text_response(series: pd.Series) -> bool:
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return False
    avg_len = values.str.len().mean()
    unique_ratio = values.nunique() / len(values)
    return avg_len >= 25 and unique_ratio >= 0.45

def infer_subjective_columns(df: pd.DataFrame) -> list[str]:
    subjective_columns = []
    for column in df.columns:
        if is_likely_text_response(df[column]):
            subjective_columns.append(column)
    return subjective_columns[:50]

def infer_categorical_columns(df: pd.DataFrame, exclude_columns: list[str]) -> list[str]:
    categorical_columns = []
    excluded = set(exclude_columns)
    for column in df.columns:
        if column in excluded:
            continue
        series = df[column].dropna()
        if series.empty or is_likely_text_response(series):
            continue
        unique_count = series.astype(str).str.strip().nunique()
        unique_ratio = unique_count / len(series)
        is_object_like = not pd.api.types.is_numeric_dtype(series)
        is_low_cardinality_number = pd.api.types.is_numeric_dtype(series) and unique_count <= 12

        if is_object_like or is_low_cardinality_number or unique_ratio <= 0.35:
            categorical_columns.append(column)
    return categorical_columns[:50]

def infer_numeric_columns(df: pd.DataFrame, exclude_columns: list[str]) -> list[str]:
    numeric_columns = []
    excluded = set(exclude_columns)
    for column in df.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        unique_count = numeric.nunique()
        unique_ratio = unique_count / len(numeric)
        column_name = str(column).lower()
        looks_like_id = unique_ratio > 0.9 and not any(keyword in column_name for keyword in LIKERT_KEYWORDS)
        if not looks_like_id:
            numeric_columns.append(column)
    return numeric_columns[:50]

def infer_demographic_columns(categorical_columns: list[str], custom_keywords: list[str] = []) -> list[str]:
    keywords = list(DEMOGRAPHIC_KEYWORDS) + [k.strip() for k in custom_keywords if k.strip()]
    return [
        col for col in categorical_columns
        if any(keyword.lower() in str(col).lower() for keyword in keywords)
    ][:6]

def parse_json_from_response(text: str) -> dict:
    """Bulletproof JSON extraction from Gemini's response text."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try finding JSON block via regex
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raise ValueError("AI 응답을 JSON 형식으로 파싱할 수 없습니다. 응답 내용: \n" + text)

def build_survey_text_summary(df: pd.DataFrame, columns: list[str]) -> str:
    blocks = []
    for column in columns:
        responses = [
            val.strip()
            for val in df[column].dropna().astype(str)
            if val.strip()
        ]
        if not responses:
            continue
        selected = responses[:MAX_RESPONSES]
        block = [f"[문항: {column}]"]
        block.extend(f"- {resp}" for resp in selected)
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)[:MAX_SURVEY_CHARS]

def build_quantitative_summary(
    df: pd.DataFrame,
    selected_demographic_columns: list[str],
    selected_objective_columns: list[str],
    selected_numeric_columns: list[str],
) -> str:
    lines = [
        f"전체 응답자 수: {len(df)}명",
        ""
    ]
    if selected_demographic_columns:
        lines.append("[인구통계 구성 비율]")
        for col in selected_demographic_columns:
            counts = df[col].dropna().value_counts().head(5)
            total = len(df[col].dropna())
            summary = "; ".join([f"{k}: {v}명({v/total*100:.1f}%)" for k, v in counts.items()])
            lines.append(f"- {col}: {summary}")
            
    if selected_objective_columns:
        lines.append("\n[객관식 설문 문항 응답 비율]")
        for col in selected_objective_columns:
            counts = df[col].dropna().value_counts().head(5)
            total = len(df[col].dropna())
            summary = "; ".join([f"{k}: {v}명({v/total*100:.1f}%)" for k, v in counts.items()])
            lines.append(f"- {col}: {summary}")

    if selected_numeric_columns:
        lines.append("\n[척도 만족도 평균 지표]")
        for col in selected_numeric_columns:
            numeric = pd.to_numeric(df[col], errors="coerce").dropna()
            if not numeric.empty:
                lines.append(f"- {col}: 평균 {numeric.mean():.2f}점 (중앙값 {numeric.median():.2f}점)")
    return "\n".join(lines)

# ========== [App Header] ==========
inject_custom_css()
st.markdown("""
    <div class="premium-header">
        <h1>📊 설문조사 2단계 입체 분석 대시보드</h1>
        <p>1단계 객관식 통계(정량) 분석을 마친 후, 2단계 주관식 텍스트(정성) 분석을 진행하여 데이터를 입체적으로 시각화합니다.</p>
    </div>
""", unsafe_allow_html=True)

# ========== [Sidebar Configuration] ==========
# 1. API key configuration block
st.sidebar.markdown("### 🔑 API 설정")
if st.secrets.get("GEMINI_API_KEY", ""):
    st.sidebar.success("✅ Gemini API Key 자동 적용됨 (보안 유지 중)")
else:
    gemini_key_input = st.sidebar.text_input(
        "Gemini API Key 입력", 
        type="password", 
        value=st.session_state.get("gemini_api_key", ""),
        help="Gemini API Key를 입력하면 공감맵, 네트워크, HMW 분석이 활성화됩니다."
    )
    if gemini_key_input:
        st.session_state["gemini_api_key"] = gemini_key_input

st.sidebar.markdown("---")

# 2. Stage Navigation
stage = st.sidebar.radio(
    "🔍 분석 단계 선택",
    ["Stage 1: 객관식 데이터 분석 (정량)", "Stage 2: 주관식 데이터 분석 (정성)"]
)

# 3. File Upload & Column Selector Logic based on active Stage
if stage == "Stage 1: 객관식 데이터 분석 (정량)":
    st.sidebar.markdown("### 📂 [Stage 1] 객관식 파일 업로드")
    uploaded_file_obj = st.sidebar.file_uploader(
        "객관식 결과 파일 업로드 (CSV/Excel)", 
        type=['csv', 'xlsx'],
        help="성별, 만족도 점수 등 정량 데이터가 들어있는 파일을 업로드해 주세요."
    )
    
    if uploaded_file_obj is not None:
        try:
            if uploaded_file_obj.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file_obj)
            else:
                df_raw = pd.read_excel(uploaded_file_obj)
            
            # Save to session state
            if "raw_df_obj" not in st.session_state or st.session_state.get("uploaded_filename_obj") != uploaded_file_obj.name:
                st.session_state.raw_df_obj = df_raw.copy()
                st.session_state.df_obj = df_raw.copy()
                st.session_state.uploaded_filename_obj = uploaded_file_obj.name
        except Exception as e:
            st.error(f"파일을 읽는 도중 오류가 발생했습니다: {e}")
            st.stop()
            
    if "df_obj" in st.session_state:
        df_obj = st.session_state.df_obj
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 정량 컬럼 분석 구성")
        
        with st.sidebar.expander("❓ 각 설정 항목의 차이점 안내"):
            st.markdown("""
                **1. 인구통계학 답변**
                * **개념**: 응답자의 사회·인구학적 배경을 나타내는 답변
                * **예시**: 성별, 연령, 거주지, 직업, 소득 등
                * **분석**: 집단 간 차이 비교 (교차 분석 기준값)
                
                **2. 객관식 / 선택형 답변**
                * **개념**: 제시된 보기 중 선택하는 답변 (범주형)
                * **예시**: "남자/여성", "20대/30대" 등 명목/순위 데이터
                * **분석**: 빈도 분석, 비율(%) 집계 (Step 2 분석)
                
                **3. 수치형 답변 (5점척도 등)**
                * **개념**: 측정 가능한 순수한 숫자로 작성하는 답변 (수량형)
                * **예시**: 만족도 점수(1~5점), 만 28세, 금액 등
                * **분석**: 평균, 표준편차, 중앙값 분석 (Step 3 분석)
                
                ---
                💡 *동일한 5점 척도 만족도 문항이라도 평균값과 세부 선택비율을 모두 분석하고 싶다면 양쪽 상자에 모두 추가하셔도 좋습니다!*
            """)
            
        # Categorical and Numeric column inferences
        st.sidebar.markdown("##### 🔍 컬럼 감지 설정")
        custom_demo_input = st.sidebar.text_input(
            "인구통계 감지 키워드 추가 (쉼표 구분)", 
            value="신분, 직급, 구분",
            help="여기에 쉼표로 구분해 입력한 키워드가 컬럼명에 포함되어 있으면 인구통계학 열로 자동 분류(체크)됩니다."
        )
        custom_keywords = [k.strip() for k in custom_demo_input.split(",") if k.strip()]
        
        detected_cat = infer_categorical_columns(df_obj, [])
        detected_dem = infer_demographic_columns(detected_cat, custom_keywords)
        detected_num = infer_numeric_columns(df_obj, [])
        
        st.sidebar.markdown("##### 📋 컬럼 선택")
        st.sidebar.caption("💡 각 선택상자 안의 빨간 항목 우측 'x'를 클릭하면 삭제되고, 빈 공간이나 우측 'v'를 클릭하면 다른 열을 선택해 추가할 수 있습니다.")
        
        all_columns = list(df_obj.columns)
        
        dem_cols = st.sidebar.multiselect(
            "인구통계학(성별, 연령 등) 열 선택",
            options=all_columns,
            default=[c for c in detected_dem if c in all_columns],
            help="기본으로 자동 감지되지 않는 열은 마우스로 박스를 클릭해 직접 추가할 수 있습니다."
        )
        
        select_all_obj = st.sidebar.checkbox("객관식 모든 열 일괄 선택", value=False, help="이 상자를 클릭하면 업로드된 파일의 모든 열을 객관식 분석 대상으로 지정합니다.")
        obj_default = all_columns if select_all_obj else [c for c in detected_cat if c in all_columns and c not in dem_cols]
        obj_cols = st.sidebar.multiselect(
            "객관식/선택형 열 선택",
            options=all_columns,
            default=obj_default
        )
        
        select_all_num = st.sidebar.checkbox("수치형 모든 열 일괄 선택", value=False, help="이 상자를 클릭하면 업로드된 파일의 모든 열을 기술통계 분석 대상으로 지정합니다.")
        num_default = all_columns if select_all_num else [c for c in detected_num if c in all_columns]
        num_cols = st.sidebar.multiselect(
            "수치형/5점척도 열 선택",
            options=all_columns,
            default=num_default
        )
        
        # Keep quantitative summary updated in session state
        st.session_state.quant_summary = build_quantitative_summary(df_obj, dem_cols, obj_cols, num_cols)

elif stage == "Stage 2: 주관식 데이터 분석 (정성)":
    st.sidebar.markdown("### 📂 [Stage 2] 주관식 파일 업로드")
    uploaded_file_sub = st.sidebar.file_uploader(
        "주관식 결과 파일 업로드 (CSV/Excel)", 
        type=['csv', 'xlsx'],
        help="사용자 의견 등 텍스트 주관식 데이터가 들어있는 파일을 업로드해 주세요."
    )
    
    if uploaded_file_sub is not None:
        try:
            if uploaded_file_sub.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file_sub)
            else:
                df_raw = pd.read_excel(uploaded_file_sub)
            
            # Save to session state
            if "raw_df_sub" not in st.session_state or st.session_state.get("uploaded_filename_sub") != uploaded_file_sub.name:
                st.session_state.raw_df_sub = df_raw.copy()
                st.session_state.df_sub = df_raw.copy()
                st.session_state.uploaded_filename_sub = uploaded_file_sub.name
        except Exception as e:
            st.error(f"파일을 읽는 도중 오류가 발생했습니다: {e}")
            st.stop()
            
    if "df_sub" in st.session_state:
        df_sub = st.session_state.df_sub
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ✍️ 주관식 컬럼 분석 구성")
        st.sidebar.caption("💡 각 선택상자 안의 빨간 항목 우측 'x'를 클릭하면 삭제되고, 빈 공간이나 우측 'v'를 클릭하면 다른 열을 선택해 추가할 수 있습니다.")
        
        all_sub_columns = list(df_sub.columns)
        detected_sub = infer_subjective_columns(df_sub)
        
        select_all_sub = st.sidebar.checkbox("주관식 모든 열 일괄 선택", value=False, help="이 상자를 클릭하면 업로드된 파일의 모든 열을 주관식 분석 대상으로 지정합니다.")
        sub_default = all_sub_columns if select_all_sub else [c for c in detected_sub if c in all_sub_columns]
        sub_cols = st.sidebar.multiselect(
            "주관식 서술형 열 선택",
            options=all_sub_columns,
            default=sub_default
        )
        
        if len(sub_cols) >= 2:
            st.sidebar.markdown("##### 🔀 네트워크 분석 모드")
            net_mode = st.sidebar.radio(
                "분석 모드 선택",
                ["단일/통합 분석", "비교 분석"],
                index=0,
                help="단일/통합 분석은 선택한 주관식 열들을 통합하여 하나의 네트워크로 그립니다. 비교 분석은 두 문항의 차이를 나란히 비교합니다."
            )
            if net_mode == "비교 분석":
                compare_col_A = st.sidebar.selectbox(
                    "기준 문항 A 선택",
                    options=sub_cols,
                    index=0
                )
                compare_col_B = st.sidebar.selectbox(
                    "비교 문항 B 선택",
                    options=sub_cols,
                    index=min(1, len(sub_cols)-1)
                )
                st.session_state.network_mode = "비교 분석"
                st.session_state.compare_col_A = compare_col_A
                st.session_state.compare_col_B = compare_col_B
            else:
                st.session_state.network_mode = "단일/통합 분석"
        else:
            st.session_state.network_mode = "단일/통합 분석"



def run_semantic_network_analysis(df: pd.DataFrame, columns: list[str], exclude_single_char: bool, custom_stopwords: set[str], client) -> dict:
    # Extract nouns for all responses using Kiwi
    raw_responses = []
    for col in columns:
        raw_responses.extend(df[col].dropna().astype(str).str.strip().tolist())
    
    # Kiwi tokenize responses
    extracted_docs = []
    all_nouns = {}
    for resp in raw_responses:
        doc_nouns = extract_nouns_kiwi(resp, exclude_single_char, custom_stopwords)
        if doc_nouns:
            extracted_docs.append(doc_nouns)
            for n in doc_nouns:
                all_nouns[n] = all_nouns.get(n, 0) + 1
                
    if not all_nouns:
        return {"error": "추출된 유효 명사가 없습니다. 불용어 설정을 확인해 주세요."}
        
    # Sort nouns by frequency
    sorted_nouns = sorted(all_nouns.items(), key=lambda x: x[1], reverse=True)
    # Take top 40 nouns to send to Gemini for grouping/standardizing
    top_nouns = sorted_nouns[:40]
    nouns_freq_list_str = "\n".join([f"- {n}: {freq}회" for n, freq in top_nouns])
    
    prompt = f"""
    설문조사 주관식 응답 데이터에서 추출된 주요 명사 리스트와 각 명사의 빈도입니다.
    이 명사들을 분석하여, 설문 피드백을 대표하는 핵심 개념 키워드(표준 키워드)를 10~15개 도출하고, 
    유사한 단어나 어형 변화어(동의어, 유의어 등)를 표준 키워드에 매핑해 주세요.
    
    [추출된 명사 빈도 리스트]
    {nouns_freq_list_str}
    
    [작성 규칙]
    - 'standard'는 명사 리스트를 종합하는 가장 대표적인 핵심 단어(명사 또는 짧은 개념어구)로 정해주십시오.
    - 'alternatives'는 명사 리스트 중 'standard'와 의미가 같거나 매우 유사하여 통합하여 처리해야 할 유사 단어들의 리스트입니다.
    - 대표 키워드는 최대 15개까지만 도출해 주십시오.

    [출력 형식]
    반드시 다음 JSON 형식으로만 응답해 주세요. 다른 마크다운 펜스나 부연 설명은 제외해 주세요.
    {{
      "mappings": [
        {{
          "standard": "자료 부족",
          "alternatives": ["콘텐츠", "자료", "부족", "정보"]
        }},
        {{
          "standard": "시스템 오류",
          "alternatives": ["오류", "에러", "작동", "서버"]
        }}
      ]
    }}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={'temperature': 0.0, 'seed': 42}
        )
        mappings_data = parse_json_from_response(response.text)
    except Exception as e:
        return {"error": f"Gemini API 호출 및 응답 파싱 중 오류 발생: {e}"}
        
    mappings = mappings_data.get("mappings", [])
    if not mappings:
        return {"error": "Gemini로부터 유효한 키워드 매핑 테이블을 생성하지 못했습니다."}
        
    # Standard keywords
    keywords = [m.get("standard") for m in mappings if m.get("standard")]
    # Map from alternative/standard to standardized term
    synonym_map = {}
    for m in mappings:
        std = m.get("standard")
        if not std:
            continue
        synonym_map[std.lower().replace(" ", "")] = std
        for alt in m.get("alternatives", []):
            synonym_map[alt.lower().replace(" ", "")] = std
            
    # Calculate response-level co-occurrence matrix and frequencies
    doc_resolved_keywords = []
    freq_dict = {k: 0 for k in keywords}
    
    for doc in extracted_docs:
        matched_standards = set()
        for noun in doc:
            clean_noun = noun.lower().replace(" ", "")
            if clean_noun in synonym_map:
                matched_standards.add(synonym_map[clean_noun])
        if matched_standards:
            doc_resolved_keywords.append(list(matched_standards))
            for k in matched_standards:
                freq_dict[k] += 1
                
    # Safeguard frequencies
    for k in keywords:
        freq_dict[k] = max(freq_dict.get(k, 0), 1)
        
    # Co-occurrence count matrix
    co_matrix = pd.DataFrame(0, index=keywords, columns=keywords)
    for doc_kws in doc_resolved_keywords:
        for i in range(len(doc_kws)):
            for j in range(i + 1, len(doc_kws)):
                k1 = doc_kws[i]
                k2 = doc_kws[j]
                if k1 in co_matrix.index and k2 in co_matrix.columns:
                    co_matrix.loc[k1, k2] += 1
                    co_matrix.loc[k2, k1] += 1
                    
    # Build NetworkX Graph
    G = nx.Graph()
    for k in keywords:
        G.add_node(k, size=freq_dict.get(k, 1))
        
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            w = co_matrix.iloc[i, j]
            if w > 0:
                G.add_edge(keywords[i], keywords[j], weight=w)
                
    # Calculate Centralities
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    closeness_cent = nx.closeness_centrality(G)
    
    df_cent = pd.DataFrame({
        '키워드': keywords,
        '연결정도 중심성': [round(degree_cent.get(k, 0.0), 3) for k in keywords],
        '매개 중심성': [round(betweenness_cent.get(k, 0.0), 3) for k in keywords],
        '근접 중심성': [round(closeness_cent.get(k, 0.0), 3) for k in keywords]
    })
    
    # Calculate Modularity Communities
    communities_list = []
    if len(G.edges) > 0:
        try:
            communities_list = list(greedy_modularity_communities(G))
        except Exception:
            pass
            
    # Assign community indices
    node_community_dict = {}
    if communities_list:
        for comm_idx, comm in enumerate(communities_list):
            for node in comm:
                node_community_dict[node] = comm_idx
    else:
        for node in G.nodes:
            node_community_dict[node] = 0
            
    return {
        "keywords": keywords,
        "co_matrix": co_matrix,
        "freq_dict": freq_dict,
        "df_cent": df_cent,
        "communities": node_community_dict,
        "G": G
    }

def render_network_analysis_results(res: dict, label: str, key_suffix: str = ""):
    if "error" in res:
        st.error(res["error"])
        return
        
    keywords = res["keywords"]
    co_matrix = res["co_matrix"]
    freq_dict = res["freq_dict"]
    df_cent = res["df_cent"]
    node_community_dict = res["communities"]
    G = res["G"]
    
    has_edges = (co_matrix.values.sum() > 0)
    if not has_edges:
        pos = {node: np.array([np.cos(2*np.pi*i/len(keywords)), np.sin(2*np.pi*i/len(keywords))]) for i, node in enumerate(keywords)}
    else:
        pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)
        
    # Draw Plotly Figure
    fig_net = go.Figure()
    
    # Draw Edges
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        w = edge[2].get('weight', 1)
        width = min(1 + w * 0.8, 6.0)
        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=width, color='rgba(148, 163, 184, 0.3)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        fig_net.add_trace(edge_trace)
        
    # Draw Nodes grouped by Modularity Communities
    unique_comms = sorted(list(set(node_community_dict.values())))
    colors_palette = px.colors.qualitative.Safe
    
    for c_idx in unique_comms:
        c_nodes = [node for node, c in node_community_dict.items() if c == c_idx]
        if not c_nodes:
            continue
            
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in c_nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"<b>{node}</b><br>출현빈도: {freq_dict.get(node, 0)}회<br>연결도: {G.degree(node)}")
            node_size.append(min(15 + freq_dict.get(node, 1) * 2, 45))
            
        sorted_c_nodes = sorted(c_nodes, key=lambda x: freq_dict.get(x, 0), reverse=True)
        repr_kws = ", ".join(sorted_c_nodes[:3])
        trace_name = f"군집 {c_idx+1} ({repr_kws})"
        node_color = colors_palette[c_idx % len(colors_palette)]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            name=trace_name,
            text=c_nodes,
            textposition="top center",
            hovertext=node_text,
            textfont=dict(size=11, color='#0f172a', family="Noto Sans KR"),
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line=dict(width=1.5, color='#ffffff')
            )
        )
        fig_net.add_trace(node_trace)
        
    fig_net.update_layout(
        title=f"어휘 의미망 네트워크 ({label})",
        titlefont_size=14,
        showlegend=True,
        legend=dict(
            title=dict(text="어휘 군집 및 대표 키워드"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(248, 250, 252, 0.4)',
        paper_bgcolor='white',
        height=550
    )
    
    col_n1, col_n2 = st.columns([2, 1])
    with col_n1:
        st.markdown(f"##### 의미망 네트워크 연관 관계 지도 - {label}")
        st.plotly_chart(fig_net, use_container_width=True, key=f"plotly_chart_{key_suffix}")
        st.caption("💡 노드를 클릭하고 잡아당기거나, 마우스 휠로 확대 및 스크롤할 수 있습니다.")
        
        st.markdown("##### 🔑 키워드 중심성 지표 분석")
        df_cent_sorted = df_cent.sort_values(by="연결정도 중심성", ascending=False)
        
        def highlight_top5(row):
            if row.name in df_cent_sorted.index[:5]:
                return ['background-color: rgba(254, 240, 138, 0.5); font-weight: bold'] * len(row)
            return [''] * len(row)
            
        st.dataframe(df_cent_sorted.style.apply(highlight_top5, axis=1), use_container_width=True, hide_index=True)
        st.caption("💡 연결정도 중심성이 높은 상위 5개 핵심 키워드가 노란색으로 강조되어 표시됩니다.")
        
    with col_n2:
        st.markdown("##### 키워드 동시출현 원형 빈도수 매트릭스")
        st.dataframe(co_matrix, use_container_width=True)
        
        st.markdown("🤖 **복사 및 다운로드**")
        
        html_buffer = io.StringIO()
        fig_net.write_html(html_buffer, include_plotlyjs='cdn')
        html_bytes = html_buffer.getvalue().encode('utf-8')
        
        st.download_button(
            label="🌐 인터랙티브 그래프 HTML 다운로드",
            data=html_bytes,
            file_name=f"keyword_network_{key_suffix}.html",
            mime="text/html",
            key=f"dl_html_{key_suffix}"
        )
        
        try:
            fig_plt, ax = plt.subplots(figsize=(6, 5))
            node_colors_plt = [colors_palette[node_community_dict[n] % len(colors_palette)] for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[freq_dict.get(n, 1)*50 + 200 for n in G.nodes()], node_color=node_colors_plt)
            nx.draw_networkx_edges(G, pos, ax=ax, width=[G[u][v]['weight']*0.8 for u,v in G.edges()], edge_color='gray', alpha=0.5)
            nx.draw_networkx_labels(G, pos, ax=ax, font_family='Malgun Gothic', font_size=10, font_weight='bold')
            ax.axis('off')
            plt.tight_layout()
            
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', dpi=150)
            img_buf.seek(0)
            plt.close()
            
            st.download_button(
                label="🖼️ 네트워크 그래프 PNG 다운로드",
                data=img_buf,
                file_name=f"keyword_network_{key_suffix}.png",
                mime="image/png",
                key=f"dl_png_{key_suffix}"
            )
        except Exception as e:
            st.caption(f"PNG 이미지 생성 대기 지연: {e}")
            
        csv_matrix = co_matrix.to_csv(index=True).encode('utf-8-sig')
        st.download_button(
            label="📥 매트릭스 데이터 CSV 다운로드",
            data=csv_matrix,
            file_name=f"co_occurrence_matrix_{key_suffix}.csv",
            mime="text/csv",
            key=f"dl_matrix_csv_{key_suffix}"
        )
        
        csv_cent = df_cent.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 중심성 지표 CSV 다운로드",
            data=csv_cent,
            file_name=f"keyword_centrality_{key_suffix}.csv",
            mime="text/csv",
            key=f"dl_cent_csv_{key_suffix}"
        )


# ========== [Main Dashboard Render] ==========
client = get_gemini_client()

if stage == "Stage 1: 객관식 데이터 분석 (정량)":
    # ------------------ [Stage 1 Render] ------------------
    if "df_obj" not in st.session_state:
        # Welcome screen for Stage 1
        st.markdown("""
            ### 👋 [Stage 1] 객관식 데이터 분석에 오신 것을 환영합니다!
            
            이 단계에서는 설문조사의 **정량적인 통계 정보**를 분석합니다. 
            성별이나 연령대와 같은 **인구통계 정보**, 그리고 만족도 평점과 같은 **숫자형 척도 데이터**를 집계하고 시각화할 수 있습니다.
            
            #### 💡 시작하는 방법:
            1. ① 왼쪽 사이드바에서 **객관식 결과 파일(CSV 또는 Excel)**을 업로드해 주세요.
            2. ② 파일 업로드 후, 데이터 결측치를 정제하고 인구통계 분석, 기술 통계 분석, 문항별 빈도분석, 교차분석 탭을 통해 다각도로 분석을 진행해 보세요.
        """)
    else:
        df_obj = st.session_state.df_obj
        
        # Tabs for Steps 1~5
        tabs_obj = st.tabs(["Step 1: 데이터 전처리", "Step 2: 인구통계 분석", "Step 3: 기술 통계 분석", "Step 4: 문항별 빈도분석", "Step 5: 교차분석"])
        
        # --- Step 1: Preprocessing ---
        with tabs_obj[0]:
            st.subheader("🛠️ 객관식 데이터 전처리 및 확인")
            
            st.markdown("""
                <div class="guide-box">
                    <div class="guide-title">💡 Step 1. 데이터 전처리 가이드</div>
                    업로드한 객관식 설문 원본 데이터에서 빈 데이터(NaN)를 정제하고, 문자열 값의 불필요한 앞뒤 공백을 제거하여 정확한 통계 집계가 가능하도록 데이터를 깨끗이 정제하는 단계입니다.
                </div>
            """, unsafe_allow_html=True)
            
            col_prep_1, col_prep_2 = st.columns([1, 3])
            with col_prep_1:
                st.markdown("##### 전처리 설정")
                do_preprocess = st.radio("데이터 전처리를 수행하시겠습니까?", ["아니오 (기본값 사용)", "예 (결측치 제거 및 공백 정제)"], index=0)
                
                if do_preprocess == "예 (결측치 제거 및 공백 정제)":
                    if st.button("전처리 실행하기"):
                        cleaned_df = st.session_state.raw_df_obj.copy()
                        cleaned_df = cleaned_df.dropna(how='all')
                        for col in cleaned_df.columns:
                            if cleaned_df[col].dtype == "object":
                                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                                cleaned_df[col] = cleaned_df[col].replace(r'^\s*$', np.nan, regex=True)
                        st.session_state.df_obj = cleaned_df
                        st.success("객관식 데이터 전처리 완료!")
                        st.rerun()
                else:
                    if st.button("원본 데이터로 복원"):
                        st.session_state.df_obj = st.session_state.raw_df_obj.copy()
                        st.info("원본 객관식 데이터로 복원되었습니다.")
                        st.rerun()
                        
            with col_prep_2:
                st.markdown("##### 데이터 구조 요약")
                shape_col_1, shape_col_2, shape_col_3 = st.columns(3)
                with shape_col_1:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(st.session_state.raw_df_obj)}</div><div class="kpi-label">원본 행(Rows) 수</div></div>', unsafe_allow_html=True)
                with shape_col_2:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df_obj)}</div><div class="kpi-label">현재 정제된 행 수</div></div>', unsafe_allow_html=True)
                with shape_col_3:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df_obj.columns)}</div><div class="kpi-label">열(Columns) 수</div></div>', unsafe_allow_html=True)
                    
            st.markdown("##### 객관식 데이터셋 미리보기 (상위 5개 행)")
            st.dataframe(df_obj.head(5), use_container_width=True)
            
        # --- Step 2: Demographic Analysis ---
        with tabs_obj[1]:
            st.subheader("👥 인구통계 분석")
            
            st.markdown("""
                <div class="guide-box">
                    <div class="guide-title">💡 Step 2. 인구통계 분석 가이드</div>
                    성별, 연령대, 직급, 직무 등 설문 응답자의 사회·인구학적 배경 분포(빈도수 및 비율)를 파악하는 단계입니다.
                </div>
            """, unsafe_allow_html=True)
            
            if not dem_cols:
                st.info("사이드바에서 분석할 '인구통계학 열'을 1개 이상 선택해 주세요.")
            else:
                selected_dem_col = st.selectbox("분석할 인구통계 컬럼 선택", dem_cols)
                series = df_obj[selected_dem_col].dropna().astype(str).str.strip()
                
                if series.empty:
                    st.warning("선택한 컬럼에 분석 가능한 유효 데이터가 없습니다.")
                else:
                    total_valid = len(series)
                    counts = series.value_counts()
                    df_freq = pd.DataFrame({
                        selected_dem_col: counts.index,
                        '빈도': counts.values,
                        '비율(%)': (counts.values / total_valid * 100).round(1)
                    })
                    
                    col_d_1, col_d_2 = st.columns([1, 1])
                    with col_d_1:
                        st.markdown(f"**{selected_dem_col} 빈도 분포표**")
                        st.dataframe(df_freq, use_container_width=True, hide_index=True)
                        
                        st.markdown("🤖 **복사 및 다운로드**")
                        tsv_data = df_freq.to_csv(sep='\t', index=False)
                        st.code(tsv_data, language='text')
                        st.caption("텍스트 상자 우측 복사 아이콘으로 스프레드시트에 쉽게 붙여넣으세요.")
                        
                        csv_data = df_freq.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 CSV 다운로드",
                            data=csv_data,
                            file_name=f"demographic_{selected_dem_col}.csv",
                            mime="text/csv"
                        )
                        
                    with col_d_2:
                        st.markdown("##### 시각화 차트")
                        chart_type = st.radio("차트 타입 선택", ["원형 차트 (Pie Chart)", "막대 차트 (Bar Chart)"], key="dem_chart")
                        
                        if chart_type == "원형 차트 (Pie Chart)":
                            fig = px.pie(df_freq, names=selected_dem_col, values='빈도', 
                                         title=f"{selected_dem_col} 분포 비율",
                                         color_discrete_sequence=px.colors.qualitative.Safe)
                        else:
                            fig = px.bar(df_freq, x=selected_dem_col, y='빈도', 
                                         title=f"{selected_dem_col} 빈도수",
                                         color=selected_dem_col,
                                         color_discrete_sequence=px.colors.qualitative.Safe)
                        fig.update_layout(font=dict(family="Noto Sans KR"))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        try:
                            plt.figure(figsize=(6, 4))
                            if chart_type == "원형 차트 (Pie Chart)":
                                plt.pie(df_freq['빈도'], labels=df_freq[selected_dem_col], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
                                plt.title(f"{selected_dem_col} 분포 비율")
                            else:
                                plt.bar(df_freq[selected_dem_col], df_freq['빈도'], color='#3b82f6')
                                plt.ylabel("빈도 (명)")
                                plt.title(f"{selected_dem_col} 빈도수")
                                plt.xticks(rotation=45)
                            plt.tight_layout()
                            img_buf = io.BytesIO()
                            plt.savefig(img_buf, format='png', dpi=150)
                            img_buf.seek(0)
                            plt.close()
                            
                            st.download_button(
                                label="🖼️ 차트 PNG 이미지 다운로드",
                                data=img_buf,
                                file_name=f"chart_{selected_dem_col}.png",
                                mime="image/png"
                            )
                        except Exception as e:
                            st.caption(f"PNG 이미지 생성 중 일시적 오류: {e}")
                            
        # --- Step 3: Descriptive Analysis ---
        with tabs_obj[2]:
            st.subheader("📈 수치형 및 척도 만족도 기술 통계")
            
            st.markdown("""
                <div class="guide-box">
                    <div class="guide-title">💡 Step 3. 기술 통계 분석 가이드</div>
                    프로그램 만족도 점수, 혹은 수치로 표기된 설문 응답의 대표값들을 구합니다. 
                    평균, 표준편차, 중앙값, 최솟값/최댓값을 구하여 흩어짐 정도를 확인하고 여러 문항의 평균값을 막대 그래프로 시각적으로 대비해 봅니다.
                </div>
            """, unsafe_allow_html=True)
            
            if not num_cols:
                st.info("사이드바에서 분석할 '수치형/5점척도 열'을 1개 이상 선택해 주세요.")
            else:
                desc_rows = []
                for col in num_cols:
                    numeric = pd.to_numeric(df_obj[col], errors="coerce").dropna()
                    if not numeric.empty:
                        desc_rows.append({
                            '문항': col,
                            '응답 수': len(numeric),
                            '평균': round(numeric.mean(), 2),
                            '표준편차': round(numeric.std(), 2) if len(numeric) > 1 else 0.0,
                            '중앙값': round(numeric.median(), 2),
                            '최솟값': round(numeric.min(), 2),
                            '최댓값': round(numeric.max(), 2)
                        })
                
                if not desc_rows:
                    st.warning("선택된 열들에서 유효한 수치 데이터를 찾지 못했습니다.")
                else:
                    df_desc = pd.DataFrame(desc_rows)
                    st.dataframe(df_desc, use_container_width=True, hide_index=True)
                    
                    st.markdown("🤖 **복사 및 다운로드**")
                    tsv_desc = df_desc.to_csv(sep='\t', index=False)
                    st.code(tsv_desc, language='text')
                    st.caption("위 텍스트 박스 우측 아이콘으로 클립보드에 바로 복사해 가세요.")
                    
                    csv_desc = df_desc.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 CSV 다운로드",
                        data=csv_desc,
                        file_name="descriptive_statistics.csv",
                        mime="text/csv"
                    )
                    
                    st.markdown("##### 주요 평점 평균값 비교")
                    fig_mean = px.bar(df_desc, x='문항', y='평균', text='평균',
                                      title="척도 만족도 문항 평균 비교",
                                      color='평균', color_continuous_scale=px.colors.sequential.Teal)
                    fig_mean.update_traces(textposition='outside')
                    fig_mean.update_layout(font=dict(family="Noto Sans KR"))
                    st.plotly_chart(fig_mean, use_container_width=True)
                    
                    # Notify user that data is saved for Stage 2 HMW
                    st.success("✅ [안내] 이 단계의 만족도 평균값과 인구통계 분포 요약이 자동으로 보존되어 Stage 2의 HMW 분석에 반영됩니다.")

        # --- Step 4: Item Frequency & Sub-factor Analysis ---
        with tabs_obj[3]:
            st.subheader("📊 문항별 빈도 및 하위 요인 분석")
            
            step3_sub_mode = st.radio(
                "분석 세부 유형 선택",
                ["문항별 빈도분석", "하위 요인별 분석"],
                horizontal=True,
                help="개별 문항에 대한 비율을 집계하는 '빈도분석'과, 유사한 여러 문항들을 하나로 묶어 연관 분포를 보는 '하위 요인별 분석' 중 선택합니다."
            )
            
            st.markdown("---")
            
            if step3_sub_mode == "문항별 빈도분석":
                st.markdown("""
                    <div class="guide-box">
                        <div class="guide-title">💡 문항별 빈도분석 가이드</div>
                        설문조사의 개별 객관식/선택형 문항들을 대상으로, 각 선택지 보기별 응답 빈도수와 비율(%)을 집계하고 시각화하는 단계입니다.
                    </div>
                """, unsafe_allow_html=True)
                
                if not obj_cols:
                    st.info("사이드바에서 분석할 '객관식/선택형 열'을 1개 이상 선택해 주세요.")
                else:
                    selected_obj_col = st.selectbox("분석할 객관식 컬럼 선택", obj_cols)
                    series = df_obj[selected_obj_col].dropna().astype(str).str.strip()
                    
                    if series.empty:
                        st.warning("선택한 컬럼에 분석 가능한 유효 데이터가 없습니다.")
                    else:
                        total_valid = len(series)
                        counts = series.value_counts()
                        df_freq = pd.DataFrame({
                            selected_obj_col: counts.index,
                            '빈도': counts.values,
                            '비율(%)': (counts.values / total_valid * 100).round(1)
                        })
                        
                        col_o_1, col_o_2 = st.columns([1, 1])
                        with col_o_1:
                            st.markdown(f"**{selected_obj_col} 빈도 분포표**")
                            st.dataframe(df_freq, use_container_width=True, hide_index=True)
                            
                            st.markdown("🤖 **복사 및 다운로드**")
                            tsv_data = df_freq.to_csv(sep='\t', index=False)
                            st.code(tsv_data, language='text')
                            st.caption("텍스트 상자 우측 복사 아이콘으로 스프레드시트에 쉽게 붙여넣으세요.")
                            
                            csv_data = df_freq.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                label="📥 CSV 다운로드",
                                data=csv_data,
                                file_name=f"frequency_{selected_obj_col}.csv",
                                mime="text/csv",
                                key=f"dl_csv_obj_{selected_obj_col}"
                            )
                            
                        with col_o_2:
                            st.markdown("##### 시각화 차트")
                            chart_type = st.radio("차트 타입 선택", ["원형 차트 (Pie Chart)", "막대 차트 (Bar Chart)"], key="obj_chart")
                            
                            if chart_type == "원형 차트 (Pie Chart)":
                                fig = px.pie(df_freq, names=selected_obj_col, values='빈도', 
                                             title=f"{selected_obj_col} 분포 비율",
                                             color_discrete_sequence=px.colors.qualitative.Safe)
                            else:
                                fig = px.bar(df_freq, x=selected_obj_col, y='빈도', 
                                             title=f"{selected_obj_col} 빈도수",
                                             color=selected_obj_col,
                                             color_discrete_sequence=px.colors.qualitative.Safe)
                            fig.update_layout(font=dict(family="Noto Sans KR"))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            try:
                                plt.figure(figsize=(6, 4))
                                if chart_type == "원형 차트 (Pie Chart)":
                                    plt.pie(df_freq['빈도'], labels=df_freq[selected_obj_col], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
                                    plt.title(f"{selected_obj_col} 분포 비율")
                                else:
                                    plt.bar(df_freq[selected_obj_col], df_freq['빈도'], color='#3b82f6')
                                    plt.ylabel("빈도 (명)")
                                    plt.title(f"{selected_obj_col} 빈도수")
                                    plt.xticks(rotation=45)
                                plt.tight_layout()
                                img_buf = io.BytesIO()
                                plt.savefig(img_buf, format='png', dpi=150)
                                img_buf.seek(0)
                                plt.close()
                                
                                st.download_button(
                                    label="🖼️ 차트 PNG 이미지 다운로드",
                                    data=img_buf,
                                    file_name=f"chart_{selected_obj_col}.png",
                                    mime="image/png",
                                    key=f"dl_img_obj_{selected_obj_col}"
                                )
                            except Exception as e:
                                st.caption(f"PNG 이미지 생성 중 일시적 오류: {e}")
                                
            elif step3_sub_mode == "하위 요인별 분석":
                st.markdown("""
                    <div class="guide-box">
                        <div class="guide-title">💡 하위 요인별 분석 가이드</div>
                        성격이 유사한 여러 만족도 문항(예: 서비스 친절도 관련 3개 문항, 시설 환경 관련 4개 문항 등)을 하나의 <b>하위 요인(그룹)</b>으로 묶어서 통합 평균과 비율 분포를 비교 및 집계하는 단계입니다.
                    </div>
                """, unsafe_allow_html=True)
                
                num_factors = st.number_input("생성할 하위 요인(그룹) 개수", min_value=1, max_value=8, value=2, step=1)
                
                st.markdown("##### 👥 하위 요인 그룹 설정")
                group_data = []
                for idx in range(int(num_factors)):
                    col_f1, col_f2 = st.columns([1, 2])
                    with col_f1:
                        g_name = st.text_input(f"요인 {idx+1} 이름", value=f"요인 {idx+1}", key=f"g_name_{idx}")
                    with col_f2:
                        g_cols = st.multiselect(
                            f"요인 {idx+1}에 포함할 문항 선택",
                            options=list(df_obj.columns),
                            default=[],
                            key=f"g_cols_{idx}"
                        )
                    if g_cols:
                        group_data.append((g_name, g_cols))
                    st.markdown("---")
                
                if not group_data:
                    st.info("각 그룹에 최소 1개 이상의 설문 문항을 선택해 추가해 주세요.")
                else:
                    st.markdown("### 📊 하위 요인별 분석 결과")
                    
                    # 1. Composite descriptive statistics
                    factor_stats = []
                    for g_name, g_cols in group_data:
                        # Combine numerical values from the group
                        combined_series = pd.concat([pd.to_numeric(df_obj[col], errors="coerce") for col in g_cols]).dropna()
                        if not combined_series.empty:
                            factor_stats.append({
                                "하위 요인 (그룹)": g_name,
                                "포함 문항 수": len(g_cols),
                                "통합 응답 수": len(combined_series),
                                "통합 평균 만족도": round(combined_series.mean(), 2),
                                "통합 표준편차": round(combined_series.std(), 2),
                                "통합 중앙값": round(combined_series.median(), 2)
                            })
                    
                    if factor_stats:
                        df_factors = pd.DataFrame(factor_stats)
                        st.markdown("##### 📊 요인별 통합 기술통계 요약표")
                        st.dataframe(df_factors, use_container_width=True, hide_index=True)
                        
                        csv_factors = df_factors.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 요인별 요약표 CSV 다운로드",
                            data=csv_factors,
                            file_name="sub_factor_summary.csv",
                            mime="text/csv",
                            key="dl_factor_summary"
                        )
                        
                        # Plotly factor comparison chart
                        fig_factor = px.bar(df_factors, x="하위 요인 (그룹)", y="통합 평균 만족도", text="통합 평균 만족도",
                                           title="하위 요인(그룹)별 통합 평균 점수 비교",
                                           color="통합 평균 만족도", color_continuous_scale=px.colors.sequential.Sunset)
                        fig_factor.update_traces(textposition='outside')
                        fig_factor.update_layout(font=dict(family="Noto Sans KR"))
                        st.plotly_chart(fig_factor, use_container_width=True)
                    
                    # 2. Combined categorical distributions
                    st.markdown("##### 📈 요인별 통합 응답 비율 분포")
                    for g_name, g_cols in group_data:
                        # Combine text responses from the group
                        combined_text = pd.concat([df_obj[col].dropna().astype(str).str.strip() for col in g_cols])
                        if not combined_text.empty:
                            total = len(combined_text)
                            counts = combined_text.value_counts()
                            df_g_freq = pd.DataFrame({
                                "응답 보기": counts.index,
                                "빈도": counts.values,
                                "비율(%)": (counts.values / total * 100).round(1)
                            })
                            
                            with st.expander(f"📢 [{g_name}] 통합 응답 상세 분포 보기 (총 {total}건)"):
                                col_gf1, col_gf2 = st.columns([1, 1])
                                with col_gf1:
                                    st.dataframe(df_g_freq, use_container_width=True, hide_index=True)
                                with col_gf2:
                                    fig_gf_pie = px.pie(df_g_freq, names="응답 보기", values="빈도",
                                                         title=f"[{g_name}] 통합 보기 선택 비율",
                                                         color_discrete_sequence=px.colors.qualitative.Safe)
                                    fig_gf_pie.update_layout(font=dict(family="Noto Sans KR"))
                                    st.plotly_chart(fig_gf_pie, use_container_width=True)

        # --- Step 5: Cross-tabulation Analysis ---
        with tabs_obj[4]:
            st.subheader("🔀 인구통계 기준 교차분석 (Crosstab)")
            
            st.markdown("""
                <div class="guide-box">
                    <div class="guide-title">💡 Step 5. 교차분석 가이드</div>
                    성별, 연령대, 직무 등의 <b>인구통계 변수(기준 행)</b>와 <b>객관식/선택형 문항(대상 열)</b>을 교차시켜 집단별 응답 분포의 차이를 확인하는 단계입니다. 
                    특정 인구통계적 집단에 따른 상세 응답 빈도와 백분율(%) 교차표를 제공하며, 누적 백분율 막대그래프를 통해 시각적으로 대비할 수 있습니다.
                </div>
            """, unsafe_allow_html=True)
            
            if not dem_cols:
                st.info("사이드바에서 분석의 기준이 될 '인구통계학 열'을 1개 이상 선택해 주세요.")
            elif not obj_cols and not num_cols:
                st.info("사이드바에서 교차 분석을 수행할 '객관식/선택형 열' 또는 '수치형/5점척도 열'을 1개 이상 선택해 주세요.")
            else:
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    row_var = st.selectbox("기준 인구통계 변수 선택 (행)", dem_cols)
                with col_c2:
                    col_var = st.selectbox("교차 대상 문항 선택 (열)", list(dict.fromkeys(list(obj_cols) + list(num_cols))))
                
                # Check data
                series_row = df_obj[row_var].dropna().astype(str).str.strip()
                series_col = df_obj[col_var].dropna().astype(str).str.strip()
                
                if series_row.empty or series_col.empty:
                    st.warning("선택한 컬럼에 분석 가능한 데이터가 없습니다.")
                else:
                    # Clean aligned data
                    df_crosstab_raw = df_obj[[row_var, col_var]].dropna().copy()
                    for col in df_crosstab_raw.columns:
                        df_crosstab_raw[col] = df_crosstab_raw[col].astype(str).str.strip()
                    df_crosstab_raw = df_crosstab_raw[df_crosstab_raw[row_var] != ""]
                    df_crosstab_raw = df_crosstab_raw[df_crosstab_raw[col_var] != ""]
                    
                    if df_crosstab_raw.empty:
                        st.warning("두 열에 동시에 존재하는 데이터가 없습니다.")
                    else:
                        # 1. Frequency count crosstab
                        df_cross_count = pd.crosstab(df_crosstab_raw[row_var], df_crosstab_raw[col_var])
                        
                        # 2. Percentage crosstab (row percentage)
                        df_cross_pct = pd.crosstab(df_crosstab_raw[row_var], df_crosstab_raw[col_var], normalize='index') * 100
                        df_cross_pct = df_cross_pct.round(1)
                        
                        col_tbl1, col_tbl2 = st.columns([1, 1])
                        with col_tbl1:
                            st.markdown(f"**🔠 {row_var} × {col_var} 교차 빈도표 (명)**")
                            st.dataframe(df_cross_count, use_container_width=True)
                            
                            tsv_cross_count = df_cross_count.to_csv(sep='\t', index=True)
                            st.code(tsv_cross_count, language='text')
                            
                        with col_tbl2:
                            st.markdown(f"**📊 {row_var} × {col_var} 집단 내 응답 비율표 (%)**")
                            st.dataframe(df_cross_pct, use_container_width=True)
                            
                            tsv_cross_pct = df_cross_pct.to_csv(sep='\t', index=True)
                            st.code(tsv_cross_pct, language='text')
                            
                        # Download buttons for Crosstabs
                        st.markdown("🤖 **교차표 다운로드**")
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            csv_cross_count = df_cross_count.to_csv(index=True).encode('utf-8-sig')
                            st.download_button(
                                label="📥 교차 빈도표 CSV 다운로드",
                                data=csv_cross_count,
                                file_name=f"crosstab_count_{row_var}_{col_var}.csv",
                                mime="text/csv",
                                key="dl_cross_count"
                            )
                        with col_dl2:
                            csv_cross_pct = df_cross_pct.to_csv(index=True).encode('utf-8-sig')
                            st.download_button(
                                label="📥 교차 비율표 CSV 다운로드",
                                data=csv_cross_pct,
                                file_name=f"crosstab_pct_{row_var}_{col_var}.csv",
                                mime="text/csv",
                                key="dl_cross_pct"
                            )
                            
                        # Stacked bar chart for crosstab
                        st.markdown("##### 📈 집단별 누적 백분율 분포 시각화")
                        df_melted = df_cross_pct.reset_index().melt(id_vars=row_var, value_name="비율(%)", var_name=col_var)
                        
                        fig_cross = px.bar(
                            df_melted,
                            x=row_var,
                            y="비율(%)",
                            color=col_var,
                            text="비율(%)",
                            title=f"{row_var} 집단별 {col_var} 응답 비율 분포",
                            labels={"비율(%)": "비율 (%)"},
                            category_orders={row_var: sorted(df_melted[row_var].unique())},
                            color_discrete_sequence=px.colors.qualitative.Safe
                        )
                        fig_cross.update_layout(
                            barmode="stack",
                            font=dict(family="Noto Sans KR"),
                            yaxis_range=[0, 100]
                        )
                        fig_cross.update_traces(textposition='inside', texttemplate='%{text:.1f}%')
                        st.plotly_chart(fig_cross, use_container_width=True)
                        
                        try:
                            plt.figure(figsize=(7, 5))
                            categories = df_cross_pct.index
                            columns_data = df_cross_pct.columns
                            
                            bottom = np.zeros(len(categories))
                            colors = plt.cm.Paired.colors
                            for i, col_data_name in enumerate(columns_data):
                                plt.bar(categories, df_cross_pct[col_data_name], bottom=bottom, label=str(col_data_name), color=colors[i % len(colors)])
                                bottom += df_cross_pct[col_data_name]
                                
                            plt.ylabel("비율 (%)")
                            plt.title(f"{row_var} 집단별 {col_var} 응답 비율 분포")
                            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                            plt.tight_layout()
                            
                            img_buf = io.BytesIO()
                            plt.savefig(img_buf, format='png', dpi=150)
                            img_buf.seek(0)
                            plt.close()
                            
                            st.download_button(
                                label="🖼️ 교차 분석 차트 PNG 다운로드",
                                data=img_buf,
                                file_name=f"crosstab_chart_{row_var}_{col_var}.png",
                                mime="image/png",
                                key="dl_cross_chart_png"
                            )
                        except Exception as e:
                            st.caption(f"PNG 이미지 생성 중 일시적 오류: {e}")

elif stage == "Stage 2: 주관식 데이터 분석 (정성)":
    # ------------------ [Stage 2 Render] ------------------
    if "df_sub" not in st.session_state:
        # Welcome screen for Stage 2
        st.markdown("""
            ### 👋 [Stage 2] 주관식 데이터 분석에 오신 것을 환영합니다!
            
            이 단계에서는 설문조사의 **자유 서술형 답변(주관식)**을 분석합니다.
            Gemini AI 패널을 이용해 사용자의 심리와 Pain Point를 입체적으로 분류하는 **공감 맵(Step 6)**, 
            텍스트 키워드 의미 연결망 그래프를 시각화하는 **네트워크 분석(Step 7)**, 
            그리고 아이디에이션을 유도하는 **HMW(How Might We) 질문 도출(Step 8)**을 진행합니다.
            
            #### 💡 시작하는 방법:
            1. 왼쪽 사이드바에서 **주관식 결과 파일(CSV 또는 Excel)**을 업로드해 주세요.
            2. 파일 업로드 후, 각 분석 탭을 클릭하고 하단의 분석 실행 버튼을 누르면 AI 연동 분석이 개시됩니다.
        """)
    else:
        df_sub = st.session_state.df_sub
        
        # Tabs for Steps 6~8
        tabs_sub = st.tabs(["Step 6: 공감 맵 분석", "Step 7: 네트워크 분석", "Step 8: HMW 도출"])
        
        # --- Step 6: Empathy Map ---
        with tabs_sub[0]:
            st.subheader("💡 AI 기반 주관식 응답 공감 맵(Empathy Map)")
            
            st.markdown("""
                <div class="guide-box">
                    <div class="guide-title">💡 Step 6. 공감 맵 분석 가이드</div>
                    Gemini AI를 사용해 주관식 의견들에 포함된 고객(사용자)의 감정과 요구사항을 분류하고 요약합니다. 
                    말한 내용(Says), 내면의 생각(Thinks), 취하는 행동(Does), 복합적인 감정(Feels)의 4개 도메인으로 나뉘어 프리미엄 카드 레이아웃으로 결과를 제공합니다.
                </div>
            """, unsafe_allow_html=True)
            
            if not sub_cols:
                st.warning("사이드바에서 분석할 '주관식 서술형 열'을 1개 이상 선택해 주세요.")
            elif client is None:
                st.warning("공감 맵 분석을 활성화하려면 왼쪽 사이드바에 **Gemini API Key**를 설정해 주세요.")
            else:
                survey_text = build_survey_text_summary(df_sub, sub_cols)
                
                if st.button("공감 맵 분석 실행", key="run_empathy_map"):
                    with st.spinner("Gemini AI가 서술형 답변들을 종합하여 다각도로 감정을 맵핑하고 있습니다..."):
                        prompt = f"""
                        설문조사의 주관식 답변 데이터를 바탕으로, 대상 사용자의 심리를 분석하여 2x2 공감 맵(Empathy Map)을 작성해주세요.
                        
                        [공감 맵의 작성 규칙 및 구성]
                        1. Says (말하는 것): 사용자가 주관식 답변에서 직접 언급하거나 표현한 핵심 의견 및 언어적 피드백
                        2. Does (행동하는 것): 사용자가 시스템 이용이나 설문 대상 경험 중에 행한 구체적인 행동, 프로세스, 패턴 (사용자의 개인 성향을 넣지 말고 관찰 가능한 사실 위주로 작성)
                        3. Thinks (생각하는 것): Says와 Does를 확실한 근거(기반)로 삼아 도출한 사용자의 생각, 기대 또는 가설 (예: '지금 실수하면 돈을 날릴 수도 있겠다', '이 시스템을 전적으로 신뢰하기는 어렵다')
                        4. Feels (느끼는 것): Says와 Does를 직접적인 근거(기반)로 삼아, 사용자가 구체적으로 '어느 시점/상황에서 어떤 감정을 느끼는지' 구체적인 문장으로 도출 (예: '결제 완료 메시지가 뜰 때 불안해함', '환불/취소 조건을 미리 확인할 수 없어 답답해함')
                        
                        ★ 중요 규칙:
                        - Thinks와 Feels는 단순히 '불편함', '아쉬움', '만족' 같은 단어 나열식의 감정 명사 단어 표현을 철저히 배제해주세요.
                        - 반드시 Says와 Does의 내용에서 논리적으로 도출되는 구체적 가설/감정 문장으로 작성해야 합니다.
                        - 각 사분면별로 최소 4개씩 도출해주세요.

                        [주관식 답변 데이터]
                        {survey_text}

                        [출력 형식]
                        반드시 다음 JSON 형식으로만 응답해주세요. 다른 부연 설명이나 마크다운 코드 펜스는 제외해주세요.
                        {{
                          "says": ["의견 1", "의견 2", "의견 3", "의견 4"],
                          "thinks": ["근거가 명확한 생각/기대 1", "근거가 명확한 생각/기대 2", "근거가 명확한 생각/기대 3", "근거가 명확한 생각/기대 4"],
                          "does": ["행동/패턴 1", "행동/패턴 2", "행동/패턴 3", "행동/패턴 4"],
                          "feels": [
                            {{"text": "상황에 기반한 구체적 감정 문장 1", "type": "pos"}},
                            {{"text": "상황에 기반한 구체적 감정 문장 2", "type": "neg"}},
                            {{"text": "상황에 기반한 구체적 감정 문장 3", "type": "neu"}},
                            {{"text": "상황에 기반한 구체적 감정 문장 4", "type": "neg"}}
                          ]
                        }}
                        (참고: feels의 type은 'pos'(긍정), 'neg'(부정), 'neu'(중립) 중 하나로 지정해주세요.)
                        """
                        
                        try:
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=prompt,
                                config={'temperature': 0.0, 'seed': 42}
                            )
                            empathy_dict = parse_json_from_response(response.text)
                            st.session_state.empathy_data = empathy_dict
                            st.success("공감 맵 분석이 완료되었습니다!")
                        except Exception as e:
                            st.error(f"공감 맵 생성 실패: {e}")
                            
                # Show Empathy Map if exists in state
                if "empathy_data" in st.session_state:
                    em_data = st.session_state.empathy_data
                    
                    # 2x2 UI Render
                    col1, col2 = st.columns(2)
                    with col1:
                        says_html = "".join([f"<li>{item}</li>" for item in em_data.get("says", [])])
                        st.markdown(f"""
                            <div class="quadrant says">
                                <div class="q-header"><div class="q-dot"></div>Says (말한다)</div>
                                <ul>{says_html}</ul>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        does_html = "".join([f"<li>{item}</li>" for item in em_data.get("does", [])])
                        st.markdown(f"""
                            <div class="quadrant does">
                                <div class="q-header"><div class="q-dot"></div>Does (행동한다)</div>
                                <ul>{does_html}</ul>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        thinks_html = "".join([f"<li>{item}</li>" for item in em_data.get("thinks", [])])
                        st.markdown(f"""
                            <div class="quadrant thinks">
                                <div class="q-header"><div class="q-dot"></div>Thinks (생각한다)</div>
                                <ul>{thinks_html}</ul>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        feels_items = []
                        for f in em_data.get("feels", []):
                            f_type = f.get("type", "neu")
                            f_text = f.get("text", "")
                            if f_type == "pos":
                                badge = '<span class="feel-badge feel-pos">긍정</span>'
                            elif f_type == "neg":
                                badge = '<span class="feel-badge feel-neg">부정</span>'
                            else:
                                badge = '<span class="feel-badge feel-neu">중립</span>'
                            feels_items.append(f"<li>{badge} {f_text}</li>")
                        feels_html = "".join(feels_items)
                        
                        st.markdown(f"""
                            <div class="quadrant feels">
                                <div class="q-header"><div class="q-dot"></div>Feels (느낀다)</div>
                                <ul>{feels_html}</ul>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    # Copy and download
                    st.markdown("---")
                    st.markdown("🤖 **복사 및 다운로드**")
                    
                    raw_empathy_text = "=== [공감 맵 분석 결과] ===\n\n"
                    raw_empathy_text += "[SAYS - 말한다]\n" + "\n".join([f"- {i}" for i in em_data.get("says", [])]) + "\n\n"
                    raw_empathy_text += "[THINKS - 생각한다]\n" + "\n".join([f"- {i}" for i in em_data.get("thinks", [])]) + "\n\n"
                    raw_empathy_text += "[DOES - 행동한다]\n" + "\n".join([f"- {i}" for i in em_data.get("does", [])]) + "\n\n"
                    raw_empathy_text += "[FEELS - 느낀다]\n" + "\n".join([f"- [{i.get('type')}] {i.get('text')}" for i in em_data.get("feels", [])])
                    
                    st.code(raw_empathy_text, language='text')
                    
                    st.download_button(
                        label="📥 공감맵 데이터 JSON 다운로드",
                        data=json.dumps(em_data, ensure_ascii=False, indent=2),
                        file_name="empathy_map.json",
                        mime="application/json"
                    )
                    
        # --- Step 7: Network Analysis ---
        # --- Step 7: Network Analysis ---
        with tabs_sub[1]:
            st.subheader("🕸️ 키워드 의미망 동시출현 네트워크 분석")
            
            st.markdown("""
                <div class="guide-box">
                    <div class="guide-title">💡 Step 7. 네트워크 분석 가이드</div>
                    사용자들의 피드백 문장에서 핵심 단어를 끄집어낸 후, 이 단어들이 서로 어떤 연결고리를 맺고 있는지 시각화합니다.
                    마우스로 노드를 움직이거나 휠로 확대/축소하며 연관 깊이를 분석할 수 있으며, 키워드 연관성 가중치 테이블과 차트를 각각 저장할 수 있습니다.
                </div>
            """, unsafe_allow_html=True)
            
            # Form filters expander
            with st.expander("🛠️ 키워드 필터링 및 형태소 분석 설정"):
                col_filt1, col_filt2 = st.columns(2)
                with col_filt1:
                    exclude_single_char = st.checkbox("1글자 명사 제외", value=True, help="분석 결과에서 '것', '수' 등 1글자 노이즈 단어를 필터링합니다.")
                with col_filt2:
                    custom_stopwords_input = st.text_area("추가 불용어 입력 (쉼표로 구분)", value="", help="네트워크에서 분석하고 싶지 않은 단어가 있다면 입력하세요.")
                custom_stopwords = {s.strip() for s in custom_stopwords_input.split(",") if s.strip()}

            network_mode = st.session_state.get("network_mode", "단일/통합 분석")
            
            if not sub_cols:
                st.warning("사이드바에서 분석할 '주관식 서술형 열'을 1개 이상 선택해 주세요.")
            elif client is None:
                st.warning("네트워크 분석을 활성화하려면 왼쪽 사이드바에 **Gemini API Key**를 설정해 주세요.")
            else:
                # Trigger Button
                if network_mode == "비교 분석":
                    compare_col_A = st.session_state.get("compare_col_A")
                    compare_col_B = st.session_state.get("compare_col_B")
                    st.info(f"비교 모드 활성화: **{compare_col_A}** vs **{compare_col_B}**")
                    
                    if st.button("네트워크 비교 분석 실행", key="run_network_compare"):
                        with st.spinner("Gemini AI와 Kiwi 형태소 분석기가 두 문항의 네트워크를 독립 분석 중..."):
                            res_A = run_semantic_network_analysis(df_sub, [compare_col_A], exclude_single_char, custom_stopwords, client)
                            res_B = run_semantic_network_analysis(df_sub, [compare_col_B], exclude_single_char, custom_stopwords, client)
                            
                            st.session_state.network_result_A = res_A
                            st.session_state.network_result_B = res_B
                            st.session_state.compare_cols_analyzed = (compare_col_A, compare_col_B)
                            
                            # Backward compatible keys mapping to column A
                            if "error" not in res_A:
                                st.session_state.network_keywords = res_A["keywords"]
                                st.session_state.network_matrix = res_A["co_matrix"]
                                st.session_state.network_frequencies = res_A["freq_dict"]
                                st.session_state.network_centrality = res_A["df_cent"]
                                st.session_state.network_communities = res_A["communities"]
                            
                            st.success("비교 분석 완료!")
                            st.rerun()
                else:
                    if st.button("네트워크 분석 실행", key="run_network_single"):
                        with st.spinner("Gemini AI와 Kiwi 형태소 분석기가 주관식 답변의 네트워크를 통합 분석 중..."):
                            res_single = run_semantic_network_analysis(df_sub, sub_cols, exclude_single_char, custom_stopwords, client)
                            
                            st.session_state.network_result_single = res_single
                            st.session_state.single_cols_analyzed = sub_cols
                            
                            # Backward compatible keys
                            if "error" not in res_single:
                                st.session_state.network_keywords = res_single["keywords"]
                                st.session_state.network_matrix = res_single["co_matrix"]
                                st.session_state.network_frequencies = res_single["freq_dict"]
                                st.session_state.network_centrality = res_single["df_cent"]
                                st.session_state.network_communities = res_single["communities"]
                                
                            st.success("통합 분석 완료!")
                            st.rerun()

                # Render Results
                if network_mode == "비교 분석":
                    if "network_result_A" in st.session_state and "network_result_B" in st.session_state:
                        compare_layout = st.radio("비교 시각화 레이아웃 선택", ["나란히 보기 (2단 컬럼)", "탭으로 보기"], horizontal=True)
                        res_A = st.session_state.network_result_A
                        res_B = st.session_state.network_result_B
                        compare_col_A = st.session_state.get("compare_col_A")
                        compare_col_B = st.session_state.get("compare_col_B")
                        
                        if compare_layout == "나란히 보기 (2단 컬럼)":
                            col_left, col_right = st.columns(2)
                            with col_left:
                                st.markdown(f"### 🅰️ {compare_col_A}")
                                render_network_analysis_results(res_A, compare_col_A, "comp_A")
                            with col_right:
                                st.markdown(f"### 🅱️ {compare_col_B}")
                                render_network_analysis_results(res_B, compare_col_B, "comp_B")
                        else:
                            comp_tabs = st.tabs([f"🅰️ {compare_col_A}", f"🅱️ {compare_col_B}"])
                            with comp_tabs[0]:
                                render_network_analysis_results(res_A, compare_col_A, "comp_A_tab")
                            with comp_tabs[1]:
                                render_network_analysis_results(res_B, compare_col_B, "comp_B_tab")
                else:
                    if "network_result_single" in st.session_state:
                        res_single = st.session_state.network_result_single
                        render_network_analysis_results(res_single, "통합 분석", "single")


        # --- Step 7: HMW 도출 ---
        with tabs_sub[2]:
            st.subheader("💡 AI 기반 HMW (How Might We) 기회 및 질문 도출")
            
            st.markdown("""
                <div class="guide-box">
                    <div class="guide-title">💡 Step 8. HMW 도출 가이드</div>
                    불편 요소나 기회 요소로부터 '우리가 어떻게 하면(How Might We)...?' 형태로 발상 전환용 질문을 뽑아내는 창의적 문제 재정의 기법입니다.
                    만약 Stage 1에서 인구통계 및 만족도 척도 요약을 집계하셨다면, 정량적 만족 지표를 기반으로 한층 정합성 있는 입체적 컨텍스트가 Gemini AI에 전달됩니다.
                </div>
            """, unsafe_allow_html=True)
            
            # Show cross-talk status
            quant_summary_status = st.session_state.get("quant_summary", "")
            if quant_summary_status:
                st.success("📊 **Stage 1 정량 분석 결과 연계 완료**: 이전 단계에서 생성한 정량 기술 통계 및 만족도 지표를 HMW 프롬프트에 자동으로 결합하여 분석합니다.")
                with st.expander("연동된 정량 분석 요약 데이터 보기"):
                    st.text(quant_summary_status)
            else:
                st.info("ℹ️ **안내**: 이전 Stage 1(정량 분석) 단계가 수행되지 않았습니다. 현재 주관식 서술형 피드백 데이터만을 바탕으로 HMW 도출을 처리합니다.")
                quant_summary_status = "정량 정보가 생략되었습니다 (주관식 단독 분석)."
                
            if not sub_cols:
                st.warning("사이드바에서 분석할 '주관식 서술형 열'을 1개 이상 선택해 주세요.")
            elif client is None:
                st.warning("HMW 질문 생성을 활성화하려면 왼쪽 사이드바에 **Gemini API Key**를 설정해 주세요.")
            else:
                if st.button("HMW 질문 생성", key="run_hmw"):
                    with st.spinner("Gemini AI가 정량 정보와 주관식 Pain Point를 조합하여 기회 7요소를 매핑하는 중..."):
                        survey_text = build_survey_text_summary(df_sub, sub_cols)
                        
                        prompt = f"""
                        설문조사의 정량 통계 요약 및 주관식 답변을 분석하여 사용자의 핵심 Pain Point와 인사이트를 도출하고,
                        [문제의 기회 요소 발견 7요소 기반 HMW 가이드]를 참고하여 창의적이고 해결 가능한 HMW(How Might We) 질문을 5~7개 생성해 주세요.

                        [Stage 1 정량/인구통계 분석 요약]
                        {quant_summary_status}

                        [Stage 2 주관식 답변 데이터]
                        {survey_text}

                        {HMW_OPPORTUNITY_GUIDE}

                        [출력 형식]
                        반드시 다음 JSON 형식으로만 응답해 주세요. 다른 마크다운 펜스나 부연 설명은 제외해 주세요.
                        {{
                          "hmw_list": [
                            {{
                              "opportunity": "긍정적 요소 강화",
                              "direction": "기존에 좋았던 협업 경험을 극대화하기",
                              "question": "우리가 어떻게 하면 참가자들이 협업 과정에서 느낀 소속감을 장기적 네트워킹으로 확장할 수 있을까?",
                              "icon": "💡"
                            }},
                            ...
                          ]
                        }}
                        """
                        
                        try:
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=prompt,
                                config={'temperature': 0.0, 'seed': 42}
                            )
                            hmw_dict = parse_json_from_response(response.text)
                            st.session_state.hmw_data = hmw_dict
                            st.success("HMW 분석 및 카드 배치 완료!")
                        except Exception as e:
                            st.error(f"HMW 질문 도출 중 에러: {e}")
                            
                # Render HMW questions if exists
                if "hmw_data" in st.session_state:
                    hmw_dict = st.session_state.hmw_data
                    
                    st.markdown("##### 기회 정의 7요소 기반 HMW 질문 카드")
                    for item in hmw_dict.get("hmw_list", []):
                        icon = item.get("icon", "💡")
                        opp = item.get("opportunity", "")
                        direc = item.get("direction", "")
                        q = item.get("question", "")
                        
                        st.markdown(f"""
                            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #10b981; padding: 20px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <span style="font-size: 24px;">{icon}</span>
                                    <div>
                                        <span style="font-size: 12px; color: #64748b; font-weight: bold; text-transform: uppercase;">{opp} · {direc}</span>
                                        <p style="font-size: 15px; font-weight: 700; color: #065f46; margin: 4px 0 0 0; line-height: 1.6;">{q}</p>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("---")
                    st.markdown("🤖 **복사 및 다운로드**")
                    
                    raw_hmw_text = "=== [HMW 질문 리스트] ===\n\n"
                    for idx, item in enumerate(hmw_dict.get("hmw_list", [])):
                        raw_hmw_text += f"{idx+1}. [{item.get('opportunity')}] {item.get('direction')}\n"
                        raw_hmw_text += f"   Q: {item.get('question')}\n\n"
                        
                    st.code(raw_hmw_text, language='text')
                    
                    st.download_button(
                        label="📥 HMW 질문 목록 TXT 다운로드",
                        data=raw_hmw_text,
                        file_name="how_might_we_questions.txt",
                        mime="text/plain"
                    )
