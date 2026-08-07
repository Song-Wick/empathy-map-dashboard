import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import io
import json
import re
from google import genai

# Enable Matplotlib in non-GUI background
plt.switch_backend('Agg')

# Set page config
st.set_page_config(
    page_title="설문조사 통계 및 공감맵/HMW 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set Korean Font for Matplotlib (to avoid broken labels in downloaded PNGs)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ========== [Settings & Consts] ==========
MAX_RESPONSES = 300
MAX_SURVEY_CHARS = 24000

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
    "지역", "거주", "경력", "직업", "학교", "학부", "학위", "gender", "age",
    "grade", "major", "department", "region", "career", "job", "role",
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
        </style>
    """, unsafe_allow_html=True)

# ========== [Helper Functions] ==========
def get_gemini_client():
    """Get genai.Client using key from secrets or session state."""
    api_key = st.session_state.get("gemini_api_key", "")
    if not api_key:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
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
    return subjective_columns[:5]

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
    return categorical_columns[:12]

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
    return numeric_columns[:12]

def infer_demographic_columns(categorical_columns: list[str]) -> list[str]:
    return [
        col for col in categorical_columns
        if any(keyword.lower() in str(col).lower() for keyword in DEMOGRAPHIC_KEYWORDS)
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
    subjective_columns: list[str],
    selected_demographic_columns: list[str],
    selected_objective_columns: list[str],
    selected_numeric_columns: list[str],
) -> str:
    lines = [
        f"전체 응답자 수: {len(df)}명",
        f"주관식 문항 수: {len(subjective_columns)}개",
        ""
    ]
    if selected_demographic_columns:
        lines.append("[인구통계 정보]")
        for col in selected_demographic_columns:
            counts = df[col].dropna().value_counts().head(5)
            total = len(df[col].dropna())
            summary = "; ".join([f"{k}: {v}명({v/total*100:.1f}%)" for k, v in counts.items()])
            lines.append(f"- {col}: {summary}")
            
    if selected_objective_columns:
        lines.append("\n[객관식 문항 분포]")
        for col in selected_objective_columns:
            counts = df[col].dropna().value_counts().head(5)
            total = len(df[col].dropna())
            summary = "; ".join([f"{k}: {v}명({v/total*100:.1f}%)" for k, v in counts.items()])
            lines.append(f"- {col}: {summary}")

    if selected_numeric_columns:
        lines.append("\n[기술통계 지표]")
        for col in selected_numeric_columns:
            numeric = pd.to_numeric(df[col], errors="coerce").dropna()
            if not numeric.empty:
                lines.append(f"- {col}: 평균 {numeric.mean():.2f}, 중앙값 {numeric.median():.2f}")
    return "\n".join(lines)

# ========== [App Header] ==========
inject_custom_css()
st.markdown("""
    <div class="premium-header">
        <h1>📊 설문조사 입체적 종합 분석 대시보드</h1>
        <p>정량적 기술 통계부터 AI 기반 공감맵, 동시출현 네트워크 분석 및 창의적 HMW 도출까지 설문 데이터를 입체적으로 시각화합니다.</p>
    </div>
""", unsafe_allow_html=True)

# ========== [Sidebar Configuration] ==========
st.sidebar.markdown("### 🔑 API 설정")
default_key = st.secrets.get("GEMINI_API_KEY", st.session_state.get("gemini_api_key", ""))
if st.secrets.get("GEMINI_API_KEY", ""):
    st.sidebar.success("✅ Secrets에서 API Key가 자동 로드되었습니다.")
gemini_key_input = st.sidebar.text_input(
    "Gemini API Key 입력", 
    type="password", 
    value=default_key,
    help="Gemini API Key를 입력하면 공감맵, 네트워크, HMW 분석이 활성화됩니다."
)
if gemini_key_input:
    st.session_state["gemini_api_key"] = gemini_key_input

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 데이터 소스 업로드")
uploaded_file = st.sidebar.file_uploader("설문조사 결과 파일 업로드 (CSV 또는 Excel)", type=['csv', 'xlsx'])

# ========== [Core Processing Logic] ==========
if uploaded_file is not None:
    # Load dataset
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"파일을 읽는 도중 오류가 발생했습니다: {e}")
        st.stop()
        
    st.sidebar.success("성공적으로 업로드되었습니다!")
    
    # Session state for data persistence
    if "df" not in st.session_state:
        st.session_state.df = df_raw.copy()
        st.session_state.raw_df = df_raw.copy()
        
    df = st.session_state.df
    
    # Step definition
    steps = ["Step 1: 데이터 업로드 및 전처리", 
             "Step 2: 인구통계 분석", 
             "Step 3: 기술 통계 분석", 
             "Step 4: 공감 맵 분석", 
             "Step 5: 네트워크 분석", 
             "Step 6: HMW 도출"]
             
    tabs = st.tabs(steps)
    
    # ------------------ [Step 1: Preprocessing] ------------------
    with tabs[0]:
        st.subheader("🛠️ 데이터 전처리 및 확인")
        
        col_prep_1, col_prep_2 = st.columns([1, 3])
        with col_prep_1:
            st.markdown("##### 전처리 설정")
            do_preprocess = st.radio("데이터 전처리를 수행하시겠습니까?", ["아니오 (기본값 사용)", "예 (결측치 제거 및 공백 정제)"], index=0)
            
            if do_preprocess == "예 (결측치 제거 및 공백 정제)":
                if st.button("전처리 실행하기"):
                    cleaned_df = st.session_state.raw_df.copy()
                    # 1. Drop rows where all elements are NaN
                    cleaned_df = cleaned_df.dropna(how='all')
                    # 2. String values strip whitespace
                    for col in cleaned_df.columns:
                        if cleaned_df[col].dtype == "object":
                            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                            # Replace empty strings with NaN
                            cleaned_df[col] = cleaned_df[col].replace(r'^\s*$', np.nan, regex=True)
                    
                    st.session_state.df = cleaned_df
                    st.success("데이터 전처리가 완료되었습니다!")
                    st.rerun()
            else:
                if st.button("원본 데이터로 복원"):
                    st.session_state.df = st.session_state.raw_df.copy()
                    st.info("원본 데이터 상태로 복원되었습니다.")
                    st.rerun()
                    
        with col_prep_2:
            st.markdown("##### 데이터 구조 요약")
            shape_col_1, shape_col_2, shape_col_3 = st.columns(3)
            with shape_col_1:
                st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(st.session_state.raw_df)}</div><div class="kpi-label">원본 행(Rows) 수</div></div>', unsafe_allow_html=True)
            with shape_col_2:
                st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df)}</div><div class="kpi-label">현재 행(Rows) 수</div></div>', unsafe_allow_html=True)
            with shape_col_3:
                st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df.columns)}</div><div class="kpi-label">열(Columns) 수</div></div>', unsafe_allow_html=True)
                
        st.markdown("##### 현재 데이터셋 미리보기 (상위 5개 행)")
        st.dataframe(df.head(5), use_container_width=True)
        
    # Infer columns based on current state of df
    detected_sub = infer_subjective_columns(df)
    
    # Save column selections globally in session state
    if "selected_sub_cols" not in st.session_state:
        st.session_state.selected_sub_cols = detected_sub
        
    # Main column settings globally accessible
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 컬럼 분석 구성")
    
    sub_cols = st.sidebar.multiselect(
        "주관식 서술형 열 선택",
        options=list(df.columns),
        default=[c for c in detected_sub if c in df.columns]
    )
    st.session_state.selected_sub_cols = sub_cols
    
    detected_cat = infer_categorical_columns(df, sub_cols)
    detected_dem = infer_demographic_columns(detected_cat)
    detected_num = infer_numeric_columns(df, sub_cols)
    
    dem_cols = st.sidebar.multiselect(
        "인구통계학(범주형) 열 선택",
        options=[c for c in df.columns if c not in sub_cols],
        default=[c for c in detected_dem if c in df.columns]
    )
    
    obj_cols = st.sidebar.multiselect(
        "객관식/범주형 열 선택",
        options=[c for c in df.columns if c not in sub_cols],
        default=[c for c in detected_cat if c in df.columns and c not in dem_cols]
    )
    
    num_cols = st.sidebar.multiselect(
        "수치형/척도형 열 선택",
        options=[c for c in df.columns if c not in sub_cols],
        default=[c for c in detected_num if c in df.columns]
    )
    
    # ------------------ [Step 2: Demographic Analysis] ------------------
    with tabs[1]:
        st.subheader("👥 인구통계 및 범주형 응답 분석")
        if not dem_cols:
            st.info("인구통계학 열이 선택되지 않았습니다. 사이드바에서 분석할 인구통계학 열을 구성해 주세요.")
        else:
            selected_dem_col = st.selectbox("분석 대상 인구통계 열 선택", dem_cols)
            
            # Compute frequency & ratio
            series = df[selected_dem_col].dropna().astype(str).str.strip()
            if series.empty:
                st.warning("선택된 열에 유효한 응답이 없습니다.")
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
                    st.markdown(f"##### {selected_dem_col} 빈도 분석 테이블")
                    st.dataframe(df_freq, use_container_width=True, hide_index=True)
                    
                    # Copy and Download options
                    st.markdown("🤖 **복사 및 다운로드**")
                    tsv_data = df_freq.to_csv(sep='\t', index=False)
                    st.code(tsv_data, language='text')
                    st.caption("위 텍스트 박스 우측의 복사 버튼을 눌러 스프레드시트 등에 바로 붙여넣으세요.")
                    
                    csv_data = df_freq.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 CSV 다운로드",
                        data=csv_data,
                        file_name=f"demographic_{selected_dem_col}.csv",
                        mime="text/csv"
                    )
                    
                with col_d_2:
                    st.markdown("##### 시각화 차트")
                    chart_type = st.radio("차트 타입 선택", ["원형 차트 (Pie Chart)", "막대 차트 (Bar Chart)"])
                    
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
                    
                    # Static PNG Download Option via Matplotlib fallback
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
                        st.caption(f"차트 PNG 생성 중 일시적 오류: {e}")

    # ------------------ [Step 3: Descriptive Analysis] ------------------
    with tabs[2]:
        st.subheader("📈 수치형 및 척도 만족도 기술 통계")
        if not num_cols:
            st.info("수치형 또는 척도형(만족도 등) 열이 선택되지 않았습니다. 사이드바에서 열을 선택해 주세요.")
        else:
            desc_rows = []
            for col in num_cols:
                numeric = pd.to_numeric(df[col], errors="coerce").dropna()
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
                st.warning("선택된 열들에서 유효한 수치형 데이터를 분석할 수 없습니다.")
            else:
                df_desc = pd.DataFrame(desc_rows)
                st.dataframe(df_desc, use_container_width=True, hide_index=True)
                
                # Copy and Download options
                st.markdown("🤖 **복사 및 다운로드**")
                tsv_desc = df_desc.to_csv(sep='\t', index=False)
                st.code(tsv_desc, language='text')
                st.caption("위 텍스트 박스 우측의 복사 버튼을 눌러 클립보드에 담을 수 있습니다.")
                
                csv_desc = df_desc.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 CSV 다운로드",
                    data=csv_desc,
                    file_name="descriptive_statistics.csv",
                    mime="text/csv"
                )
                
                # Simple Plotly Bar Chart comparing means
                st.markdown("##### 평균값 비교 시각화")
                fig_mean = px.bar(df_desc, x='문항', y='평균', text='평균',
                                  title="척도형 문항 평균 비교",
                                  color='평균', color_continuous_scale=px.colors.sequential.Teal)
                fig_mean.update_traces(textposition='outside')
                fig_mean.update_layout(font=dict(family="Noto Sans KR"))
                st.plotly_chart(fig_mean, use_container_width=True)

    # Check for Gemini Client
    client = get_gemini_client()
    
    # ------------------ [Step 4: Empathy Map Analysis] ------------------
    with tabs[3]:
        st.subheader("💡 AI 기반 주관식 응답 공감 맵(Empathy Map)")
        
        if not sub_cols:
            st.warning("주관식 답변 열을 1개 이상 선택해 주세요.")
        elif client is None:
            st.warning("공감 맵 분석을 진행하려면 사이드바에 **Gemini API Key**를 입력해 주세요.")
        else:
            survey_text = build_survey_text_summary(df, sub_cols)
            
            st.info("선택된 주관식 응답 내용을 분석하여 공감맵(Says, Thinks, Does, Feels)을 자동 구성합니다.")
            
            # Perform Gemini Analysis
            if st.button("공감 맵 분석 실행", key="run_empathy_map"):
                with st.spinner("Gemini AI가 주관식 응답의 감정과 태도를 입체적으로 분류 중입니다..."):
                    prompt = f"""
                    설문조사의 주관식 답변 데이터를 바탕으로, 대상 사용자의 심리를 분석하여 2x2 공감 맵(Empathy Map)을 작성해주세요.
                    공감 맵은 다음 4가지 영역으로 구성됩니다:
                    1. Says (말하는 것): 사용자가 겉으로 표현한 대표적인 의견, 요구사항, 피드백
                    2. Thinks (생각하는 것): 사용자가 명시적으로 말하지는 않았지만 마음속으로 고민하거나 바라는 생각/기대
                    3. Does (행동하는 것): 사용자가 직면한 상황에서 취하는 구체적인 행동, 프로세스, 습관
                    4. Feels (느끼는 것): 사용자가 느끼는 긍정적/부정적/중립적 감정, 불안, 기쁨, 답답함

                    [주관식 답변 데이터]
                    {survey_text}

                    [출력 형식]
                    반드시 다음 JSON 형식으로만 응답해주세요. 다른 부연 설명이나 마크다운 코드 펜스는 제외해주세요.
                    {{
                      "says": ["의견 1", "의견 2", "의견 3", "의견 4"],
                      "thinks": ["고민 1", "고민 2", "고민 3", "고민 4"],
                      "does": ["행동 1", "행동 2", "행동 3", "행동 4"],
                      "feels": [
                        {{"text": "감정 1", "type": "pos"}},
                        {{"text": "감정 2", "type": "neg"}},
                        {{"text": "감정 3", "type": "neu"}},
                        {{"text": "감정 4", "type": "neg"}}
                      ]
                    }}
                    (참고: feels의 type은 'pos'(긍정), 'neg'(부정), 'neu'(중립) 중 하나로 지정해주세요. 각 영역별로 최소 4개씩 도출해주세요.)
                    """
                    
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt,
                            config={'temperature': 0.1}
                        )
                        empathy_dict = parse_json_from_response(response.text)
                        st.session_state.empathy_data = empathy_dict
                        st.success("공감 맵 분석이 완료되었습니다!")
                    except Exception as e:
                        st.error(f"분석 중 오류 발생: {e}")
                        
            # Render Empathy Map if exists in session state
            if "empathy_data" in st.session_state:
                em_data = st.session_state.empathy_data
                
                # Render 2x2 grid using HTML cards
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
                    
                # Copy & Download Buttons
                st.markdown("---")
                st.markdown("🤖 **복사 및 다운로드**")
                
                # Format raw text for easy copy
                raw_empathy_text = "=== [공감 맵 분석 결과] ===\n\n"
                raw_empathy_text += "[SAYS - 말한다]\n" + "\n".join([f"- {i}" for i in em_data.get("says", [])]) + "\n\n"
                raw_empathy_text += "[THINKS - 생각한다]\n" + "\n".join([f"- {i}" for i in em_data.get("thinks", [])]) + "\n\n"
                raw_empathy_text += "[DOES - 행동한다]\n" + "\n".join([f"- {i}" for i in em_data.get("does", [])]) + "\n\n"
                raw_empathy_text += "[FEELS - 느낀다]\n" + "\n".join([f"- [{i.get('type')}] {i.get('text')}" for i in em_data.get("feels", [])])
                
                st.code(raw_empathy_text, language='text')
                st.caption("위 텍스트 박스에서 손쉽게 텍스트로 분석 결과를 복사해 가세요.")
                
                st.download_button(
                    label="📥 JSON 다운로드",
                    data=json.dumps(em_data, ensure_ascii=False, indent=2),
                    file_name="empathy_map.json",
                    mime="application/json"
                )

    # ------------------ [Step 5: Network Analysis] ------------------
    with tabs[4]:
        st.subheader("🕸️ 주관식 키워드 동시출현 네트워크 분석")
        
        if not sub_cols:
            st.warning("주관식 답변 열을 1개 이상 선택해 주세요.")
        elif client is None:
            st.warning("네트워크 분석을 실행하려면 사이드바에 **Gemini API Key**를 입력해 주세요.")
        else:
            st.info("Gemini AI를 활용해 텍스트에서 주요 핵심 키워드(10~15개)를 명사 위주로 자동 추출한 후, 답변 본문에서 동시에 등장하는 관계를 매핑합니다.")
            
            if st.button("네트워크 분석 실행", key="run_network"):
                with st.spinner("핵심 키워드를 분석하고 동시출현 관계망을 구성하는 중..."):
                    survey_text = build_survey_text_summary(df, sub_cols)
                    
                    # 1. Ask Gemini to extract keywords
                    prompt = f"""
                    설문조사 주관식 답변들에서 핵심이 되는 명사 및 구문(키워드)을 10~15개 추출해 주세요.
                    주로 프로그램 참가자들이 겪은 경험, Pain Point, 요구사항과 관련된 중요한 키워드를 골라야 합니다.
                    출력 형식은 오직 쉼표로만 구분된 문자열이어야 합니다. 다른 설명이나 특수문자, 마크다운 펜스는 절대 포함하지 마세요.

                    예: 협업, 일정 관리, 소통, 의견 대립, 성취감, 역할 분담, 난이도 조절

                    [주관식 답변 데이터]
                    {survey_text}
                    """
                    
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt
                        )
                        keywords_raw = response.text.strip()
                        keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
                        
                        # Remove duplicates and clean
                        keywords = list(dict.fromkeys(keywords))
                        
                        # 2. Gather all responses to calculate co-occurrence
                        all_responses = []
                        for col in sub_cols:
                            all_responses.extend([
                                str(val).strip()
                                for val in df[col].dropna()
                                if str(val).strip()
                            ])
                            
                        # Calculate frequencies of keywords
                        freq_dict = {}
                        for kw in keywords:
                            freq_dict[kw] = sum(1 for r in all_responses if kw in r)
                            
                        # Build co-occurrence matrix
                        co_matrix = pd.DataFrame(0, index=keywords, columns=keywords)
                        for r in all_responses:
                            present = [kw for kw in keywords if kw in r]
                            for i in range(len(present)):
                                for j in range(i + 1, len(present)):
                                    co_matrix.loc[present[i], present[j]] += 1
                                    co_matrix.loc[present[j], present[i]] += 1
                                    
                        st.session_state.network_keywords = keywords
                        st.session_state.network_matrix = co_matrix
                        st.session_state.network_frequencies = freq_dict
                        st.success("네트워크 분석 관계망 구성 완료!")
                    except Exception as e:
                        st.error(f"네트워크 분석 중 오류 발생: {e}")
                        
            # Render network plot if exists
            if "network_matrix" in st.session_state:
                keywords = st.session_state.network_keywords
                co_matrix = st.session_state.network_matrix
                freq_dict = st.session_state.network_frequencies
                
                # Check empty graph
                has_edges = (co_matrix.values.sum() > 0)
                
                # Create NetworkX graph
                G = nx.Graph()
                for kw in keywords:
                    G.add_node(kw, size=freq_dict.get(kw, 1))
                for i in range(len(keywords)):
                    for j in range(i + 1, len(keywords)):
                        weight = co_matrix.iloc[i, j]
                        if weight > 0:
                            G.add_edge(keywords[i], keywords[j], weight=weight)
                            
                # Layout
                if not has_edges:
                    pos = {node: np.array([np.cos(2*np.pi*i/len(keywords)), np.sin(2*np.pi*i/len(keywords))]) for i, node in enumerate(keywords)}
                else:
                    pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)
                    
                # Draw Interactive Network Graph via Plotly
                edge_traces = []
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    weight = edge[2].get('weight', 1)
                    width = min(1 + weight * 0.8, 6.0)
                    edge_trace = go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        line=dict(width=width, color='rgba(148, 163, 184, 0.4)'),
                        hoverinfo='none',
                        mode='lines'
                    )
                    edge_traces.append(edge_trace)
                    
                node_x = []
                node_y = []
                node_text = []
                node_size = []
                node_color = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(f"<b>{node}</b><br>빈도수: {freq_dict.get(node, 0)}회<br>연결된 키워드 수: {G.degree(node)}")
                    node_size.append(min(15 + freq_dict.get(node, 1) * 2, 45))
                    node_color.append(G.degree(node))
                    
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=[node for node in G.nodes()],
                    textposition="top center",
                    hovertext=node_text,
                    textfont=dict(size=12, color='#0f172a', family="Noto Sans KR"),
                    marker=dict(
                        showscale=True,
                        colorscale='Viridis',
                        reversescale=True,
                        color=node_color,
                        size=node_size,
                        colorbar=dict(
                            title='연결도 (Degree)',
                            thickness=15,
                            x=1.02,
                            len=0.6
                        ),
                        line=dict(width=2, color='white')
                    )
                )
                
                fig_net = go.Figure(
                    data=edge_traces + [node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=20, r=20, t=20),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(248, 250, 252, 0.4)',
                        paper_bgcolor='white',
                        height=550
                    )
                )
                
                col_n_1, col_n_2 = st.columns([2, 1])
                with col_n_1:
                    st.markdown("##### 인터랙티브 키워드 네트워크 관계망")
                    st.plotly_chart(fig_net, use_container_width=True)
                    st.caption("💡 노드를 클릭해 드래그하거나 휠을 사용해 줌인/아웃이 가능합니다. 키워드에 마우스를 호버하면 상세 정보가 표시됩니다.")
                
                with col_n_2:
                    st.markdown("##### 동시출현 키워드 행렬 (상위 연관 관계)")
                    st.dataframe(co_matrix, use_container_width=True)
                    
                    # Downloads & Copy
                    st.markdown("🤖 **복사 및 다운로드**")
                    
                    # 1. Save interactive graph as HTML
                    html_buffer = io.StringIO()
                    fig_net.write_html(html_buffer, include_plotlyjs='cdn')
                    html_bytes = html_buffer.getvalue().encode('utf-8')
                    
                    st.download_button(
                        label="🌐 인터랙티브 그래프 HTML 다운로드",
                        data=html_bytes,
                        file_name="keyword_network.html",
                        mime="text/html"
                    )
                    
                    # 2. Draw static Matplotlib network for static image download
                    try:
                        fig_plt, ax = plt.subplots(figsize=(6, 5))
                        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[freq_dict.get(n, 1)*50 + 200 for n in G.nodes()], node_color=node_color, cmap=plt.cm.viridis)
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
                            file_name="keyword_network.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.caption(f"PNG 파일 렌더링 지연 발생: {e}")
                    
                    # 3. Download Matrix
                    csv_matrix = co_matrix.to_csv(index=True).encode('utf-8-sig')
                    st.download_button(
                        label="📥 동시출현 매트릭스 CSV 다운로드",
                        data=csv_matrix,
                        file_name="co_occurrence_matrix.csv",
                        mime="text/csv"
                    )

    # ------------------ [Step 6: HMW (How Might We) Analysis] ------------------
    with tabs[5]:
        st.subheader("💡 AI 기반 HMW (How Might We) 기회 정의")
        
        if not sub_cols:
            st.warning("주관식 답변 열을 1개 이상 선택해 주세요.")
        elif client is None:
            st.warning("HMW 질문 생성을 시작하려면 사이드바에 **Gemini API Key**를 입력해 주세요.")
        else:
            st.info("설문조사 응답과 정량 요약 정보에서 도출된 Pain Point를 바탕으로, 아이디에이션의 방아쇠가 될 긍정적이고 창의적인 HMW 질문을 자동 도출합니다.")
            
            if st.button("HMW 질문 생성", key="run_hmw"):
                with st.spinner("Gemini AI가 HMW 질문 가이드라인(7요소)을 적용하여 생성하는 중..."):
                    survey_text = build_survey_text_summary(df, sub_cols)
                    quant_summary = build_quantitative_summary(df, sub_cols, dem_cols, obj_cols, num_cols)
                    
                    prompt = f"""
                    설문조사의 정량 통계 요약 및 주관식 답변을 분석하여 사용자의 핵심 Pain Point와 인사이트를 도출하고,
                    [문제의 기회 요소 발견 7요소 기반 HMW 가이드]를 참고하여 창의적이고 해결 가능한 HMW(How Might We) 질문을 5~7개 생성해 주세요.

                    [인구통계 및 기술통계 요약]
                    {quant_summary}

                    [주관식 답변 데이터]
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
                            config={'temperature': 0.2}
                        )
                        hmw_dict = parse_json_from_response(response.text)
                        st.session_state.hmw_data = hmw_dict
                        st.success("HMW 기회 도출 완료!")
                    except Exception as e:
                        st.error(f"HMW 질문 생성 중 오류 발생: {e}")
                        
            # Render HMW questions if exists
            if "hmw_data" in st.session_state:
                hmw_dict = st.session_state.hmw_data
                
                # Card UI display
                st.markdown("##### 기회 요소 7요소 기반 HMW 카드")
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
                    
                # Copy & Download Options
                st.markdown("---")
                st.markdown("🤖 **복사 및 다운로드**")
                
                # Format raw text for easy copy
                raw_hmw_text = "=== [HMW 질문 리스트] ===\n\n"
                for idx, item in enumerate(hmw_dict.get("hmw_list", [])):
                    raw_hmw_text += f"{idx+1}. [{item.get('opportunity')}] {item.get('direction')}\n"
                    raw_hmw_text += f"   Q: {item.get('question')}\n\n"
                    
                st.code(raw_hmw_text, language='text')
                st.caption("위 텍스트 박스 우측 복사 아이콘으로 간편히 클립보드에 복사하세요.")
                
                st.download_button(
                    label="📥 HMW 질문 TXT 다운로드",
                    data=raw_hmw_text,
                    file_name="how_might_we_questions.txt",
                    mime="text/plain"
                )
else:
    # Landing message if no file uploaded
    st.info("👈 왼쪽 사이드바에서 분석할 설문조사 데이터(CSV/Excel) 파일을 업로드해 주세요.")
    
    # Beautiful mockup description of features
    st.markdown("""
        <div style="padding: 20px; background-color: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; margin-top: 15px;">
            <h4 style="margin-top:0; color:#1e293b;">🚀 주요 제공 기능 및 분석 가이드</h4>
            <ul style="font-size: 13.5px; line-height: 1.8; color: #475569; padding-left: 20px; margin-bottom: 0;">
                <li><strong>Step 1: 데이터 업로드 및 전처리</strong> - 빈 데이터(NaN), 중복값, 공백 제거 기능 제공</li>
                <li><strong>Step 2: 인구통계 분석</strong> - 성별, 연령, 전공 등 빈도분석 테이블 및 반응형 원형/막대 차트 시각화</li>
                <li><strong>Step 3: 기술 통계 분석</strong> - 만족도와 같은 5점 척도/수치형 데이터 평균, 중앙값, 분산 경향성 도출</li>
                <li><strong>Step 4: 공감 맵 분석</strong> - 주관식 답변에서 Says, Thinks, Does, Feels를 도출하는 고급 AI 세션</li>
                <li><strong>Step 5: 네트워크 분석</strong> - 주관식 답변 핵심 키워드 간의 연관 관계를 보여주는 동시출현 의미 관계망 시각화</li>
                <li><strong>Step 6: HMW 도출</strong> - 문제 정의 가이드와 Pain Point에 기반해 아이디어를 확장하는 7요소 기반 HMW 질문 카드 자동생성</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
