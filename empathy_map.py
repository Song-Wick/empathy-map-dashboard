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
    # Gemini sometimes escapes the emphasis tags shown in the template instructions.
    html = re.sub(r"&lt;\s*strong\s*&gt;", "<strong>", html, flags=re.IGNORECASE)
    html = re.sub(r"&lt;\s*/\s*strong\s*&gt;", "</strong>", html, flags=re.IGNORECASE)
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
                - 아래 제공된 [HTML 템플릿]의 구조(태그, 클래스명 등)와 CSS 스타일을 단 하나도 수정하거나 삭제하지 말고 100% 그대로 유지하세요.
                - 템플릿 내부의 안내 텍스트(예: "여기에 분석 결과를 작성하세요")만 실제 분석 결과로 교체하세요.
                - 마크다운 기호(```html 등)는 완전히 제외하고 순수 HTML 텍스트만 리턴해야 합니다.
                
                [설문 응답 데이터]
                {survey_text}

                [HTML 템플릿]
                <!DOCTYPE html>
                <html lang="ko">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>프로그램 참가자 설문조사 분석 대시보드</title>
                    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
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
                            flex-wrap: wrap;
                            gap: 16px;
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
                        .tag-prob   {{ background: #fef2f2; color: #991b1b; }}
                        .tag-action {{ background: #fffbeb; color: #92400e; }}

                        .issue-card p {{
                            font-size: 13px;
                            color: #3a4553;
                            line-height: 1.75;
                        }}

                        /* ── SECTION 2: EMPATHY MAP ── */
                        .empathy-wrapper {{
                            margin-bottom: 40px;
                        }}
                        .empathy-center {{
                            background: #1e2a3a;
                            color: #fff;
                            text-align: center;
                            padding: 14px 24px;
                            border-radius: 10px 10px 0 0;
                            font-size: 13px;
                            font-weight: 700;
                            letter-spacing: 0.04em;
                        }}
                        .empathy-grid {{
                            display: grid;
                            grid-template-columns: 1fr 1fr;
                            gap: 0;
                            border: 0.5px solid #e3e8ef;
                            border-radius: 0 0 10px 10px;
                            overflow: hidden;
                        }}
                        .quadrant {{
                            padding: 22px 24px;
                            background: #fff;
                        }}
                        .quadrant:nth-child(1) {{ border-right: 0.5px solid #e3e8ef; border-bottom: 0.5px solid #e3e8ef; }}
                        .quadrant:nth-child(2) {{ border-bottom: 0.5px solid #e3e8ef; }}
                        .quadrant:nth-child(3) {{ border-right: 0.5px solid #e3e8ef; }}

                        .q-header {{
                            display: flex;
                            align-items: center;
                            gap: 8px;
                            margin-bottom: 14px;
                        }}
                        .q-dot {{
                            width: 10px; height: 10px;
                            border-radius: 50%;
                            flex-shrink: 0;
                        }}
                        .says .q-dot   {{ background: #f59e0b; }}
                        .thinks .q-dot {{ background: #3b82f6; }}
                        .does .q-dot   {{ background: #ef4444; }}
                        .feels .q-dot  {{ background: #8b5cf6; }}

                        .q-title {{
                            font-size: 12px;
                            font-weight: 700;
                            letter-spacing: 0.06em;
                            text-transform: uppercase;
                            color: #1e2a3a;
                        }}
                        .q-title-ko {{
                            font-size: 11px;
                            color: #8a95a3;
                            font-weight: 400;
                        }}

                        .quadrant ul {{
                            list-style: none;
                            padding: 0;
                        }}
                        .quadrant ul li {{
                            font-size: 12.5px;
                            color: #3a4553;
                            line-height: 1.65;
                            padding: 5px 0;
                            border-bottom: 0.5px solid #f0f2f5;
                        }}
                        .quadrant ul li:last-child {{ border-bottom: none; }}
                        .quadrant ul li::before {{
                            content: '—';
                            color: #c8d0dc;
                            margin-right: 7px;
                        }}

                        .feel-badge {{
                            display: inline-block;
                            font-size: 11px;
                            font-weight: 700;
                            padding: 2px 8px;
                            border-radius: 4px;
                            margin-right: 2px;
                        }}
                        .feel-pos  {{ background: #ecfdf5; color: #065f46; }}
                        .feel-neg  {{ background: #fef2f2; color: #991b1b; }}
                        .feel-neu  {{ background: #f5f3ff; color: #5b21b6; }}

                        /* ── SECTION 3: NETWORKING ── */
                        .methodology-note {{
                            background: #f8f9fb;
                            border: 1px solid #e3e8ef;
                            border-left: 3px solid #3b82f6;
                            padding: 12px 16px;
                            font-size: 12px;
                            color: #5a6575;
                            border-radius: 4px 8px 8px 4px;
                            margin-bottom: 20px;
                            line-height: 1.6;
                        }}
                        .methodology-note strong {{
                            color: #1e2a3a;
                        }}
                        .network-wrapper {{
                            margin-bottom: 40px;
                        }}
                        .keyword-row {{
                            display: flex;
                            flex-wrap: wrap;
                            gap: 8px;
                            margin-bottom: 18px;
                        }}
                        .kw-badge {{
                            display: inline-flex;
                            align-items: center;
                            gap: 6px;
                            background: #f0f2f5;
                            color: #1e2a3a;
                            border: 0.5px solid #d0d7e3;
                            padding: 6px 14px;
                            border-radius: 6px;
                            font-size: 12px;
                            font-weight: 700;
                        }}
                        .kw-badge .dot {{
                            width: 6px; height: 6px;
                            border-radius: 50%;
                            background: #10b981;
                        }}
                        .network-list {{
                            display: flex;
                            flex-direction: column;
                            gap: 8px;
                        }}
                        .network-item {{
                            background: #f8f9fb;
                            border: 0.5px solid #e3e8ef;
                            border-radius: 8px;
                            padding: 14px 18px;
                            font-size: 13px;
                            color: #3a4553;
                            line-height: 1.7;
                        }}
                        .network-item .em {{
                            font-weight: 700;
                            color: #1e2a3a;
                        }}

                        /* ── SECTION 4: PAIN POINT IDENTIFICATION ── */
                        .pp-grid {{
                            display: grid;
                            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                            gap: 16px;
                            margin-bottom: 40px;
                        }}
                        .pp-card {{
                            background: #fff;
                            border: 0.5px solid #e3e8ef;
                            border-radius: 10px;
                            padding: 20px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
                        }}
                        .pp-card.level-1 {{ border-top: 4px solid #f59e0b; }}
                        .pp-card.level-2 {{ border-top: 4px solid #ef4444; }}
                        .pp-card.level-3 {{ border-top: 4px solid #8b5cf6; }}
                        
                        .pp-header {{
                            font-weight: 700;
                            font-size: 14px;
                            margin-bottom: 12px;
                            color: #1e2a3a;
                        }}
                        .pp-card ul {{
                            list-style: none;
                            padding: 0;
                        }}
                        .pp-card ul li {{
                            position: relative;
                            padding-left: 14px;
                            margin-bottom: 8px;
                            font-size: 13px;
                            color: #3a4553;
                            line-height: 1.6;
                        }}
                        .pp-card ul li::before {{
                            content: "•";
                            color: #b0bac8;
                            font-weight: bold;
                            position: absolute;
                            left: 0;
                        }}

                        /* ── SECTION 5: PROBLEM DEFINITION (RESTRUCTURED) ── */
                        .prob-def-box {{
                            background: #f8f9fb;
                            border: 0.5px solid #e3e8ef;
                            border-radius: 10px;
                            padding: 24px;
                            margin-bottom: 40px;
                        }}
                        .prob-stmt {{
                            display: flex;
                            flex-wrap: wrap;
                            gap: 12px;
                        }}
                        .stmt-block {{
                            background: #fff;
                            padding: 18px 20px;
                            border-radius: 8px;
                            flex: 1;
                            min-width: 220px;
                            border: 0.5px solid #e3e8ef;
                            border-left: 3px solid #3b82f6;
                            box-shadow: 0 2px 6px rgba(0,0,0,0.02);
                        }}
                        .stmt-block.needs {{ border-left-color: #10b981; }}
                        .stmt-block.insight {{ border-left-color: #ef4444; }}
                        
                        .stmt-label {{
                            font-size: 11px;
                            font-weight: 700;
                            color: #8a95a3;
                            text-transform: uppercase;
                            margin-bottom: 8px;
                        }}
                        .stmt-text {{
                            font-size: 13.5px;
                            color: #1e2a3a;
                            line-height: 1.7;
                        }}
                        .stmt-text strong {{
                            color: #1e2a3a;
                            background: rgba(59, 130, 246, 0.1);
                            padding: 0 4px;
                            border-radius: 2px;
                        }}

                        /* ── SECTION 6: HMW ── */
                        .hmw-list {{
                            display: flex;
                            flex-direction: column;
                            gap: 12px;
                            margin-bottom: 20px;
                        }}
                        .hmw-item {{
                            background: #fff;
                            border: 0.5px solid #e3e8ef;
                            border-left: 4px solid #10b981;
                            padding: 20px 24px;
                            border-radius: 8px;
                            display: flex;
                            align-items: center;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
                        }}
                        .hmw-icon {{
                            font-size: 24px;
                            margin-right: 18px;
                            flex-shrink: 0;
                        }}
                        .hmw-content h4 {{
                            font-size: 12px;
                            color: #8a95a3;
                            margin-bottom: 4px;
                            font-weight: 500;
                        }}
                        .hmw-content p {{
                            font-size: 14.5px;
                            font-weight: 700;
                            color: #065f46;
                            margin: 0;
                            line-height: 1.6;
                        }}

                        /* ── FOOTER ── */
                        .footer {{
                            margin-top: 48px;
                            padding-top: 20px;
                            border-top: 0.5px solid #e3e8ef;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                        }}
                        .footer-left {{
                            font-size: 11px;
                            color: #b0bac8;
                        }}
                        .footer-right {{
                            font-size: 11px;
                            color: #b0bac8;
                        }}

                        /* ── DIVIDER ── */
                        .section-divider {{
                            height: 0.5px;
                            background: #e3e8ef;
                            margin: 36px 0;
                        }}

                        @media print {{
                            body {{ background: #fff; padding: 0; }}
                            .page {{ box-shadow: none; border-radius: 0; }}
                        }}
                    </style>
                </head>
                <body>
                <div class="page">
                    <div class="header">
                        <div class="header-meta">분석 보고서 &nbsp;·&nbsp; Survey Analysis Report</div>
                        <h1>프로그램 참가자 설문조사 분석 대시보드</h1>
                        <div class="header-sub">데이터 기반 설문 피드백 종합 분석 및 문제 해결 도출</div>
                        <div class="header-divider">
                            <span class="header-pill">이슈 구조화</span>
                            <span class="header-pill">공감 맵</span>
                            <span class="header-pill">네트워킹 분석</span>
                            <span class="header-pill">Pain Point 식별</span>
                            <span class="header-pill">문제 정의 및 HMW</span>
                        </div>
                    </div>
                    <div class="content">
                        <div class="section-label">
                            <div class="section-number">1</div>
                            <div>
                                <span class="section-title">이슈 구조화</span>
                                <span class="section-title-en">Issue Structuring</span>
                            </div>
                        </div>
                        <div class="issue-grid">
                            <div class="issue-card user">
                                <span class="issue-tag tag-user">사용자</span>
                                <p>여기에 사용자에 대한 분석 결과를 작성하세요.</p>
                            </div>
                            <div class="issue-card need">
                                <span class="issue-tag tag-need">요구사항</span>
                                <p>여기에 요구사항에 대한 분석 결과를 작성하세요.</p>
                            </div>
                            <div class="issue-card goal">
                                <span class="issue-tag tag-goal">목표</span>
                                <p>여기에 목표에 대한 분석 결과를 작성하세요.</p>
                            </div>
                            <div class="issue-card prob">
                                <span class="issue-tag tag-prob">문제점</span>
                                <p>여기에 문제점에 대한 분석 결과를 작성하세요.</p>
                            </div>
                            <div class="issue-card action">
                                <span class="issue-tag tag-action">행동</span>
                                <p>여기에 행동에 대한 분석 결과를 작성하세요.</p>
                            </div>
                        </div>
                        
                        <div class="section-divider"></div>
                        
                        <div class="section-label">
                            <div class="section-number">2</div>
                            <div>
                                <span class="section-title">공감 맵</span>
                                <span class="section-title-en">Empathy Map</span>
                            </div>
                        </div>
                        <div class="empathy-wrapper">
                            <div class="empathy-center">프로그램 참가자</div>
                            <div class="empathy-grid">
                                <div class="quadrant says">
                                    <div class="q-header">
                                        <div class="q-dot"></div>
                                        <span class="q-title">Says</span>
                                        <span class="q-title-ko">말한다</span>
                                    </div>
                                    <ul>
                                        <li>여기에 '말한다(Says)' 분석 결과를 작성하세요. (여러 개의 li 태그 사용)</li>
                                    </ul>
                                </div>
                                <div class="quadrant thinks">
                                    <div class="q-header">
                                        <div class="q-dot"></div>
                                        <span class="q-title">Thinks</span>
                                        <span class="q-title-ko">생각한다</span>
                                    </div>
                                    <ul>
                                        <li>여기에 '생각한다(Thinks)' 분석 결과를 작성하세요. (여러 개의 li 태그 사용)</li>
                                    </ul>
                                </div>
                                <div class="quadrant does">
                                    <div class="q-header">
                                        <div class="q-dot"></div>
                                        <span class="q-title">Does</span>
                                        <span class="q-title-ko">행동한다</span>
                                    </div>
                                    <ul>
                                        <li>여기에 '행동한다(Does)' 분석 결과를 작성하세요. (여러 개의 li 태그 사용)</li>
                                    </ul>
                                </div>
                                <div class="quadrant feels">
                                    <div class="q-header">
                                        <div class="q-dot"></div>
                                        <span class="q-title">Feels</span>
                                        <span class="q-title-ko">느낀다</span>
                                    </div>
                                    <ul>
                                        <li>여기에 '느낀다(Feels)' 분석 결과를 작성하세요. 긍정/부정/중립에 따라 적절히 &lt;span class="feel-badge feel-pos"&gt; &lt;span class="feel-badge feel-neg"&gt; &lt;span class="feel-badge feel-neu"&gt; 를 활용하세요.</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        
                        <div class="section-divider"></div>
                        
                        <div class="section-label">
                            <div class="section-number">3</div>
                            <div>
                                <span class="section-title">네트워킹 분석</span>
                                <span class="section-title-en">Networking Analysis</span>
                            </div>
                        </div>
                        <div class="methodology-note">
                            💡시각적인 점과 선 형태의 단순 그래프 대신, AI가 설문 응답의 전체 문맥을 파악하여 핵심 키워드 간의 인과관계와 상호작용을 스토리텔링 방식으로 풀어낸 <strong>의미 연결망(Semantic Network)</strong>입니다.
                        </div>    
                        <div class="network-wrapper">
                            <div class="keyword-row">
                                <span class="kw-badge"><span class="dot"></span>분석된 키워드1</span>
                                <span class="kw-badge"><span class="dot"></span>분석된 키워드2</span>
                                <span class="kw-badge"><span class="dot"></span>분석된 키워드3</span>
                            </div>
                            <div class="network-list">
                                <div class="network-item">
                                    키워드를 &lt;span class="em"&gt;강조&lt;/span&gt;하여 이들의 상호작용과 관계망을 서술하세요. (여러 개의 태그 생성 가능)
                                </div>
                            </div>
                        </div>

                        <div class="section-divider"></div>

                        <div class="section-label">
                            <div class="section-number">4</div>
                            <div>
                                <span class="section-title">Pain Point 식별 및 구조화</span>
                                <span class="section-title-en">Pain Point Identification</span>
                            </div>
                        </div>
                        <div class="pp-grid">
                            <div class="pp-card level-1">
                                <div class="pp-header">상호작용 레벨의 마찰</div>
                                <ul>
                                    <li>여기에 물리적, 시스템적, 즉각적인 마찰이나 불편 사항을 작성하세요.</li>
                                </ul>
                            </div>
                            <div class="pp-card level-2">
                                <div class="pp-header">사용자 여정 레벨의 지연</div>
                                <ul>
                                    <li>여기에 목적을 달성하는 과정에서 겪는 혼란이나 막막함 등을 작성하세요.</li>
                                </ul>
                            </div>
                            <div class="pp-card level-3">
                                <div class="pp-header">장기적 관계 레벨의 단절</div>
                                <ul>
                                    <li>여기에 신뢰 저하, 동기 부여 상실, 지속적인 참여를 가로막는 요소를 작성하세요.</li>
                                </ul>
                            </div>
                        </div>

                        <div class="section-label">
                            <div class="section-number">5</div>
                            <div>
                                <span class="section-title">문제 재정의</span>
                                <span class="section-title-en">Problem Redefinition</span>
                            </div>
                        </div>
                        <div class="prob-def-box">
                            <div class="prob-stmt">
                                <div class="stmt-block user">
                                    <div class="stmt-label">사용자 (User)</div>
                                    <div class="stmt-text">여기에 대상 사용자를 정의하세요. (예: 영상제에 참여하여 기획력을 표현하고자 하는 학생들은)</div>
                                </div>
                                <div class="stmt-block needs">
                                    <div class="stmt-label">니즈 (Needs to)</div>
                                    <div class="stmt-text">여기에 사용자가 진정으로 필요로 하는 것을 정의하세요. 중요한 부분은 <strong>강조</strong>하세요.</div>
                                </div>
                                <div class="stmt-block insight">
                                    <div class="stmt-label">인사이트 (Because)</div>
                                    <div class="stmt-text">여기에 니즈가 발생하는 근본적인 원인과 맥락을 정의하세요. 중요한 부분은 <strong>강조</strong>하세요.</div>
                                </div>
                            </div>
                        </div>

                        <div class="section-label">
                            <div class="section-number">6</div>
                            <div>
                                <span class="section-title">How Might We (HMW) 도출</span>
                                <span class="section-title-en">Ideation Trigger</span>
                            </div>
                        </div>
                        <div class="hmw-list">
                            <div class="hmw-item">
                                <div class="hmw-icon">💡</div>
                                <div class="hmw-content">
                                    <h4>여기에 해결할 문제의 방향성(예: 기획의 막막함 해결)을 작성하세요</h4>
                                    <p>우리가 어떻게 하면 ~ 할 수 있을까? (창의적인 아이디어를 유도하는 질문 작성)</p>
                                </div>
                            </div>
                            <div class="hmw-item">
                                <div class="hmw-icon">🚀</div>
                                <div class="hmw-content">
                                    <h4>여기에 두 번째 문제의 방향성을 작성하세요</h4>
                                    <p>우리가 어떻게 하면 ~ 할 수 있을까?</p>
                                </div>
                            </div>
                            <div class="hmw-item">
                                <div class="hmw-icon">🏆</div>
                                <div class="hmw-content">
                                    <h4>여기에 세 번째 문제의 방향성을 작성하세요</h4>
                                    <p>우리가 어떻게 하면 ~ 할 수 있을까?</p>
                                </div>
                            </div>
                        </div>

                        <div class="footer">
                            <div class="footer-left">프로그램 참가자 설문조사 분석 대시보드</div>
                            <div class="footer-right">설문 분석 · Survey Analysis Report</div>
                        </div>
                    </div>
                </div>
                </body>
                </html>
                """
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config={'temperature': 0.0}
                )

                # 결과물을 세션 상태에 저장합니다. (마크다운 백틱 제거)
                if not getattr(response, "text", None):
                    raise ValueError("Gemini 응답이 비어 있습니다.")
                st.session_state.html_content = sanitize_generated_html(response.text)
                
        # 생성된 대시보드가 세션 상태에 존재하면 화면에 출력
        if st.session_state.html_content is not None:
            st.subheader("2. 감정 신호 분석 기반 공감 맵 & 문제 정의 대시보드")
            
            # 다운로드 버튼 추가
            st.download_button(
                label="📥 대시보드 다운로드 (HTML)",
                data=st.session_state.html_content,
                file_name="survey_dashboard_with_hmw.html",
                mime="text/html"
            )
            
            # 화면 표시 유지 (내용이 길어졌으므로 height를 1500으로 상향 조정)
            components.html(st.session_state.html_content, height=1500, scrolling=True)
            
    except pd.errors.EmptyDataError:
        st.error("업로드한 파일에 데이터가 없습니다.")
    except (UnicodeDecodeError, ValueError) as e:
        st.error(f"입력 또는 생성 결과를 확인해 주세요: {e}")
    except Exception as e:
        st.error(f"처리 중 오류가 발생했습니다: {e}")
