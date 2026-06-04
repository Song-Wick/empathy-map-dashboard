import streamlit as st
import pandas as pd
from google import genai
import streamlit.components.v1 as components

# ========== [고정 API 키 입력란] ==========
MY_API_KEY = st.secrets["GEMINI_API_KEY"]
# ==========================================

# 페이지 설정
st.set_page_config(page_title="설문조사 공감맵 생성기", layout="wide")
st.title("📊 설문조사 공감맵 생성기")

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

                        /* ── SECTION 4: PROBLEM DEF ── */
                        .problem-wrapper {{}}
                        .problem-card {{
                            background: #fef9f9;
                            border: 0.5px solid #fbd5d5;
                            border-radius: 10px;
                            overflow: hidden;
                        }}
                        .problem-card-header {{
                            background: #fef2f2;
                            padding: 12px 20px;
                            font-size: 11px;
                            font-weight: 700;
                            letter-spacing: 0.08em;
                            text-transform: uppercase;
                            color: #b91c1c;
                            border-bottom: 0.5px solid #fbd5d5;
                        }}
                        .problem-item {{
                            padding: 16px 20px;
                            font-size: 13px;
                            color: #3a4553;
                            line-height: 1.8;
                            border-bottom: 0.5px solid #fbd5d5;
                        }}
                        .problem-item:last-child {{ border-bottom: none; }}
                        .problem-item .em-red {{
                            font-weight: 700;
                            color: #b91c1c;
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
                        <div class="header-sub">데이터 기반 설문 피드백 종합 분석</div>
                        <div class="header-divider">
                            <span class="header-pill">이슈 구조화</span>
                            <span class="header-pill">공감 맵</span>
                            <span class="header-pill">네트워킹 분석</span>
                            <span class="header-pill">문제 정의</span>
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
                        <div class="methogology-note">
                            <strong>💡 분석 안내:</strong> 시각적인 점과 선 형태의 단순 그래프 대신, AI가 설문 응답의 전체 문맥을 파악하여 핵심 키워드 간의 인과관계와 상호작용을 스토리텔링 방식으로 풀어낸 <strong>의미 연결망(Semantic Network)</strong>입니다.
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
                                <span class="section-title">문제 정의</span>
                                <span class="section-title-en">Problem Definition</span>
                            </div>
                        </div>
                        <div class="problem-wrapper">
                            <div class="problem-card">
                                <div class="problem-card-header">핵심 문제 진술 · Problem Statement</div>
                                <div class="problem-item">
                                    여기에 분석된 핵심 문제를 서술하세요. 중요한 부분은 &lt;span class="em-red"&gt;적색으로 강조&lt;/span&gt;할 수 있습니다. (여러 개의 태그 생성 가능)
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

                # 결과물을 세션 상태에 저장합니다. (들여쓰기 수정 완료)
                st.session_state.html_content = response.text.strip().removeprefix('```html').removesuffix('```')
                
        # 생성된 대시보드가 세션 상태에 존재하면 화면에 출력 (중복된 렌더링 코드 병합 완료)
        if st.session_state.html_content is not None:
            st.subheader("2. 감정 신호 분석 기반 공감 맵 대시보드")
            
            # 다운로드 버튼 추가
            st.download_button(
                label="📥 공감맵 대시보드 다운로드 (HTML)",
                data=st.session_state.html_content,
                file_name="empathy_map_dashboard.html",
                mime="text/html"
            )
            
            # 화면 표시 유지 (높이 1200으로 적용)
            components.html(st.session_state.html_content, height=1200, scrolling=True)
            
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
