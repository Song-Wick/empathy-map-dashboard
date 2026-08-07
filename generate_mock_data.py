import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate mock data
n = 50
gender = np.random.choice(['남성', '여성'], size=n, p=[0.4, 0.6])
age = np.random.choice(['20대', '30대', '40대', '50대'], size=n, p=[0.5, 0.3, 0.15, 0.05])
role = np.random.choice(['기획자', '개발자', '디자이너', '마케터'], size=n, p=[0.25, 0.35, 0.20, 0.20])

satisfaction = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.05, 0.15, 0.45, 0.30])
recommendation = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.02, 0.08, 0.20, 0.40, 0.30])

good_feedback = [
    "팀원들과의 협업이 매끄럽게 진행되어 매우 즐거웠습니다. 소통이 원활했습니다.",
    "실무 중심의 프로젝트 기획 과정이 큰 도움이 되었습니다. 일정 관리도 체계적이었습니다.",
    "네트워킹 시간이 유익했고, 다양한 사람들과 피드백을 주고받을 수 있어서 좋았습니다.",
    "개발 단계에서 직접 배운 기술을 적용할 수 있어서 성취감이 높았습니다.",
    "체계적인 조율 과정 덕분에 마찰 없이 마칠 수 있었습니다.",
    "기획과 마케팅의 연결고리를 확실히 알 수 있는 유익한 시간이었습니다.",
    "소통 채널이 잘 마련되어 의견 조율이 쉬웠습니다.",
    "역할 분담이 공평하게 이루어져 일정 관리가 편했습니다.",
    "난이도 조절이 잘 되어서 초보자도 쉽게 따라갈 수 있었습니다.",
    "서로 격려하고 의견 대립을 건강하게 해결하는 협업 문화가 감명 깊었습니다."
]

bad_feedback = [
    "일정 관리가 조금 타이트해서 마감 직전에 협업에 약간의 마찰이 있었습니다.",
    "난이도 조절이 조금 어려웠습니다. 초반 기획 단계에 시간이 너무 많이 걸렸어요.",
    "소통에 약간의 의견 대립이 있었지만, 잘 해결되었습니다.",
    "역할 분담 과정에서 조금 애매한 부분이 있어 일정 조율이 지연되었습니다.",
    "프로그램 진행 시간이 조금 길어서 체력적으로 소통에 한계가 느껴졌습니다.",
    "초반에 기획 조율하는 데 너무 많은 의견 대립이 생겨 시간 낭비가 있었습니다.",
    "난이도가 생각보다 높아서 개발 일정을 조율하기 벅찼습니다.",
    "비대면 협업 시 소통의 즉각성이 떨어져 일정 관리에 애로사항이 있었습니다.",
    "역할 분담에 명확한 가이드가 부족해서 초기에 혼란스러웠습니다.",
    "팀 간 피드백 조율 과정에서 다소 피로감을 느꼈습니다."
]

good_responses = np.random.choice(good_feedback, size=n)
bad_responses = np.random.choice(bad_feedback, size=n)

# 1. Objective / Demographic Data
df_objective = pd.DataFrame({
    '성별': gender,
    '연령대': age,
    '직무 역할': role,
    '전반적 만족도': satisfaction,
    '타인 추천의사': recommendation
})

# 2. Subjective / Text Feedback Data
df_subjective = pd.DataFrame({
    '좋았던 점 (주관식)': good_responses,
    '아쉬웠던 점 (주관식)': bad_responses
})

# Save to CSV files with UTF-8-SIG for proper Korean display in Excel
df_objective.to_csv('mock_objective_data.csv', index=False, encoding='utf-8-sig')
df_subjective.to_csv('mock_subjective_data.csv', index=False, encoding='utf-8-sig')

print("Mock objective and subjective data generated successfully!")
