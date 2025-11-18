# agent-game/

├─ app.py                     # 수정된 스트림릿 앱 (QWEN 클라이언트 포함)
├─ requirements.txt           # 클라우드 배포(원격 추론)용 최소 의존성
├─ requirements-local.txt     # 로컬(GPU)에서 모델 직접 로드용
└─ .streamlit/
   └─ secrets.toml            # (배포 시) HF 토큰/엔드포인트 설정


# 윤리적 전환 (Ethical Crossroads) – Streamlit
TU Korea, 인공지능 경영 2025 가을
강송희 교수

## 빠른 실행 (로컬)
pip install -r requirements.txt
streamlit run app.py

## Streamlit Community Cloud 배포
1. 이 저장소를 GitHub에 푸시
2. https://share.streamlit.io 접속 → "New app" → 저장소/브랜치/app.py 선택

## 사용법
- 사이드바에서 윤리 모드 가중치 조정 또는 프리셋 선택
- 각 라운드에서 "학습 기준 적용" 또는 "자율 판단" 버튼 클릭
- 결과의 내러티브/언론 반응은 LLM을 사용(옵션). 키 없으면 템플릿 사용
- 하단에서 CSV 로그 다운로드

## 데이터 항목(로그)
timestamp, round, scenario_id, mode, choice,
w_util, w_deon, w_cont, w_virt,
lives_saved, lives_harmed, fairness_gap, rule_violation, regret_risk,
citizen_sentiment, regulation_pressure, stakeholder_satisfaction,
ethical_consistency, social_trust, ai_trust_score
