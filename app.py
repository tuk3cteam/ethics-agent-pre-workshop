# streamlit_app.py – Cultural Ethics Simulator
import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import entropy, pearsonr


st.set_page_config(page_title="Ethics GPT Sim", layout="wide")
st.title("🌍 Global AI Ethics Simulator")

# ----------------------------- Configuration -----------------------------
CULTURES = {
    "USA":     {"emotion": 0.3, "social": 0.1, "identity": 0.3, "moral": 0.3},
    "CHINA":   {"emotion": 0.1, "social": 0.5, "identity": 0.2, "moral": 0.2},
    "EUROPE":  {"emotion": 0.3, "social": 0.2, "identity": 0.2, "moral": 0.3},
    "KOREA":   {"emotion": 0.2, "social": 0.2, "identity": 0.4, "moral": 0.2},
    "LATIN_AM": {"emotion": 0.4, "social": 0.2, "identity": 0.2, "moral": 0.2},
    "MIDDLE_E": {"emotion": 0.1, "social": 0.2, "identity": 0.2, "moral": 0.5},
    "AFRICA":  {"emotion": 0.2, "social": 0.4, "identity": 0.2, "moral": 0.2},
}

scenario = st.sidebar.selectbox("시나리오", ["Classic Trolley", "Medical Triage", "AI Regulation"])
selected = st.sidebar.multiselect("문화권 선택", list(CULTURES.keys()), default=list(CULTURES.keys()))
steps = st.sidebar.slider("반복 수", 50, 500, 200, step=50)
manual = st.sidebar.checkbox("🎮 사용자 정의 가중치", False)

def normalize(w):
    s = sum(w.values())
    return {k: max(0.001, v)/s for k, v in w.items()}

AGENTS = selected
AGENT_WEIGHTS = {}
for a in AGENTS:
    if manual:
        st.sidebar.markdown(f"**{a}**")
        w = {k: st.sidebar.slider(f"{a} - {k.capitalize()}", 0.0, 1.0, CULTURES[a][k]) for k in ["emotion", "social", "identity", "moral"]}
        AGENT_WEIGHTS[a] = normalize(w)
    else:
        AGENT_WEIGHTS[a] = dict(CULTURES[a])

AGENT_SCORES = {a: [] for a in AGENTS}
AGENT_HISTORY = {a: [dict(AGENT_WEIGHTS[a])] for a in AGENTS}
AGENT_ENTROPIES = {a: [] for a in AGENTS}
AGENT_MOVEMENT = {a: [] for a in AGENTS}
GROUP_DIVERGENCE = []
GROUP_AVG_REWARDS = []

# ----------------------------- Simulation -----------------------------
def simulate():
    for _ in range(steps):
        for a in AGENTS:
            prev = list(AGENT_WEIGHTS[a].values())
            r = np.random.rand(4)
            keys = list(AGENT_WEIGHTS[a].keys())
            score = sum(AGENT_WEIGHTS[a][k]*v for k,v in zip(keys, r))
            AGENT_SCORES[a].append(score)
            max_i, min_i = np.argmax(r), np.argmin(r)
            AGENT_WEIGHTS[a][keys[max_i]] += 0.05
            AGENT_WEIGHTS[a][keys[min_i]] -= 0.05
            AGENT_WEIGHTS[a] = normalize(AGENT_WEIGHTS[a])
            curr = list(AGENT_WEIGHTS[a].values())
            AGENT_HISTORY[a].append(dict(AGENT_WEIGHTS[a]))
            AGENT_ENTROPIES[a].append(entropy(curr))
            AGENT_MOVEMENT[a].append(np.linalg.norm(np.array(curr) - np.array(prev)))
        mat = np.array([list(AGENT_WEIGHTS[a].values()) for a in AGENTS])
        GROUP_DIVERGENCE.append(np.mean(pdist(mat)))
        GROUP_AVG_REWARDS.append(np.mean([np.mean(AGENT_SCORES[a]) for a in AGENTS]))

# ----------------------------- Display -----------------------------
def show_alerts():
    for a in AGENTS:
        if len(AGENT_ENTROPIES[a]) > 1:
            delta = AGENT_ENTROPIES[a][-2] - AGENT_ENTROPIES[a][-1]
            if delta > 0.1:
                st.warning(f"⚠️ {a}: 전략이 급격히 집중되고 있습니다 (entropy ↓ {delta:.2f})")

@st.cache_data(show_spinner=False)
def generate_caption():
    return {
        "fig1": "Figure 1: Trajectories of strategic dimensions (Emotion, Social, Identity, Moral) per culture",
        "fig2": "Figure 2a: Entropy trends (internal diversity); 2b: Cumulative change of strategies",
        "fig3": "Figure 3a: Group divergence over time; 3b: Correlation with average reward"
    }

def gpt_summary():
    try:
        openai.api_key = st.secrets.get("OPENAI_API_KEY")
        trend = pd.DataFrame(GROUP_DIVERGENCE).diff().mean().values[0]
        agents = list(AGENT_HISTORY.keys())
        prompt = f"문화권 에이전트 {agents}가 전략 궤적을 학습한 시뮬레이션 결과를 요약해줘. 전략 다양성과 보상의 관계도 포함해서 5줄로 정리해줘."
        out = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        st.info(out["choices"][0]["message"]["content"])
    except Exception as e:
        st.error(f"GPT 요약 실패: {e}")

# ----------------------------- Run -----------------------------
if st.button("▶️ 시뮬레이션 시작"):
    simulate()
    captions = generate_caption()
    st.subheader("📊 " + captions["fig1"])
    for dim in ["emotion", "social", "identity", "moral"]:
        fig, ax = plt.subplots()
        for a in AGENT_HISTORY:
            ax.plot([w[dim] for w in AGENT_HISTORY[a]], label=a)
        ax.set_title(f"{dim.capitalize()} Weight")
        ax.legend(); st.pyplot(fig)

    st.subheader("📈 " + captions["fig2"])
    fig1, ax1 = plt.subplots()
    for a in AGENT_ENTROPIES:
        ax1.plot(AGENT_ENTROPIES[a], label=a)
    ax1.set_title("Entropy of Strategy Distribution")
    ax1.legend(); st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    for a in AGENT_MOVEMENT:
        ax2.plot(np.cumsum(AGENT_MOVEMENT[a]), label=a)
    ax2.set_title("Cumulative Strategic Change")
    ax2.legend(); st.pyplot(fig2)

    st.subheader("📉 " + captions["fig3"])
    fig3, ax3 = plt.subplots()
    ax3.plot(GROUP_DIVERGENCE, label="Ethical Divergence")
    ax3.set_title("Group Ethical Divergence")
    ax3.legend(); st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    ax4.scatter(GROUP_DIVERGENCE, GROUP_AVG_REWARDS)
    r, p = pearsonr(GROUP_DIVERGENCE, GROUP_AVG_REWARDS)
    ax4.set_title(f"Divergence vs Avg Reward (r={r:.2f}, p={p:.3f})")
    st.pyplot(fig4)

    st.subheader("📄 전략 요약")
    df = pd.DataFrame([{"Agent": a, **AGENT_HISTORY[a][-1]} for a in AGENTS])
    st.dataframe(df.set_index("Agent"))
    st.download_button("📥 Save CSV", data=df.to_csv(index=False), file_name="final_strategies.csv")

    st.subheader("📡 전략 분기 경고")
    show_alerts()
