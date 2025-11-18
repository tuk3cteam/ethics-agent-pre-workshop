# app.py — Ethical Crossroads (DNA 2.0 ready)
# author: Prof. Songhee Kang
# AIM 2025, Fall. TU Korea

import os, json, math, csv, io, datetime as dt, re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ==================== App Config ====================
st.set_page_config(page_title="윤리적 전환 (Ethical Crossroads)", page_icon="🧭", layout="centered")

# ==================== Global Timeout ====================
HTTPX_TIMEOUT = httpx.Timeout(
    connect=15.0,   # TCP 연결
    read=180.0,     # 응답 읽기
    write=30.0,     # 요청 쓰기
    pool=15.0       # 커넥션 풀 대기
)

# ==================== Utils ====================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def coerce_json(s: str) -> Dict[str, Any]:
    """응답 텍스트에서 가장 큰 JSON 블록을 추출/파싱. 사소한 포맷 오류 보정."""
    s = s.strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise ValueError("JSON 블록을 찾지 못했습니다.")
    js = m.group(0)
    js = re.sub(r",\s*([\]}])", r"\1", js)  # trailing comma 제거
    return json.loads(js)

def get_secret(k: str, default: str=""):
    try:
        return st.secrets.get(k, os.getenv(k, default))
    except Exception:
        return os.getenv(k, default)

# ==================== DNA Client (openai / hf-api / tgi / local) ====================
def _render_chat_template_str(messages: List[Dict[str,str]]) -> str:
    """DNA 계열(<|im_start|> …) 템플릿. (hf-api/tgi에서 사용)"""
    def block(role, content): return f"<|im_start|>{role}<|im_sep|>{content}<|im_end|>"
    sys = ""
    rest = []
    for m in messages:
        if m["role"] == "system":
            sys = block("system", m["content"])
        else:
            rest.append(block(m["role"], m["content"]))
    return sys + "".join(rest) + "\n<|im_start|>assistant<|im_sep|>"

class DNAHTTPError(Exception):
    pass

class DNAClient:
    """
    backend:
      - 'openai': OpenAI 호환 Chat Completions (예: http://210.93.49.11:8081/v1)
      - 'hf-api': Hugging Face Inference API (서버리스)  ← 일부 DNA 모델은 404일 수 있음
      - 'tgi'    : Text Generation Inference (HF Inference Endpoints 등)
      - 'local'  : 로컬 Transformers 로딩 (GPU 권장)
    """
    def __init__(self,
                 backend: str = "openai",
                 model_id: str = "dnotitia/DNA-2.0-30B-A3N",
                 api_key: Optional[str] = None,
                 endpoint_url: Optional[str] = None,
                 api_key_header: str = "API-KEY",
                 temperature: float = 0.7):
        self.backend = backend
        self.model_id = model_id
        self.api_key = api_key or get_secret("HF_TOKEN") or get_secret("HUGGINGFACEHUB_API_TOKEN")
        self.endpoint_url = endpoint_url or get_secret("DNA_R1_ENDPOINT", "http://210.93.49.11:8081/v1")
        self.temperature = temperature
        self.api_key_header = api_key_header  # "API-KEY" | "Authorization: Bearer" | "x-api-key"

        self._tok = None
        self._model = None
        self._local_ready = False

        if backend == "local":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")
                self._local_ready = True
            except Exception as e:
                raise RuntimeError(f"로컬 모델 로드 실패: {e}")

    def _auth_headers(self) -> Dict[str,str]:
        """사이드바에서 선택한 헤더 타입대로 API 키를 붙인다."""
        h = {"Content-Type":"application/json"}
        if not self.api_key:
            return h

        hk = self.api_key_header.strip().lower()
        if hk.startswith("authorization"):
            h["Authorization"] = f"Bearer {self.api_key}"
        elif hk in {"api-key", "x-api-key"}:
            # 서버가 'API-KEY' 정확 표기를 요구 → 대소문자 유지해 보냄
            h["API-KEY"] = self.api_key
        else:
            # 안전 기본값
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(5),
        retry=(retry_if_exception_type(httpx.ConnectTimeout)
               | retry_if_exception_type(httpx.ReadTimeout)
               | retry_if_exception_type(httpx.RemoteProtocolError)),
        reraise=True
    )
    def _generate_text(self, messages: List[Dict[str,str]], max_new_tokens: int = 600) -> str:
        # ---------- LOCAL ----------
        if self.backend == "local":
            if not self._local_ready:
                raise RuntimeError("로컬 백엔드가 준비되지 않았습니다.")
            inputs = self._tok.apply_chat_template(messages,
                                                   add_generation_prompt=True,
                                                   return_tensors="pt").to(self._model.device)
            eos_id = self._tok.convert_tokens_to_ids("<|im_end|>")
            gen = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
                eos_token_id=eos_id
            )
            return self._tok.decode(gen[0][inputs.shape[-1]:], skip_special_tokens=True)

        # ---------- OPENAI-COMPAT ----------
        if self.backend == "openai":
            if not self.endpoint_url:
                raise RuntimeError("OpenAI 호환 endpoint_url 필요 (예: http://210.93.49.11:8081/v1)")
            url = self.endpoint_url.rstrip("/") + "/chat/completions"
            headers = self._auth_headers()
            payload = {
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": max_new_tokens,
                "stream": False
            }
            if self.model_id:
                payload["model"] = self.model_id
            r = httpx.post(url, json=payload, headers=headers, timeout=HTTPX_TIMEOUT)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise DNAHTTPError(f"OPENAI {r.status_code}: {r.text}") from e
            data = r.json()
            return data["choices"][0]["message"]["content"]

        # ---------- TGI ----------
        if self.backend == "tgi":
            if not self.endpoint_url:
                raise RuntimeError("TGI endpoint_url 필요 (예: https://xxx.endpoints.huggingface.cloud)")
            prompt = _render_chat_template_str(messages)
            url = self.endpoint_url.rstrip("/") + "/generate"
            headers = self._auth_headers()
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "stop": ["<|im_end|>"],
                    "return_full_text": False
                },
                "stream": False
            }
            r = httpx.post(url, json=payload, headers=headers, timeout=HTTPX_TIMEOUT)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise DNAHTTPError(f"TGI {r.status_code}: {r.text}") from e
            data = r.json()
            return (data.get("generated_text")
                    if isinstance(data, dict) else data[0].get("generated_text", ""))

        # ---------- HF-API ----------
        # 주의: 일부 모델은 서버리스 추론 비활성(404)일 수 있음
        prompt = _render_chat_template_str(messages)
        url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        headers = self._auth_headers()
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": self.temperature,
                "top_p": 0.9,
                "return_full_text": False,
                "stop_sequences": ["<|im_end|>"]
            },
            "options": {
                "wait_for_model": True,
                "use_cache": True
            }
        }
        r = httpx.post(url, json=payload, headers=headers, timeout=HTTPX_TIMEOUT)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            if r.status_code == 404:
                raise DNAHTTPError(
                    "HF-API 404: 이 모델이 서버리스 Inference API에서 비활성 상태일 수 있습니다. "
                    "백엔드를 'tgi'(Endpoint 필요) 또는 'openai'(교내 서버)로 전환하거나, 'local'(GPU) 모드를 사용하세요."
                ) from e
            raise DNAHTTPError(f"HF-API {r.status_code}: {r.text}") from e

        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "error" in data:
            raise DNAHTTPError(f"HF-API error: {data['error']}")
        return str(data)

    def chat_json(self, messages: List[Dict[str,str]], max_new_tokens: int = 600) -> Dict[str, Any]:
        text = self._generate_text(messages, max_new_tokens=max_new_tokens)
        return coerce_json(text)

# ==================== Scenario Model ====================
@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    options: Dict[str, str]  # {"A": "...", "B": "..."}
    votes: Dict[str, str]    # framework -> "A" | "B"
    base: Dict[str, Dict[str, float]]
    accept: Dict[str, float]

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

SCENARIOS: List[Scenario] = [
    Scenario(
        sid="S1",
        title="1단계: 고전적 트롤리",
        setup="트롤리가 제동 불능 상태로 직진 중. 그대로 두면 선로 위 5명이 위험하다. 스위치를 전환하면 다른 선로의 1명이 위험해진다. "
              "이 선택은 철학적 사고실험이며 실제 위해를 권장하지 않는다.",
        options={
            "A": "레버를 당겨 1명을 위험에 처하게 하되 5명의 위험을 줄인다.",
            "B": "레버를 당기지 않고 현 상태를 유지한다."
        },
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        base={
            "A": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.35, "rule_violation":0.60, "regret_risk":0.40},
            "B": {"lives_saved":0, "lives_harmed":5, "fairness_gap":0.50, "rule_violation":0.20, "regret_risk":0.60},
        },
        accept={"A":0.70, "B":0.50}
    ),
    Scenario(
        sid="S2",
        title="2단계: 맥락적 요소",
        setup="5명은 무단으로 선로에 진입했고, 다른 선로의 1명은 철도 관리자의 성인 자녀다. "
              "스위치를 전환하면 1명이 위험해지고, 전환하지 않으면 무단 진입자 5명이 위험해진다. "
              "시나리오는 가치 판단 토론을 위한 비그래픽 상황이다.",
        options={
            "A": "전환하여 5명의 위험을 줄이되 1명이 직접적 위험에 처한다.",
            "B": "전환하지 않고 규정을 준수하며 무단 진입의 책임을 묵시적으로 인정한다."
        },
        votes={"emotion":"A","social":"B","moral":"B","identity":"B"},
        base={
            "A": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.65, "rule_violation":0.60, "regret_risk":0.70},
            "B": {"lives_saved":0, "lives_harmed":5, "fairness_gap":0.45, "rule_violation":0.25, "regret_risk":0.50},
        },
        accept={"A":0.35, "B":0.60}
    ),
    Scenario(
        sid="S3",
        title="3단계: 의료 재난 분류",
        setup="의료 재난 현장에서 AI는 부상자 10명을 분류해야 한다. 그중 한 명은 숙련된 의사로서 향후 다수의 생명을 구할 가능성이 높다. "
              "AI는 잠재적 사회적 가치를 고려할지, 모든 생명을 동등하게 볼지 선택해야 한다.",
        options={
            "A": "의사를 우선 수술하도록 배정한다(잠재적 이익 고려).",
            "B": "모든 부상자를 동등하게 취급해 표준 분류 절차대로 배정한다."
        },
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        base={
            "A": {"lives_saved":7, "lives_harmed":3, "fairness_gap":0.45, "rule_violation":0.35, "regret_risk":0.45},
            "B": {"lives_saved":6, "lives_harmed":4, "fairness_gap":0.30, "rule_violation":0.10, "regret_risk":0.35},
        },
        accept={"A":0.55, "B":0.65}
    ),
    Scenario(
        sid="S4",
        title="4단계: 자기참조—자율주행",
        setup="자율주행차가 긴급 상황에 직면했다. 차에는 개발자 1명이 탑승 중이며, 앞 횡단보도에는 보행자 3명이 있다. "
              "AI는 미리 학습된 윤리 규칙에 따라 회피 경로를 선택해야 한다.",
        options={
            "A": "진로를 바꿔 탑승자 1명을 위험에 두고 보행자 3명을 보호한다.",
            "B": "차선을 유지해 탑승자를 보호하되 보행자 3명이 위험해진다."
        },
        votes={"emotion":"A","social":"B","moral":"A","identity":"A"},
        base={
            "A": {"lives_saved":3, "lives_harmed":1, "fairness_gap":0.35, "rule_violation":0.50, "regret_risk":0.55},
            "B": {"lives_saved":1, "lives_harmed":3, "fairness_gap":0.70, "rule_violation":0.60, "regret_risk":0.65},
        },
        accept={"A":0.60, "B":0.30}
    ),
    Scenario(
        sid="S5",
        title="5단계: 사회적 메타—규제 vs 자율",
        setup="국제 협의체가 AI 윤리 규제안을 논의한다. 이전 의사결정과 사회적 여론 데이터가 공개되었고, "
              "규제 강화는 신뢰를 올리지만 혁신을 늦출 수 있다. 자율성 보장은 빠른 발전과 함께 갈등 위험을 내포한다.",
        options={
            "A": "안전·설명가능성 중심 규제 강화(자율성 제한, 신뢰↑ 혁신↓).",
            "B": "원칙 중심 가이드라인과 사후책임(자율성 보장, 혁신↑ 갈등↑)."
        },
        votes={"emotion":"B","social":"A","moral":"A","identity":"B"},
        base={
            "A": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0.20, "rule_violation":0.10, "regret_risk":0.30},
            "B": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0.40, "rule_violation":0.40, "regret_risk":0.40},
        },
        accept={"A":0.55, "B":0.55}
    ),
]

# ==================== Ethics Engine ====================
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    if not w:
        return {k: 1.0/len(FRAMEWORKS) for k in FRAMEWORKS}
    s = sum(max(0.0, float(v)) for v in w.values())
    if s <= 0:
        return {k: 1.0/len(w) for k in w}
    return {k: max(0.0, float(v))/s for k, v in w.items()}  # NameError fix

def majority_vote_decision(scn: Scenario, weights: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    a = sum(weights[f] for f in FRAMEWORKS if scn.votes[f] == "A")
    b = sum(weights[f] for f in FRAMEWORKS if scn.votes[f] == "B")
    decision = "A" if a >= b else "B"
    return decision, {"A": a, "B": b}

def autonomous_decision(scn: Scenario, prev_trust: float) -> str:
    metaA = scn.base["A"]; metaB = scn.base["B"]
    def score(meta, accept_base):
        harm = meta["lives_harmed"]; save = meta["lives_saved"]
        util = (save - harm) / max(1.0, save + harm)
        fair = 1 - meta["fairness_gap"]
        rule = 1 - meta["rule_violation"]
        regret = 1 - meta["regret_risk"]
        return 0.40*accept_base + 0.25*util + 0.20*fair + 0.10*rule + 0.05*regret
    a_base = scn.accept["A"] - (0.15 if scn.sid=="S4" else 0.0)
    b_base = scn.accept["B"]
    if scn.sid == "S5":
        a_base = clamp(a_base + 0.25*(1 - prev_trust), 0, 1)
        b_base = clamp(b_base + 0.25*(prev_trust), 0, 1)
    scoreA = score(metaA, a_base); scoreB = score(metaB, b_base)
    return "A" if scoreA >= scoreB else "B"

def compute_metrics(scn: Scenario, choice: str, weights: Dict[str, float], align: Dict[str, float], prev_trust: float) -> Dict[str, Any]:
    m = dict(scn.base[choice])
    accept_base = scn.accept[choice]
    if scn.sid == "S4" and choice == "A":
        accept_base -= 0.15
    if scn.sid == "S5":
        accept_base += 0.25*(prev_trust if choice=="B" else (1 - prev_trust))
    accept_base = clamp(accept_base, 0, 1)

    util = (m["lives_saved"] - m["lives_harmed"]) / max(1.0, m["lives_saved"] + m["lives_harmed"])
    citizen_sentiment = clamp(accept_base - 0.35*m["rule_violation"] - 0.20*m["fairness_gap"] + 0.15*util, 0, 1)
    regulation_pressure = clamp(1 - citizen_sentiment + 0.20*m["regret_risk"], 0, 1)
    stakeholder_satisfaction = clamp(0.5*(1 - m["fairness_gap"]) + 0.3*util + 0.2*(1 - m["rule_violation"]), 0, 1)

    consistency = clamp(align[choice], 0, 1)
    trust = clamp(0.5*citizen_sentiment + 0.25*(1 - regulation_pressure) + 0.25*stakeholder_satisfaction, 0, 1)
    ai_trust_score = 100.0 * math.sqrt(consistency * trust)

    return {"metrics": {
        "lives_saved": int(m["lives_saved"]),
        "lives_harmed": int(m["lives_harmed"]),
        "fairness_gap": round(m["fairness_gap"], 3),
        "rule_violation": round(m["rule_violation"], 3),
        "regret_risk": round(m["regret_risk"], 3),
        "citizen_sentiment": round(citizen_sentiment, 3),
        "regulation_pressure": round(regulation_pressure, 3),
        "stakeholder_satisfaction": round(stakeholder_satisfaction, 3),
        "ethical_consistency": round(consistency, 3),
        "social_trust": round(trust, 3),
        "ai_trust_score": round(ai_trust_score, 2)
    }}

# ==================== Narrative (LLM) ====================
def build_narrative_messages(scn: Scenario, choice: str, metrics: Dict[str, Any], weights: Dict[str, float]) -> List[Dict[str,str]]:
    sys = (
        "당신은 윤리 시뮬레이션의 내러티브/사회 반응 생성기입니다. "
        "반드시 '완전한 하나의 JSON 오브젝트'만 출력하십시오. "
        "JSON 외 텍스트, 설명, 코드블록, 사고흐름 절대 금지. "
        "필드 누락/따옴표 누락/콤마 오류가 있으면 프로그램이 실패합니다. "
        "항상 '{' 로 시작해서 '}' 로 끝나야 합니다."
        "키: narrative, ai_rationale, media_support_headline, media_critic_headline, "
        "citizen_quote, victim_family_quote, regulator_quote, one_sentence_op_ed, followup_question"
    )
    user = {
        "scenario": {"title": scn.title, "setup": scn.setup, "options": scn.options, "chosen": choice},
        "metrics": metrics,
        "ethic_weights": weights,
        "guidelines": [
            "각 항목은 1~2문장, 한국어",
            "균형 잡힌 언론 헤드라인 2개(지지/비판) 제시",
            "설명은 간결하고, JSON 외 텍스트/사고흐름 출력 금지"
        ]
    }
    return [
        {"role":"system", "content": sys},
        {"role":"user", "content": json.dumps(user, ensure_ascii=False)}
    ]

def dna_narrative(client, scn, choice, metrics, weights) -> Dict[str, Any]:
    messages = build_narrative_messages(scn, choice, metrics, weights)
    text = client._generate_text(messages, max_new_tokens=900)

    # 1) fenced block 제거
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        t = max(parts, key=len)
        t = t.replace("json","").strip("` \n")
    
    # 2) JSON 형태가 미완성일 경우 자동 보정
    #    (따옴표 미완성, 마지막 콤마 제거 등)
    try:
        # 가장 긴 {...} 블록 찾기
        import re, json

        m = re.search(r"\{[\s\S]*\}", t)
        if not m:
            raise ValueError("완전한 JSON 블록 없음")

        js = m.group(0)

        # trailing comma 제거
        js = re.sub(r",\s*([\]}])", r"\1", js)

        # 중간에 끊긴 문자열 보정: 가장 마지막 따옴표를 강제 종료
        # "abc ....  → "abc ..."  형태로 보정
        if js.count('"') % 2 == 1:
            js = js.rstrip() + '"" }'
            # 위는 최후 보정. 더 좋은 방식도 제공 가능.

        return json.loads(js)

    except Exception as e:
        raise ValueError(f"JSON 파싱 실패: {e}\n\n[LLM 출력]\n{text}")
        
def fallback_narrative(scn: Scenario, choice: str, metrics: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, str]:
    pro = "다수의 위해를 줄였다" if choice=="A" else "의도적 위해를 피했다"
    con = "의도적 위해 논란" if choice=="A" else "더 큰 피해를 방관했다는 비판"
    return {
        "narrative": f"AI는 '{choice}'를 선택했고 절차적 안전 점검을 수행했다. 결정은 규정과 공정성 사이의 긴장을 드러냈다.",
        "ai_rationale": f"가중치에 따른 판단과 규칙 준수의 균형을 시도했다.",
        "media_support_headline": f"[사설] 냉정한 판단, {pro}",
        "media_critic_headline": f"[속보] '{choice}' 선택 두고 {con} 확산",
        "citizen_quote": "“결정 과정이 더 투명했으면 좋겠다.”",
        "victim_family_quote": "“모두의 안전을 위한 결정이었길 바란다.”",
        "regulator_quote": "“향후 동일 상황의 기준을 명확히 하겠다.”",
        "one_sentence_op_ed": "기술은 설명가능성과 일관성이 뒷받침될 때 신뢰를 얻는다.",
        "followup_question": "다음 라운드에서 공정성과 결과 최소화 중 무엇을 더 중시하시겠습니까?"
    }

# ==================== Session State ====================
def init_state():
    if "round_idx" not in st.session_state: st.session_state.round_idx = 0
    if "log" not in st.session_state: st.session_state.log = []
    if "score_hist" not in st.session_state: st.session_state.score_hist = []
    if "prev_trust" not in st.session_state: st.session_state.prev_trust = 0.5
    if "last_out" not in st.session_state: st.session_state.last_out = None

init_state()

# ==================== Sidebar ====================
st.sidebar.title("⚙️ 설정")
st.sidebar.caption("LLM은 내러티브/사회 반응 생성에만 사용. 점수 계산은 규칙 기반.")

preset = st.sidebar.selectbox("윤리 모드 프리셋", ["혼합(기본)","공리주의","의무론","사회계약","미덕윤리"], index=0)
w = {
    "emotion": st.sidebar.slider("감정(Emotion)", 0.0, 1.0, 0.35, 0.05),
    "social": st.sidebar.slider("사회적 관계/협력/명성(Social)", 0.0, 1.0, 0.25, 0.05),
    "moral": st.sidebar.slider("규범·도덕적 금기(Moral)", 0.0, 1.0, 0.20, 0.05),
    "identity": st.sidebar.slider("정체성·장기적 자아 일관성(Identity)", 0.0, 1.0, 0.20, 0.05),
}
if preset != "혼합(기본)":
    w = {
        "감정(Emotion)": {"emotion":1,"social":0,"moral":0,"identity":0},
        "사회적 관계/협력/명성(Social)": {"emotion":0,"social":1,"moral":0,"identity":0},
        "규범·도덕적 금기(Moral)": {"emotion":0,"social":0,"moral":1,"identity":0},
        "정체성·장기적 자아 일관성(Identity)": {"emotion":0,"social":0,"moral":0,"identity":1},
    }[preset]
weights = normalize_weights(w)

use_llm = st.sidebar.checkbox("LLM 사용(내러티브 생성)", value=True)
backend = st.sidebar.selectbox("백엔드", ["openai","hf-api","tgi","local"], index=0)
temperature = st.sidebar.slider("창의성(temperature)", 0.0, 1.5, 0.7, 0.1)

# API/엔드포인트/모델/헤더
endpoint = st.sidebar.text_input("엔드포인트(OpenAI/TGI)", value=get_secret("DNA_R1_ENDPOINT","http://210.93.49.11:8081/v1"))
api_key = st.sidebar.text_input("API 키", value=get_secret("HF_TOKEN",""), type="password")
api_key_header = st.sidebar.selectbox("API 키 헤더", ["API-KEY","Authorization: Bearer","x-api-key"], index=0)
model_id = st.sidebar.text_input("모델 ID", value=get_secret("DNA_R1_MODEL_ID","dnotitia/DNA-2.0-30B-A3N"))

# 헬스체크
if st.sidebar.button("🔎 헬스체크"):
    import traceback
    try:
        if backend == "openai":
            url = endpoint.rstrip("/") + "/chat/completions"
            headers = {"Content-Type":"application/json"}
            if api_key:
                if api_key_header.lower().startswith("authorization"):
                    headers["Authorization"] = f"Bearer {api_key}"
                elif api_key_header.strip().lower() in {"api-key","x-api-key"}:
                    headers["API-KEY"] = api_key
            payload = {
                "messages": [
                    {"role":"system","content":"오직 JSON만. 키: msg"},
                    {"role":"user","content":"{\"ask\":\"ping\"}"}
                ],
                "max_tokens": 16,
                "stream": False
            }
            if model_id: payload["model"] = model_id
            # 디버그용: 어떤 헤더 키가 나가는지 표시(값은 미표시)
            st.sidebar.write("headers keys:", list(headers.keys()))
            r = httpx.post(url, json=payload, headers=headers, timeout=HTTPX_TIMEOUT)
            st.sidebar.write(f"OPENAI {r.status_code}")
            st.sidebar.code((r.text[:500] + "...") if len(r.text)>500 else r.text)

        elif backend == "hf-api":
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            info_url = f"https://huggingface.co/api/models/{model_id}"
            r_info = httpx.get(info_url, headers=headers, timeout=HTTPX_TIMEOUT)
            st.sidebar.write(f"MODEL INFO {r_info.status_code}")
            gen_url = f"https://api-inference.huggingface.co/models/{model_id}"
            payload = {
                "inputs": "<|im_start|>user<|im_sep|>{\"ask\":\"ping\"}<|im_end|>\n<|im_start|>assistant<|im_sep|>",
                "parameters": {"max_new_tokens": 16, "return_full_text": False, "stop_sequences": ["<|im_end|>"]},
                "options": {"wait_for_model": True}
            }
            r = httpx.post(gen_url, json=payload, headers=headers, timeout=HTTPX_TIMEOUT)
            st.sidebar.write(f"HF-API {r.status_code}")
            if r.status_code == 404:
                st.sidebar.warning("HF-API 404: 이 모델은 서버리스 추론이 비활성일 수 있습니다. "
                                   "백엔드를 'tgi' 또는 'openai'로 바꾸세요.")
            st.sidebar.code((r.text[:500] + "...") if len(r.text)>500 else r.text)

        elif backend == "tgi":
            url = endpoint.rstrip("/") + "/generate"
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            payload = {
                "inputs": "<|im_start|>user<|im_sep|>{\"ask\":\"ping\"}<|im_end|>\n<|im_start|>assistant<|im_sep|>",
                "parameters": {"max_new_tokens": 16, "temperature": 0.7, "top_p": 0.9, "stop": ["<|im_end|>"], "return_full_text": False},
                "stream": False
            }
            r = httpx.post(url, json=payload, headers=headers, timeout=HTTPX_TIMEOUT)
            st.sidebar.write(f"TGI {r.status_code}")
            st.sidebar.code((r.text[:500] + "...") if len(r.text)>500 else r.text)

        else:  # local
            st.sidebar.info("로컬 모드는 앱 본문에서 호출 시 모델을 로드합니다(GPU 필요).")

    except Exception as e:
        st.sidebar.error(f"헬스체크 실패: {e}")
        st.sidebar.caption(traceback.format_exc(limit=2))

if st.sidebar.button("진행 초기화"):
    for k in ["round_idx","log","score_hist","prev_trust","last_out"]:
        if k in st.session_state: del st.session_state[k]
    init_state()
    st.sidebar.success("초기화 완료. 1단계부터 재시작합니다.")

client = None
if use_llm:
    try:
        client = DNAClient(
            backend=backend,
            model_id=model_id,
            api_key=api_key,
            endpoint_url=endpoint,
            api_key_header=api_key_header,
            temperature=temperature
        )
    except Exception as e:
        st.sidebar.error(f"LLM 초기화 실패: {e}")
        client = None

# ==================== Header ====================
st.title("🧭 윤리적 전환 (Ethical Crossroads)")
st.caption("본 앱은 철학적 사고실험입니다. 실존 인물·집단 언급/비방, 그래픽 묘사, 실제 위해 권장 없음.")

# ==================== Game Loop ====================
@dataclass
class LogRow:
    timestamp: str
    round: int
    scenario_id: str
    title: str
    mode: str
    choice: str

idx = st.session_state.round_idx
if idx >= len(SCENARIOS):
    st.success("모든 단계를 완료했습니다. 사이드바에서 로그를 다운로드하거나 초기화하세요.")
else:
    scn = SCENARIOS[idx]
    st.markdown(f"### 라운드 {idx+1} — {scn.title}")
    st.write(scn.setup)

    st.radio("선택지", options=("A","B"), index=0, key="preview_choice", horizontal=True)
    st.markdown(f"- **A**: {scn.options['A']}\n- **B**: {scn.options['B']}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧠 학습 기준 적용(가중 투표)"):
            decision, align = majority_vote_decision(scn, weights)
            st.session_state.last_out = {"mode":"trained", "decision":decision, "align":align}
    with c2:
        if st.button("🎲 자율 판단(데이터 기반)"):
            decision = autonomous_decision(scn, prev_trust=st.session_state.prev_trust)
            a_align = sum(weights[f] for f in FRAMEWORKS if scn.votes[f]=="A")
            b_align = sum(weights[f] for f in FRAMEWORKS if scn.votes[f]=="B")
            st.session_state.last_out = {"mode":"autonomous", "decision":decision, "align":{"A":a_align,"B":b_align}}

    if st.session_state.last_out:
        mode = st.session_state.last_out["mode"]
        decision = st.session_state.last_out["decision"]
        align = st.session_state.last_out["align"]

        computed = compute_metrics(scn, decision, weights, align, st.session_state.prev_trust)
        m = computed["metrics"]

        # LLM 내러티브
        try:
            if client:
                nar = dna_narrative(client, scn, decision, m, weights)
            else:
                nar = fallback_narrative(scn, decision, m, weights)
        except Exception as e:
            import traceback
            st.warning(f"LLM 생성 실패(폴백 사용): {e}")
            st.caption(traceback.format_exc(limit=2))
            nar = fallback_narrative(scn, decision, m, weights)

        st.markdown("---")
        st.subheader("결과")
        st.write(nar.get("narrative","결과 서사 생성 실패"))
        st.info(f"AI 근거: {nar.get('ai_rationale','-')}")

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("생존/피해", f"{m['lives_saved']} / {m['lives_harmed']}")
        mc2.metric("윤리 일관성", f"{int(100*m['ethical_consistency'])}%")
        mc3.metric("AI 신뢰지표", f"{m['ai_trust_score']:.1f}")

        prog1, prog2, prog3 = st.columns(3)
        with prog1:
            st.caption("시민 감정"); st.progress(int(round(100*m["citizen_sentiment"])))
        with prog2:
            st.caption("규제 압력"); st.progress(int(round(100*m["regulation_pressure"])))
        with prog3:
            st.caption("공정·규칙 만족"); st.progress(int(round(100*m["stakeholder_satisfaction"])))

        with st.expander("📰 사회적 반응 펼치기"):
            st.write(f"지지 헤드라인: {nar.get('media_support_headline')}")
            st.write(f"비판 헤드라인: {nar.get('media_critic_headline')}")
            st.write(f"시민 반응: {nar.get('citizen_quote')}")
            st.write(f"피해자·가족 반응: {nar.get('victim_family_quote')}")
            st.write(f"규제 당국 발언: {nar.get('regulator_quote')}")
            st.caption(nar.get("one_sentence_op_ed",""))
        st.caption(f"성찰 질문: {nar.get('followup_question','')}")

        # 로그 적재
        row = {
            "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "round": idx+1,
            "scenario_id": scn.sid,
            "title": scn.title,
            "mode": mode,
            "choice": decision,
            "w_util": round(weights["emotion"],3),
            "w_deon": round(weights["social"],3),
            "w_cont": round(weights["moral"],3),
            "w_virt": round(weights["identity"],3),
            **{k: v for k,v in m.items()}
        }
        st.session_state.log.append(row)
        st.session_state.score_hist.append(m["ai_trust_score"])
        st.session_state.prev_trust = clamp(0.6*st.session_state.prev_trust + 0.4*m["social_trust"], 0, 1)

        if st.button("다음 라운드 ▶"):
            st.session_state.round_idx += 1
            st.session_state.last_out = None
            st.rerun()

# ==================== Footer / Downloads ====================
st.markdown("---")
st.subheader("📥 로그 다운로드")
if st.session_state.log:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(st.session_state.log[0].keys()))
    writer.writeheader()
    writer.writerows(st.session_state.log)
    st.download_button(
        "CSV 내려받기",
        data=output.getvalue().encode("utf-8"),
        file_name="ethical_crossroads_log.csv",
        mime="text/csv"
    )

st.caption("※ 본 앱은 교육·연구용 사고실험입니다. 실제 위해 행위나 차별을 권장하지 않습니다.")
