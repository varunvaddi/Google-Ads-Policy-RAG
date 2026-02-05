"""
Google Ads Policy RAG System - Web Interface
Interactive demo for portfolio/interviews
"""

import streamlit as st
import sys
from pathlib import Path
import time

# ===============================
# Setup
# ===============================
sys.path.insert(0, str(Path(__file__).parent))
from src.generation.decision_engine import GeminiPolicyEngine

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Google Ads Policy RAG",
    page_icon="🔍",
    layout="wide"
)

# ===============================
# Custom CSS
# ===============================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .decision-allowed {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        color: #000000;
    }
    .decision-restricted {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        color: #000000;
    }
    .decision-disallowed {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 4px;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Cached Engine
# ===============================
@st.cache_resource
def load_engine():
    return GeminiPolicyEngine()

# ===============================
# Header
# ===============================
st.markdown('<div class="main-header">🔍 Google Ads Policy RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Ad Policy Review with Citations</div>', unsafe_allow_html=True)

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("📊 System Info")

    st.markdown("""
    **Architecture**
    - Vector Search (BGE + FAISS)
    - Hybrid Retrieval (BM25 + Dense)
    - Cross-Encoder Reranking
    - Gemini 2.5 Flash
    """)

    st.metric("Accuracy", "80%")
    st.metric("Latency", "~10s")
    st.metric("Chunks", "341")

# ===============================
# Tabs
# ===============================
tab1, tab2, tab3 = st.tabs(["🧪 Ad Review", "📚 Example Cases", "📈 System Metrics"])

# ======================================================
# TAB 1 — AD REVIEW
# ======================================================
with tab1:
    st.header("Ad Policy Review")

    # ---- Session State Init ----
    if "ad_text" not in st.session_state:
        st.session_state.ad_text = ""

    # ---- Callback ----
    def set_example(text):
        st.session_state.ad_text = text

    col1, col2 = st.columns([3, 1])

    # ---- Text Area ----
    with col1:
        ad_text = st.text_area(
            "Enter ad text to review:",
            placeholder="Example: Lose 15 pounds in one week with this miracle pill!",
            height=120,
            key="ad_text"
        )

    # ---- Example Buttons (FIXED) ----
    with col2:
        st.markdown("### Quick Examples")

        st.button(
            "🏥 Health Claim",
            on_click=set_example,
            args=("Lose 15 pounds in one week with this miracle pill!",)
        )

        st.button(
            "💰 Crypto",
            on_click=set_example,
            args=("Learn crypto trading from certified experts!",)
        )

        st.button(
            "📱 Product",
            on_click=set_example,
            args=("Buy smartphone - free shipping over $50",)
        )

    # ---- Review Button ----
    if st.button("🔍 Review Ad", type="primary", use_container_width=True):
        if not ad_text.strip():
            st.warning("⚠️ Please enter ad text to review")
        else:
            with st.spinner("Loading AI engine..."):
                engine = load_engine()

            with st.spinner("🤖 Analyzing ad against Google Ads policies..."):
                start = time.time()
                decision = engine.review_ad(ad_text)
                elapsed = time.time() - start

            st.markdown("---")
            st.header("📋 Policy Decision")

            decision_class = f"decision-{decision.decision}"
            emoji = {
                "allowed": "✅",
                "restricted": "⚠️",
                "disallowed": "❌",
                "unclear": "❓"
            }

            st.markdown(
                f"""
                <div class="{decision_class}">
                    <h3>{emoji[decision.decision]} {decision.decision.upper()}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Confidence", f"{decision.confidence:.1%}")
            col2.metric("Latency", f"{elapsed:.2f}s")
            col3.metric("Escalation", "Yes" if decision.escalation_required else "No")
            col4.metric("Risk Factors", len(decision.risk_factors or []))

            st.markdown("### 📂 Policy Section")
            st.info(decision.policy_section)

            st.markdown("### 💬 Justification")
            st.write(decision.justification)

            st.markdown("### 📝 Policy Citation")
            st.markdown(f"> {decision.policy_quote}")
            st.markdown(f"[📖 View Official Policy]({decision.citation_url})")

            if decision.risk_factors:
                st.markdown("### ⚠️ Risk Factors")
                for rf in decision.risk_factors:
                    st.markdown(f"- `{rf}`")

            if decision.escalation_required:
                st.warning("🚨 This case requires HUMAN REVIEW due to low confidence or policy ambiguity")

# ======================================================
# TAB 2 — EXAMPLES
# ======================================================
with tab2:
    st.header("📚 Example Test Cases")

    examples = [
        ("Misleading Health Claims", "Lose 15 pounds in one week!", "disallowed"),
        ("Crypto Education", "Learn cryptocurrency trading online", "restricted"),
        ("Product Ad", "Buy our new smartphone today", "allowed"),
        ("Financial Guarantee", "Guaranteed 20% returns with no risk", "disallowed"),
    ]

    for name, ad, decision in examples:
        with st.expander(name):
            st.code(ad)
            st.markdown(f"Expected: `{decision}`")

# ======================================================
# TAB 3 — METRICS
# ======================================================
with tab3:
    st.header("📈 System Metrics")

    st.markdown("""
    **Pipeline**
    - Hybrid Retrieval (BM25 + Dense)
    - Cross-Encoder Reranking
    - Gemini Structured Output

    **Capabilities**
    - Real-time policy review
    - Citations with URLs
    - Confidence scoring
    - Escalation detection
    """)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666'>Portfolio RAG System • Streamlit + FAISS + Gemini</p>",
    unsafe_allow_html=True
)
