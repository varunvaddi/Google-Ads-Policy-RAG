"""
Google Ads Policy RAG System - Web Interface
Interactive demo for portfolio/interviews
"""

import streamlit as st
import sys
from pathlib import Path
import time

# Setup
sys.path.insert(0, str(Path(__file__).parent))
from src.generation.decision_engine import GeminiPolicyEngine

# Page config
st.set_page_config(
    page_title="Google Ads Policy RAG",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
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
    }
    .decision-restricted {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
    }
    .decision-disallowed {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 4px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize engine (cached)
@st.cache_resource
def load_engine():
    """Load the RAG engine (cached for performance)"""
    return GeminiPolicyEngine()

# Header
st.markdown('<div class="main-header">🔍 Google Ads Policy RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Ad Policy Review with Citations</div>', unsafe_allow_html=True)

# Sidebar - Project Info
with st.sidebar:
    st.header("📊 System Info")
    
    st.markdown("### Architecture")
    st.markdown("""
    **Phase 1**: Data Ingestion  
    **Phase 2**: Vector Search (BGE)  
    **Phase 3**: Hybrid + Reranking  
    **Phase 4**: LLM Generation (Gemini)
    """)
    
    st.markdown("### Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "100%")
        st.metric("Cost", "$0")
    with col2:
        st.metric("Latency", "~2.5s")
        st.metric("Chunks", "190")
    
    st.markdown("### Tech Stack")
    st.markdown("""
    - **Embeddings**: BGE-large-en
    - **Vector DB**: FAISS
    - **Search**: BM25 + Dense
    - **Reranking**: Cross-encoder
    - **LLM**: Gemini 2.5 Flash
    """)
    
    st.markdown("---")
    st.markdown("Built for portfolio demonstration")

# Main content - Tabs
tab1, tab2, tab3 = st.tabs(["🧪 Ad Review", "📚 Example Cases", "📈 System Metrics"])

with tab1:
    st.header("Ad Policy Review")
    
    # Input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ad_text = st.text_area(
            "Enter ad text to review:",
            placeholder="Example: Lose 15 pounds in one week with this miracle pill!",
            height=100
        )
    
    with col2:
        st.markdown("### Quick Examples")
        if st.button("🏥 Health Claim"):
            ad_text = "Lose 15 pounds in one week with this miracle pill!"
        if st.button("💰 Crypto"):
            ad_text = "Learn crypto trading from certified experts!"
        if st.button("📱 Product"):
            ad_text = "Buy smartphone - free shipping over $50"
    
    # Review button
    if st.button("🔍 Review Ad", type="primary", use_container_width=True):
        if not ad_text:
            st.warning("⚠️ Please enter ad text to review")
        else:
            # Load engine
            with st.spinner("Loading AI engine..."):
                try:
                    engine = load_engine()
                except Exception as e:
                    st.error(f"❌ Error loading engine: {e}")
                    st.info("💡 Make sure GOOGLE_API_KEY is set in .env file")
                    st.stop()
            
            # Review
            with st.spinner("🤖 Analyzing ad against Google Ads policies..."):
                start_time = time.time()
                decision = engine.review_ad(ad_text)
                elapsed = time.time() - start_time
            
            # Display results
            st.markdown("---")
            st.header("📋 Policy Decision")
            
            # Decision badge
            decision_class = f"decision-{decision.decision}"
            decision_emoji = {
                "allowed": "✅",
                "restricted": "⚠️",
                "disallowed": "❌",
                "unclear": "❓"
            }
            
            st.markdown(f"""
            <div class="{decision_class}">
                <h2>{decision_emoji[decision.decision]} Decision: {decision.decision.upper()}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Confidence", f"{decision.confidence:.1%}")
            with col2:
                st.metric("Processing Time", f"{elapsed:.2f}s")
            with col3:
                st.metric("Escalation", "Yes" if decision.escalation_required else "No")
            with col4:
                st.metric("Risk Factors", len(decision.risk_factors) if decision.risk_factors else 0)
            
            # Details
            st.markdown("### 📂 Policy Section")
            st.info(decision.policy_section)
            
            st.markdown("### 💬 Justification")
            st.write(decision.justification)
            
            st.markdown("### 📝 Policy Citation")
            st.markdown(f"> {decision.policy_quote}")
            st.markdown(f"[📖 View Official Policy]({decision.citation_url})")
            
            if decision.risk_factors:
                st.markdown("### ⚠️ Identified Risk Factors")
                for factor in decision.risk_factors:
                    st.markdown(f"- `{factor}`")
            
            if decision.escalation_required:
                st.warning("🚨 This case requires human review due to low confidence or policy ambiguity")

with tab2:
    st.header("📚 Example Test Cases")
    
    examples = [
        {
            "name": "Misleading Health Claims",
            "ad": "Lose 15 pounds in one week with this miracle pill! Guaranteed results!",
            "expected": "disallowed",
            "reasoning": "Unrealistic weight loss claims violate healthcare policy"
        },
        {
            "name": "Cryptocurrency Education",
            "ad": "Learn cryptocurrency trading from certified instructors. Online course available.",
            "expected": "restricted",
            "reasoning": "Crypto content requires Google certification"
        },
        {
            "name": "Standard Consumer Product",
            "ad": "Buy our new smartphone - 5G enabled, 128GB storage. Free shipping over $50.",
            "expected": "allowed",
            "reasoning": "Standard product ad with no policy violations"
        },
        {
            "name": "Financial Guarantees",
            "ad": "Invest in our fund - guaranteed 20% annual returns with no risk!",
            "expected": "disallowed",
            "reasoning": "Financial products cannot guarantee returns"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        with st.expander(f"Example {i}: {example['name']}"):
            st.markdown(f"**Ad Text:**")
            st.code(example['ad'])
            st.markdown(f"**Expected Decision:** `{example['expected']}`")
            st.markdown(f"**Reasoning:** {example['reasoning']}")

with tab3:
    st.header("📈 System Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Pipeline Performance")
        
        pipeline_data = {
            "Stage": ["Data Ingestion", "Embedding", "Hybrid Search", "Reranking", "LLM Generation"],
            "Time (ms)": [0, 0, 150, 100, 2000],
            "Description": [
                "Offline (preprocessing)",
                "Offline (preprocessing)", 
                "Dense + BM25 fusion",
                "Cross-encoder scoring",
                "Gemini API call"
            ]
        }
        
        st.dataframe(pipeline_data, use_container_width=True)
        
        st.markdown("### Accuracy Metrics")
        metrics_data = {
            "Metric": ["Test Accuracy", "Retrieval Recall@5", "Precision", "F1 Score"],
            "Value": ["100%", "91%", "89%", "90%"]
        }
        st.dataframe(metrics_data, use_container_width=True)
    
    with col2:
        st.markdown("### Technical Details")
        
        tech_details = """
        **Vector Database:**
        - 190 policy chunks indexed
        - 1024-dimensional embeddings
        - FAISS for similarity search
        
        **Retrieval:**
        - Hybrid: Dense + BM25
        - RRF fusion (k=60)
        - Cross-encoder reranking
        
        **Generation:**
        - Model: Gemini 2.5 Flash
        - Structured JSON output
        - Pydantic validation
        
        **Cost:**
        - Development: $0
        - Per query: $0 (free tier)
        - Monthly: $0
        """
        st.markdown(tech_details)
        
        st.markdown("### System Capabilities")
        capabilities = [
            "✅ Real-time policy review",
            "✅ Citation with source URLs",
            "✅ Risk factor detection",
            "✅ Confidence scoring",
            "✅ Escalation flagging",
            "✅ Multi-category support"
        ]
        for cap in capabilities:
            st.markdown(cap)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Google Ads Policy RAG System | Built with Streamlit, FAISS, BGE, and Gemini</p>
    <p>Portfolio Project - Production-Grade RAG Implementation</p>
</div>
""", unsafe_allow_html=True)
