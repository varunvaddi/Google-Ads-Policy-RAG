"""
Google Ads Policy RAG System - Gradio Demo
HuggingFace Spaces Deployment
Integrated with actual project structure
"""

import gradio as gr
import os
import sys
import time
from typing import Dict, List, Tuple
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import your actual components
from src.retrieval.hybrid_search import HybridSearch
from src.generation.decision_engine import DecisionEngine
from src.generation.decision_schema import PolicyDecision

class RAGSystem:
    """Main RAG system interface"""
    
    def __init__(self):
        """Initialize the RAG system with your actual components"""
        print("🚀 Loading RAG system...")
        
        try:
            # Initialize your hybrid search (BM25 + FAISS + Reranker)
            self.retriever = HybridSearch(
                faiss_index_path='data/embeddings/faiss_index.index',
                chunks_path='data/embeddings/chunks.json',  # or .pkl
                bm25_index_path='data/embeddings/bm25_index.pkl'
            )
            print("✅ Retrieval system loaded")
            
            # Initialize your decision engine (Gemini + Pydantic)
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            self.generator = DecisionEngine(api_key=api_key)
            print("✅ Generation system loaded")
            
            print("✅ System ready!")
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            raise
        
    def process_query(self, query: str) -> Dict:
        """
        Process a query through the RAG pipeline
        
        Args:
            query: User's ad text or policy question
            
        Returns:
            Dictionary with decision, confidence, citations, etc.
        """
        start_time = time.time()
        
        try:
            # 1. Retrieval phase - Use your hybrid search
            retrieved_chunks = self.retriever.search(
                query=query,
                top_k=5  # Get top 5 after reranking
            )
            
            # 2. Generation phase - Use your decision engine
            decision = self.generator.generate_decision(
                query=query,
                context_chunks=retrieved_chunks
            )
            
            # 3. Format response
            result = {
                'decision': decision.decision,  # APPROVE/REJECT/NEEDS_REVIEW
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'risk_factors': decision.risk_factors,
                'recommendations': decision.recommendations if hasattr(decision, 'recommendations') else [],
                'citations': [
                    {
                        'text': chunk.get('text', chunk.get('content', '')),
                        'source': chunk.get('source', chunk.get('metadata', {}).get('source', 'Unknown')),
                        'score': chunk.get('score', chunk.get('rerank_score', 0.0))
                    }
                    for chunk in retrieved_chunks
                ],
                'latency': round(time.time() - start_time, 2)
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                'decision': 'ERROR',
                'confidence': 0.0,
                'reasoning': f'System error: {str(e)}',
                'risk_factors': [],
                'recommendations': ['Please try again or contact support'],
                'citations': [],
                'latency': round(time.time() - start_time, 2)
            }


def create_demo_interface():
    """Create the Gradio interface"""
    
    # Initialize RAG system
    try:
        rag = RAGSystem()
        system_ready = True
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        system_ready = False
    
    def process_and_format(query: str) -> Tuple[str, str, str, str]:
        """
        Process query and format results for Gradio
        
        Returns:
            Tuple of (decision_html, reasoning, citations_html, metrics_html)
        """
        if not system_ready:
            error_html = """
            <div style="padding: 20px; border-radius: 10px; background: #ffebee;">
                <h2 style="margin: 0; color: #c62828;">⚠️ System Error</h2>
                <p style="margin: 10px 0 0 0;">
                    RAG system failed to initialize. Please check logs and ensure:
                    <br>• GOOGLE_API_KEY is set in environment
                    <br>• Data files exist in data/embeddings/
                    <br>• All dependencies are installed
                </p>
            </div>
            """
            return error_html, "System not ready", "", ""
        
        if not query.strip():
            return "⚠️ Please enter a query", "", "", ""
        
        # Process through RAG
        result = rag.process_query(query)
        
        # Format decision
        decision_colors = {
            'REJECT': 'red',
            'APPROVE': 'green',
            'NEEDS_REVIEW': 'orange',
            'ERROR': 'gray'
        }
        decision_icons = {
            'REJECT': '❌',
            'APPROVE': '✅',
            'NEEDS_REVIEW': '⚠️',
            'ERROR': '⚠️'
        }
        
        decision = result['decision']
        color = decision_colors.get(decision, 'gray')
        icon = decision_icons.get(decision, '?')
        
        decision_html = f"""
        <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #{color}33, #{color}22);">
            <h2 style="margin: 0; color: {color};">
                {icon} {decision}
            </h2>
            <p style="margin: 10px 0 0 0; font-size: 18px;">
                Confidence: <strong>{result['confidence']:.0%}</strong>
            </p>
        </div>
        """
        
        # Format reasoning
        reasoning = result['reasoning']
        
        # Format risk factors
        if result.get('risk_factors'):
            reasoning += "\n\n**🚨 Risk Factors:**\n"
            for risk in result['risk_factors']:
                reasoning += f"- {risk}\n"
        
        # Format recommendations
        if result.get('recommendations'):
            reasoning += "\n\n**💡 Recommendations:**\n"
            for rec in result['recommendations']:
                reasoning += f"- {rec}\n"
        
        # Format citations
        citations_html = "<h3>📚 Supporting Evidence</h3>"
        for i, citation in enumerate(result['citations'], 1):
            citations_html += f"""
            <div style="margin: 10px 0; padding: 15px; background: #f5f5f5; border-left: 4px solid #4CAF50; border-radius: 5px;">
                <p style="margin: 0; font-weight: bold; color: #333;">Citation {i} (Score: {citation['score']:.3f})</p>
                <p style="margin: 5px 0; color: #666; font-style: italic; font-size: 0.9em;">{citation['source']}</p>
                <p style="margin: 5px 0; color: #333; font-size: 0.95em;">{citation['text'][:200]}...</p>
            </div>
            """
        
        # Format metrics
        metrics_html = f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 20px;">
            <div style="padding: 15px; background: #e3f2fd; border-radius: 8px; text-align: center;">
                <p style="margin: 0; font-size: 24px; font-weight: bold; color: #1976d2;">80%</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;">Decision Accuracy</p>
            </div>
            <div style="padding: 15px; background: #f3e5f5; border-radius: 8px; text-align: center;">
                <p style="margin: 0; font-size: 24px; font-weight: bold; color: #7b1fa2;">78%</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;">Retrieval Recall@5</p>
            </div>
            <div style="padding: 15px; background: #fff3e0; border-radius: 8px; text-align: center;">
                <p style="margin: 0; font-size: 24px; font-weight: bold; color: #e65100;">{result['latency']}s</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;">Query Latency</p>
            </div>
        </div>
        """
        
        return decision_html, reasoning, citations_html, metrics_html
    
    # Example queries for recruiters to try
    examples = [
        ["I want to advertise prescription pain medication online"],
        ["Offering personal loans with no credit check required"],
        ["Free trial for our premium software product - cancel anytime"],
        ["Investment opportunity with guaranteed 20% annual returns"],
        ["Medical device that naturally cures diabetes without medication"],
        ["Limited time offer: Buy one get one free on all products"],
        ["Are tobacco products allowed in Google Ads?"],
        ["Can I advertise alcohol delivery services?"]
    ]
    
    # Custom CSS for better appearance
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    # Create interface
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        # Header
        gr.HTML("""
        <div class="header">
            <h1 style="margin: 0; font-size: 2.5em;">🔍 Google Ads Policy RAG System</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">Production-Grade AI for Policy Compliance</p>
            <div style="margin-top: 15px; display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px;">80% Accuracy</span>
                <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px;">78% Recall@5</span>
                <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px;">341 Chunks</span>
                <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px;">&lt;3s Latency</span>
            </div>
        </div>
        """)
        
        # System Status
        if system_ready:
            gr.Markdown("✅ **System Status:** Ready")
        else:
            gr.Markdown("❌ **System Status:** Initialization Failed - Check logs")
        
        # Description
        gr.Markdown("""
        ## How It Works
        
        This system uses **hybrid retrieval** (BM25 + Dense Embeddings) combined with **cross-encoder reranking** 
        and **Google Gemini** to make accurate policy decisions. All decisions are backed by citations from official 
        Google Ads policies.
        
        ### 🎯 Try It Out
        Enter your ad text or ask a policy question below:
        """)
        
        # Input section
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Ad Text or Policy Question",
                    placeholder="e.g., 'Can I advertise prescription medication?' or paste your ad text",
                    lines=3
                )
                submit_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
        
        # Examples
        gr.Examples(
            examples=examples,
            inputs=query_input,
            label="📝 Example Queries"
        )
        
        # Output section
        gr.Markdown("## 📊 Results")
        
        with gr.Row():
            with gr.Column(scale=1):
                decision_output = gr.HTML(label="Decision")
            
        with gr.Row():
            with gr.Column(scale=2):
                reasoning_output = gr.Markdown(label="Analysis")
            with gr.Column(scale=1):
                citations_output = gr.HTML(label="Citations")
        
        metrics_output = gr.HTML(label="System Metrics")
        
        # Connect the button
        submit_btn.click(
            fn=process_and_format,
            inputs=query_input,
            outputs=[decision_output, reasoning_output, citations_output, metrics_output]
        )
        
        # Architecture section
        gr.Markdown("""
        ---
        ## 🏗️ System Architecture
        
        **Pipeline:**
        1. **Data Ingestion**: 25 policy pages → 341 clean chunks (40s)
        2. **Vector Generation**: BGE-large embeddings, 1024-dim (24s)  
        3. **Hybrid Search**: FAISS + BM25 + RRF fusion (<1s)
        4. **Reranking**: Cross-encoder (BGE-reranker-large) 
        5. **Generation**: Google Gemini 2.5 Flash + Pydantic (2-3s)
        
        **Tech Stack:**
        - Embeddings: BGE-large-en-v1.5
        - Vector DB: FAISS FlatIP
        - Keyword: BM25Okapi  
        - Reranking: BGE-reranker-large
        - LLM: Google Gemini 2.5 Flash
        - Validation: Pydantic v2.5
        
        **Performance:**
        - ✅ 80% Decision Accuracy
        - ✅ 78% Retrieval Recall@5
        - ✅ 0.778 MRR (Mean Reciprocal Rank)
        - ✅ <3s Query Latency
        - ✅ Zero Hallucinations (100% citations)
        - ✅ $0 Infrastructure Cost
        """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666; margin-top: 30px; border-top: 1px solid #ddd;">
            <p style="margin: 5px 0;">
                <strong>Built by:</strong> [Your Name] | 
                <a href="https://github.com/yourusername/google-ads-policy-rag" target="_blank">GitHub</a> | 
                <a href="https://linkedin.com/in/yourprofile" target="_blank">LinkedIn</a>
            </p>
            <p style="margin: 5px 0; font-size: 0.9em;">
                Production-ready RAG system • Python • BGE • FAISS • Gemini
            </p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo_interface()
    
    # Launch with public sharing enabled
    demo.launch(
        share=False,  # Set to False for HuggingFace Spaces (automatic)
        server_name="0.0.0.0",  # Required for HuggingFace
        server_port=7860,  # Default Gradio port
        show_error=True
    )
