"""
Prompt Templates - FINAL VERSION
"""

POLICY_REVIEW_SYSTEM_PROMPT = """You are a Google Ads Policy Expert AI Assistant.

CRITICAL RULES:
1. Base ALL decisions on provided policy context only
2. Always cite exact policy language
3. Use clear decisions: allowed/restricted/disallowed/unclear
4. If unclear, set decision="unclear" and escalation_required=true
5. Extract specific violating phrases

OUTPUT: Valid JSON matching PolicyDecision schema."""

POLICY_REVIEW_USER_PROMPT = """Review this ad against Google Ads policies.

AD CONTENT:
{ad_text}

RELEVANT POLICIES:
{policy_context}

Provide JSON with:
- decision: "allowed"|"restricted"|"disallowed"|"unclear"
- confidence: 0.0-1.0
- policy_section: hierarchy (e.g., "Healthcare > Weight Loss")
- citation_url: official policy URL
- justification: clear explanation
- policy_quote: exact policy text
- risk_factors: list of violating phrases
- escalation_required: true if needs human review"""

POLICY_QA_SYSTEM_PROMPT = """You are a Google Ads Policy Expert.

Answer questions using ONLY provided policy context.
Always cite sources and quote policies.
If policy doesn't address question, say so explicitly."""

POLICY_QA_USER_PROMPT = """Answer this policy question.

QUESTION:
{question}

RELEVANT POLICIES:
{policy_context}

Provide JSON answer with citations."""


def format_policy_context(retrieved_chunks: list) -> str:
    """Format policy chunks for LLM"""
    formatted_parts = []
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        hierarchy = " > ".join(chunk['metadata']['hierarchy'])
        url = chunk['metadata']['url']
        content = chunk['content']
        
        formatted = f"""
POLICY {i}:
Category: {hierarchy}
Source: {url}

Content:
{content}

---
"""
        formatted_parts.append(formatted)
    
    return "\n".join(formatted_parts)


def format_policy_review_prompt(ad_text: str, policy_chunks: list) -> dict:
    """Create prompt for ad review"""
    policy_context = format_policy_context(policy_chunks)
    
    return {
        "system": POLICY_REVIEW_SYSTEM_PROMPT,
        "user": POLICY_REVIEW_USER_PROMPT.format(
            ad_text=ad_text,
            policy_context=policy_context
        )
    }


def format_policy_qa_prompt(question: str, policy_chunks: list) -> dict:
    """Create prompt for Q&A"""
    policy_context = format_policy_context(policy_chunks)
    
    return {
        "system": POLICY_QA_SYSTEM_PROMPT,
        "user": POLICY_QA_USER_PROMPT.format(
            question=question,
            policy_context=policy_context
        )
    }