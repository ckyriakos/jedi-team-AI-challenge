import re
"""
Guardrails module for content filtering and classification
"""

def is_chitchat(query: str) -> bool:
    """
    Determine if a query is chitchat/casual conversation rather than a data query
    """
    chitchat_keywords = [
        "hello", "hi", "hey", "how are you", "what's up", 
        "good morning", "good afternoon", "good evening", "who are you", 
        "are you a bot", "tell me a joke", "thank you", "thanks", "bye",
        "goodbye", "how do you do", "nice to meet you", "pleasure to meet you",
        "what can you do", "help me", "can you help", "what's your name"
    ]
    
    query_lower = query.lower().strip()
    
    # Check for exact matches or substrings
    for keyword in chitchat_keywords:
        #if keyword in query_lower:
    #    return True
        if query_lower == keyword or re.fullmatch(rf".*\b{re.escape(keyword)}\b.*", query_lower):
            return True
    # Check for question patterns that are chitchat
    chitchat_patterns = [
        "how are you",
        "what are you",
        "who made you",
        "what can you do",
        "are you real",
        "are you human"
    ]
    
    for pattern in chitchat_patterns:
        if pattern in query_lower:
            return True
    
    return False


def is_safe_content(text: str) -> bool:
    """
    Basic content safety filter
    """
    unsafe_keywords = [
        "hack", "exploit", "virus", "malware", "phishing",
        "illegal", "drugs", "violence", "harmful"
    ]
    
    text_lower = text.lower()
    return not any(keyword in text_lower for keyword in unsafe_keywords)


def classify_query_type(query: str) -> str:
    """
    Classify the type of query to help with routing
    """
    query_lower = query.lower()
    
    # Data/research queries
    if any(word in query_lower for word in ["data", "research", "study", "survey", "insights", "gwi", "market", "consumer"]):
        return "data_query"
    
    # Comparison queries
    if any(word in query_lower for word in ["compare", "vs", "versus", "difference", "better", "worse"]):
        return "comparison"
    
    # Trend queries
    if any(word in query_lower for word in ["trend", "trending", "popular", "growth", "decline", "increase"]):
        return "trend_analysis"
    
    # Chitchat
    if is_chitchat(query):
        return "chitchat"
    
    # Default
    return "general_query"
