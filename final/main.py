from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from typing import Generator
import uvicorn
import os
import markdown
import json
import time
from model_config import retriever, llm  # Make sure this is defined
from terminalui import add_message, get_chat, list_user_chats, export_chat_to_markdown#/, delete_chat
from utils import decide_action, is_relevant, generate_chat_title, format_source_citation
from feedback_manager import add_feedback, AdaptiveFeedbackManager
from guardrails import is_chitchat, classify_query_type, is_safe_content
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from evaluation import EnhancedEvaluator 
# Import the new adaptive system
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime, timedelta





# Initialize the FastAPI app and components
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize embedding model for relevance checking
embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"device": "cuda", "trust_remote_code": True}
)

# Initialize the adaptive system
adaptive_feedback_manager = AdaptiveFeedbackManager()
evaluator = EnhancedEvaluator()

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/chat")
async def adaptive_chat_endpoint(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    user_id = data.get("user_id", "default")
    chat_id = data.get("chat_id", None)
    
    # Safety check
    if not is_safe_content(user_input):
        def safety_stream():
            yield f"data: [ASSISTANT]\n\n"
            yield f"data: I can't help with that request. Please ask something else.\n\n"
            yield "data: [END]\n\n"
        return StreamingResponse(safety_stream(), media_type="text/event-stream")
    
    # Get or create chat
    chat_id, history = get_chat(user_id, chat_id)
    add_message(user_id, chat_id, "user", user_input)
    
    # Classify query type
    query_type = classify_query_type(user_input)
    
    def adaptive_agent_stream() -> Generator[str, None, None]:
        start_time = time.time()
        
        # Get adaptive strategy
        adaptive_config, strategy_info = adaptive_feedback_manager.get_adaptive_strategy(user_input)
        
        yield f"data: [ADAPTATION] Using adaptive strategy for {strategy_info['query_type']} query\n\n"
        
        if strategy_info['strategy_explanation']:
            for explanation in strategy_info['strategy_explanation']:
                yield f"data: [STRATEGY] {explanation}\n\n"
        
        # Handle chitchat early
        if is_chitchat(user_input):
            yield f"data: [THOUGHT] Chitchat detected - providing friendly response. \n\n"
            yield f"data: [ASSISTANT] \n\n"
            answer = "Hi! I'm here to help you with GWI market research insights and data analysis. What would you like to know?"
            yield f"data: {answer}\n\n"
            yield "data: [END]\n\n"
            # At the end, emit one full assistant message
            
            add_message(user_id, chat_id, "assistant", answer)
            
            # Store feedback for chitchat
            response_time = time.time() - start_time
            adaptive_feedback_manager.add_feedback_with_context(
                user_id, chat_id, user_input, answer, True,
                {"score": 1, "status": "good", "confidence": 0.9}, "chitchat", response_time
            )
            return
        
        # Step 1: Try retrieval from internal vectorstore with adaptive threshold
        yield f"data: [THOUGHT] Searching internal GWI vectorstore (threshold: {adaptive_config.vectorstore_threshold:.2f}, weight: {adaptive_config.vectorstore_weight:.2f})...\n\n"
        retrieved_docs = retriever.invoke(user_input)
        
        if retrieved_docs and is_relevant(retrieved_docs, user_input, embedding_model, threshold=adaptive_config.vectorstore_threshold):
            yield f"data: [THOUGHT] Found relevant vectorstore data. Generating response...\n\n"
            # First, send citations for retrieved documents
            yield f"data: [CITATION]Found {len(retrieved_docs)} relevant source(s)\n\n"
                
            for i, doc in enumerate(retrieved_docs):
                citation = format_source_citation(doc, i)
                yield f"data: [SOURCE]{citation}\n\n"
            
            yield f"data: [ASSISTANT] \n\n" 
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            
            prompt = f"""You are a helpful GWI research assistant. Use the following GWI insights to answer the user's question.
Only use the provided context. If the context doesn't contain enough information, say so. Ask followup questions to engage the user.


Context:
{context}

Question: {user_input}

Answer:"""
            
            # Generate response from vectorstore data
            stream = llm.stream(prompt)
            answer = ""
            for chunk in stream:
                token = chunk.content
                answer += token
                yield f"data: {token} \n\n"
                #yield f"data: {answer} \n\n" returns result like this According to the provided context, 12% of Gen Z inAccording to the provided context, 12% of Gen Z in NashvilleAccor
            
            # Evaluate the vectorstore response
            eval_result = evaluator.evaluate_response(user_input, answer, retrieved_docs, "vectorstore")
            yield f"data: [EVALUATION] Vectorstore response score: {eval_result.score:.1f} - {eval_result.status.upper()} (confidence: {eval_result.confidence:.2f})\n\n"
            
            # Store feedback with context
            response_time = time.time() - start_time
            was_helpful = eval_result.score > 0 and eval_result.status not in ['fail', 'poor']
            adaptive_feedback_manager.add_feedback_with_context(
                user_id, chat_id, user_input, answer, was_helpful,
                {"score": eval_result.score, "status": eval_result.status, "confidence": eval_result.confidence}, 
                "vectorstore", response_time
            )
            
            # Use adaptive weight for decision making
            if eval_result.score >= (2 - adaptive_config.vectorstore_weight):  # Adaptive threshold
                # Vectorstore response is good enough based on adaptive config
                yield f"data: [ASSISTANT]: {answer}\n\n"
                yield "data: [END]\n\n"
                add_message(user_id, chat_id, "assistant", answer)
                return
            else:
                yield f"data: [THOUGHT] Vectorstore response quality insufficient (score: {eval_result.score:.1f} < threshold: {2 - adaptive_config.vectorstore_weight:.1f}). Trying knowledge base...\n\n"
        else:
            yield f"data: [THOUGHT] No relevant vectorstore data found. Trying knowledge base...\n\n"
            #yield f"data: [CITATION]\n\n"# No specific sources found, using general knowledge\n\n"
        
        # Step 2: Try knowledge base (LLM's internal knowledge) with adaptive weighting
        yield f"data: [THOUGHT] Querying knowledge base (weight: {adaptive_config.knowledge_base_weight:.2f})...\n\n"
        yield f"data: [ASSISTANT] \n\n"
        knowledge_msg = [HumanMessage(content=user_input)]
        context_section = ""
        if retrieved_docs:
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            context_section = f"\nGWI Context (for reference):\n{context}\n"
        
        prompt = f"""You are a helpful assistan specialized in answering questions related to consumer insights.
        If there icontext is relevant to the question use it to answer. If there multiple questions break them down and reply. Ask followup questions to engage the user.
        Don't be too verbose unless asked.


Context:
{context_section}

Question: {knowledge_msg}

Answer:""" 
        try:
            # Generate response from knowledge base
            stream = llm.stream(prompt)
            knowledge_answer = ""
            reasoning =""
            final = ""
            for chunk in stream:
                if hasattr(chunk, "reasoning") and hasattr(chunk, "final_answer"):
                # Only happens once, at the end

                    # print("reasoning") Debug reasoning
                    reasoning = chunk.reasoning
                    final = chunk.final_answer

                    yield f"data: [THOUGHT] {reasoning.strip()}\n\n"
                    yield f"data: [ASSISTANT]\n\n"
                    
                    for line in final.strip().split():
                        yield f"data: {line} \n\n"
                        knowledge_answer += line + " "
                    break 
                
                else:
                    # print("content") DEbug reasoning
                    token = chunk.content
                    knowledge_answer += token
                    yield f"data: {token}\n\n"
            
            # Evaluate the knowledge base response
            eval_result = evaluator.evaluate_response(user_input, knowledge_answer, [])
            yield f"data: [EVALUATION] Knowledge base response score: {eval_result.score:.1f} - {eval_result.status.upper()} (confidence: {eval_result.confidence:.2f})\n\n"
            
            # Store feedback
            response_time = time.time() - start_time
            was_helpful = eval_result.score > 0 and eval_result.status not in ['fail', 'poor']
            adaptive_feedback_manager.add_feedback_with_context(
                user_id, chat_id, user_input, knowledge_answer, was_helpful,
                {"score": eval_result.score, "status": eval_result.status, "confidence": eval_result.confidence}, 
                "knowledge_base", response_time
            )
            
            # Use adaptive weight for decision making
            if eval_result.score >= (1 - adaptive_config.knowledge_base_weight):  # Adaptive threshold
                # Knowledge base response is good enough
                #yield f"data: [ASSISTANT]: {knowledge_answer}\n\n"
                yield "data: [END]\n\n"
                add_message(user_id, chat_id, "assistant", knowledge_answer)
                return
            else:
                yield f"data: [THOUGHT] Knowledge base response quality insufficient (score: {eval_result.score:.1f}). Trying web search...\n\n"
                
        except Exception as e:
            yield f"data: [ERROR] Knowledge base query failed: {str(e)}. Trying web search...\n\n"
        
        # Step 3: Web search fallback with adaptive configuration
        try:
            yield f"data: [THOUGHT] Performing web search (weight: {adaptive_config.web_search_weight:.2f})...\n\n"
            web_search = DuckDuckGoSearchResults()
            web_results = web_search.run(user_input)
            yield f"data: [THOUGHT] Found web results. Generating response...\n\n"
            yield f"data: [ASSISTANT] \n\n"
            
            # Include context from vectorstore if it exists (even if not relevant enough)
            context_section = ""
            if retrieved_docs:
                context = "\n".join([doc.page_content for doc in retrieved_docs])
                context_section = f"\nGWI Context (for reference):\n{context}\n"
            
            prompt = f"""You are a helpful research assistant. Use the following web search results to answer the user's question.
Provide a comprehensive answer based on the search results. You can also reference the GWI context if relevant.
{context_section}
Web Search Results:
{web_results}

Question: {user_input}

Answer:"""
            
            stream = llm.stream(prompt)
            web_answer = ""
            for chunk in stream:
                token = chunk.content
                web_answer += token
                yield f"data: {token}\n\n"
            
            # Evaluate web search response
            eval_result = evaluator.evaluate_response(user_input, web_answer, retrieved_docs)
            
            # Store feedback
            response_time = time.time() - start_time
            was_helpful = eval_result.score > -1  # More lenient for web search
            adaptive_feedback_manager.add_feedback_with_context(
                user_id, chat_id, user_input, web_answer, was_helpful,
                {"score": eval_result.score, "status": eval_result.status, "confidence": eval_result.confidence}, 
                "web_search", response_time
            )
            yield f"data: [ASSISTANT]: {web_answer}\n\n" 
            yield "data: [END]\n\n"
            add_message(user_id, chat_id, "assistant", web_answer)
            
        except Exception as e:
            yield f"data: [ERROR] Web search failed: {str(e)}\n\n"
            fallback_answer = "I apologize, but I couldn't find relevant information in our database, knowledge base, or through web search. Could you try rephrasing your question?"
            yield f"data: {fallback_answer}\n\n"
            yield "data: [END]\n\n"
            add_message(user_id, chat_id, "assistant", fallback_answer)
            
            # Store feedback for fallback
            response_time = time.time() - start_time
            adaptive_feedback_manager.add_feedback_with_context(
                user_id, chat_id, user_input, fallback_answer, False,
                {"score": -2, "status": "fail", "confidence": 0.1}, 
                "fallback", response_time
            )

    return StreamingResponse(adaptive_agent_stream(), media_type="text/event-stream")

@app.get("/list_user_chats")
async def list_chats(user_id: str):
    return JSONResponse(content=list_user_chats(user_id))

@app.post("/feedback")
async def submit_feedback(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    chat_id = data.get("chat_id")
    score = data.get("score")  # 1 for thumbs up, -1 for thumbs down
    notes = data.get("notes", "")
    
    # Get the last assistant message for feedback
    from terminalui import load_conversations
    db = load_conversations()
    messages = db.get(user_id, {}).get(chat_id, [])
    
    # Find last user query and assistant response
    last_query = ""
    last_answer = ""
    for i in range(len(messages)-1, -1, -1):
        if messages[i]["role"] == "assistant" and not last_answer:
            last_answer = messages[i]["content"]
        elif messages[i]["role"] == "user" and not last_query:
            last_query = messages[i]["content"]
            break
    
    if last_query and last_answer:
        add_feedback(user_id, chat_id, last_query, last_answer, score > 0)
        
        # Also add to adaptive feedback system
        adaptive_feedback_manager.add_feedback_with_context(
            user_id, chat_id, last_query, last_answer, score > 0,
            {"score": score, "status": "user_feedback", "confidence": 1.0}, 
            "user_feedback", 0
        )
    
    return JSONResponse(content={"status": "ok", "message": "Feedback recorded"})

@app.get("/performance")
async def get_performance_dashboard():
    """Enhanced performance dashboard with adaptive insights"""
    try:
        # Get original analytics
        from terminalui import load_conversations
        db = load_conversations()
        
        total_chats = sum(len(user_chats) for user_chats in db.values())
        total_messages = sum(
            len(messages) 
            for user_chats in db.values() 
            for messages in user_chats.values()
        )
        
        # Get adaptive performance insights
        trends = adaptive_feedback_manager.analyze_performance_trends()
        
        # Get current adaptive configuration
        current_config = adaptive_feedback_manager.config.to_dict()
        
        # Get query patterns
        query_patterns = {}
        for pattern_key, pattern_data in adaptive_feedback_manager.query_patterns.items():
            if pattern_data['total'] > 0:
                query_patterns[pattern_key] = {
                    "success_rate": pattern_data['success'] / pattern_data['total'],
                    "total_queries": pattern_data['total']
                }
        
        return JSONResponse(content={
            "basic_stats": {
                "total_chats": total_chats,
                "total_messages": total_messages
            },
            "adaptive_performance": trends,
            "current_config": current_config,
            "query_patterns": query_patterns,
            "learning_insights": {
                "total_patterns_learned": len(adaptive_feedback_manager.query_patterns),
                "config_adaptations": len([x for x in trends.values() if isinstance(x, dict) and x.get('total', 0) > 0])
            }
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/analytics")
async def get_analytics():
    """Simple analytics endpoint (keeping for backward compatibility)"""
    try:
        from terminalui import load_conversations
        db = load_conversations()
        
        total_chats = sum(len(user_chats) for user_chats in db.values())
        total_messages = sum(
            len(messages) 
            for user_chats in db.values() 
            for messages in user_chats.values()
        )
        
        # Count feedback
        feedback_count = 0
        positive_feedback = 0
        try:
            with open("feedback_log.jsonl", "r") as f:
                for line in f:
                    feedback_count += 1
                    feedback_data = json.loads(line)
                    if feedback_data.get("was_helpful", False):
                        positive_feedback += 1
        except FileNotFoundError:
            pass
        
        return JSONResponse(content={
            "total_chats": total_chats,
            "total_messages": total_messages,
            "feedback_count": feedback_count,
            "satisfaction_rate": positive_feedback / feedback_count if feedback_count > 0 else 0
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/export")
async def export_chat(user_id: str, chat_id: str):
    markdown_text = export_chat_to_markdown(user_id, chat_id, f"{chat_id}.md")
    html = markdown.markdown(markdown_text)
    return HTMLResponse(content=html)

@app.get("/resume")
async def resume_chat(chat_id: str, user_id: str):
    _, history = get_chat(user_id, chat_id)
    return JSONResponse(content=history)


#@app.delete("/delete_chat/{user_id}/{chat_id}")
#def api_delete_chat(user_id: str, chat_id: str):
#    success = delete_chat(user_id, chat_id)
#    return {"status": "deleted" if success else "not found"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
