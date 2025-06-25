from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from typing import Generator
import uvicorn
import os
import markdown
import json
from ollamaex import retriever, llm  # Make sure this is defined
from terminalui import add_message, get_chat, list_user_chats, export_chat_to_markdown
from utils import decide_action, is_relevant, run_web_search, generate_chat_title
from guardrails import is_chitchat, classify_query_type, is_safe_content
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from evaluation import EnhancedEvaluator
from feedback_manager import AdaptiveConfig, AdaptiveFeedbackManager

app = FastAPI()

# Create an instance of the evaluator class
evaluator = EnhancedEvaluator()
adaptive_feedback_manager = AdaptiveFeedbackManager()
app.mount("/static", StaticFiles(directory="static"), name="static")


# Initialize embedding model for relevance checking
embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"device": "cuda", "trust_remote_code": True}
)

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("../final/static/index.html", "r") as f:
        return f.read()

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    user_id = data.get("user_id", "default")
    chat_id = data.get("chat_id", None)
    
    # Safety check
    if not is_safe_content(user_input):
        def safety_stream():
            yield f"data: I can't help with that request. Please ask something else.\n\n"
            yield "data: [END]\n\n"
        return StreamingResponse(safety_stream(), media_type="text/event-stream")
    
    # Get or create chat
    chat_id, history = get_chat(user_id, chat_id)
    add_message(user_id, chat_id, "user", user_input)
    
    # Classify query type
    query_type = classify_query_type(user_input)
    
    def agent_stream() -> Generator[str, None, None]:
        yield f"data: [THOUGHT] Analyzing query type: {query_type}\n\n"
        
        # Handle chitchat early
        if is_chitchat(user_input):
            yield f"data: [THOUGHT] Chitchat detected - providing friendly response\n\n"
            answer = "Hi! I'm here to help you with GWI market research insights and data analysis. What would you like to know?"
            yield f"data: {answer}\n\n"
            yield "data: [END]\n\n"
            add_message(user_id, chat_id, "assistant", answer)
            return
        
        # Step 1: Try retrieval from internal vectorstore
        yield f"data: [THOUGHT] Searching internal GWI vectorstore...\n\n"
        retrieved_docs = retriever.invoke(user_input)
        
        if retrieved_docs and is_relevant(retrieved_docs, user_input, embedding_model):
            yield f"data: [THOUGHT] Found relevant vectorstore data. Generating response...\n\n"
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            
            prompt = f"""You are a helpful GWI research assistant. Use the following GWI insights to answer the user's question.
Only use the provided context. If the context doesn't contain enough information, say so.

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
                yield f"data: {token}\n\n"
            
            # Evaluate the vectorstore response
            eval_result = evaluator.evaluate_response(user_input, answer, retrieved_docs,"vectorstore")
            yield f"data: [EVALUATION] Vectorstore response score: {eval_result.score} - {eval_result.status.upper()}\n\n"
            
            if eval_result.status != 'fail':
                # Vectorstore response is good, we're done
                yield "data: [END]\n\n"
                add_message(user_id, chat_id, "assistant", answer)
                return
            else:
                yield f"data: [THOUGHT] Vectorstore response quality insufficient. Trying knowledge base...\n\n"
        else:
            yield f"data: [THOUGHT] No relevant vectorstore data found. Trying knowledge base...\n\n"
        
        # Step 2: Try knowledge base (LLM's internal knowledge)
        yield f"data: [THOUGHT] Querying knowledge base...\n\n"
        knowledge_msg = [HumanMessage(content=user_input)]
        
        try:
            # Generate response from knowledge base
            stream = llm.stream(knowledge_msg)
            knowledge_answer = ""
            for chunk in stream:
                token = chunk.content
                knowledge_answer += token
                yield f"data: {token}\n\n"
            
            # Evaluate the knowledge base response
            eval_result = evaluator.evaluate_response(user_input, knowledge_answer, [])
            yield f"data: [EVALUATION] Knowledge base response score: {eval_result.score} - {eval_result.status.upper()}\n\n"
            
            if eval_result.status != 'fail':
                # Knowledge base response is good, we're done
                yield "data: [END]\n\n"
                add_message(user_id, chat_id, "assistant", knowledge_answer)
                return
            else:
                yield f"data: [THOUGHT] Knowledge base response quality insufficient. Trying web search...\n\n"
                
        except Exception as e:
            yield f"data: [ERROR] Knowledge base query failed: {str(e)}. Trying web search...\n\n"
        
        # Step 3: Web search fallback
        try:
            yield f"data: [THOUGHT] Performing web search...\n\n"
            web_search = DuckDuckGoSearchResults()
            web_results = web_search.run(user_input)
            yield f"data: [THOUGHT] Found web results. Generating response...\n\n"
            
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
            
            yield "data: [END]\n\n"
            add_message(user_id, chat_id, "assistant", web_answer)
            
        except Exception as e:
            yield f"data: [ERROR] Web search failed: {str(e)}\n\n"
            fallback_answer = "I apologize, but I couldn't find relevant information in our database, knowledge base, or through web search. Could you try rephrasing your question?"
            yield f"data: {fallback_answer}\n\n"
            yield "data: [END]\n\n"
            add_message(user_id, chat_id, "assistant", fallback_answer)

    return StreamingResponse(agent_stream(), media_type="text/event-stream")

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
    
    return JSONResponse(content={"status": "ok", "message": "Feedback recorded"})

@app.get("/analytics")
async def get_analytics():
    """Simple analytics endpoint"""
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
