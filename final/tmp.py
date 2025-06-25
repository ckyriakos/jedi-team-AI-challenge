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
from utils import decide_action, evaluate_response, is_relevant, run_web_search, generate_chat_title
#from feedback_manager import handle_feedback
from guardrails import is_chitchat
from websearchutils import  format_web_search_citation
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def format_source_citation(doc, index):
    """Format a retrieved document as a citation"""
    # Extract metadata if available
    metadata = getattr(doc, 'metadata', {})
    source = metadata.get('source', f'Document {index + 1}')
    page = metadata.get('page', '')
    
    # Create citation text
    citation_text = f"<strong>Source {index + 1}:</strong> {source}"
    if page:
        citation_text += f" (Page {page})"
    
    # Add snippet of content
    content_snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
    citation_text += f"<br><em>Excerpt:</em> {content_snippet}"
    
    return citation_text

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    user_id = data.get("user_id", "default")

    chat_id = data.get("chat_id", None)
    chat_id, history = get_chat(user_id, chat_id)
    messages = [HumanMessage(content=user_input)]
    add_message(user_id, chat_id, "user", user_input)
    
    if is_chitchat(user_input):
        answer = "Hi, How can I help you?"
        def simple_stream():
            yield f"data: [ASSISTANT]\n\n"
            yield f"data: {answer}\n\n"
            yield "data: [END]\n\n"
            
        add_message(user_id, chat_id, "assistant", answer)
        return StreamingResponse(simple_stream(), media_type="text/event-stream") 
    else:
        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(user_input)
        # In your chat endpoint, after document retrieval:
        # web_results = None
        #if not retrieved_docs or len(retrieved_docs) < 3:
        # Supplement with web search
        #    web_results = run_web_search(user_input)
    
        #if web_results:
        #    for i, result in enumerate(web_results):
        #        citation = format_web_search_citation(result, len(retrieved_docs) + i)
        #        yield f"data: [SOURCE]{citation}\n\n"
        def token_stream() -> Generator[str, None, None]:
            answer = ""
            #yield f"data: [ASSISTANT]\n\n"
            if retrieved_docs:
                # First, send citations for retrieved documents
                yield f"data: [CITATION]Found {len(retrieved_docs)} relevant source(s)\n\n"
                
                for i, doc in enumerate(retrieved_docs):
                    citation = format_source_citation(doc, i)
                    yield f"data: [SOURCE]{citation}\n\n"
                
                # Create context-aware prompt
                context = "\n".join([f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
                prompt = f"""You are a helpful assistant. Use the following context to answer the question. When referencing information, mention the source number (e.g., "According to Source 1..." or "Based on Source 2...").

Context:
{context}

Question: {user_input}

Answer:"""
                stream = llm.stream(prompt)
            else:
                yield f"data: [CITATION]No specific sources found, using general knowledge\n\n"
                stream = llm.stream(messages)

            # Start assistant response
            yield f"data: [ASSISTANT]\n\n"
            
            # Stream the main response
            for chunk in stream:
                token = chunk.content
                if answer and not answer.endswith(" ") and not token.startswith(" "):
                    token = " " + token
                answer += token
                yield f"data: {token}\n\n"
            
            # End signal
            yield "data: [END]\n\n"
            
            # Save the complete response
            add_message(user_id, chat_id, "assistant", answer)

        return StreamingResponse(token_stream(), media_type="text/event-stream")

@app.get("/list_user_chats")
async def list_chats(user_id: str):
    return JSONResponse(content=list_user_chats(user_id))

@app.post("/feedback")
async def submit_feedback(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    chat_id = data.get("chat_id")
    score = data.get("score")  # 1 for positive, -1 for negative
    
    # Here you could implement your feedback storage logic
    # For now, just return success
    print(f"Received feedback: user={user_id}, chat={chat_id}, score={score}")
    
    return JSONResponse(content={"status": "ok"})

@app.get("/analytics")
async def get_analytics():
    # Mock analytics data - replace with real implementation
    return JSONResponse(content={
        "total_chats": 42,
        "total_messages": 156,
        "satisfaction_rate": 0.85,
        "feedback_count": 23
    })

@app.get("/performance")
async def get_performance():
    # Mock performance data - replace with real implementation
    return JSONResponse(content={
        "basic_stats": {
            "total_chats": 42,
            "total_messages": 156
        },
        "learning_insights": {
            "total_patterns_learned": 12
        },
        "query_patterns": {
            "research": {"success_rate": 0.92, "total_queries": 25},
            "data_analysis": {"success_rate": 0.88, "total_queries": 18}
        },
        "current_config": {
            "vectorstore_weight": 0.7,
            "knowledge_base_weight": 0.2,
            "web_search_weight": 0.1
        }
    })

@app.get("/export")
async def export_chat(chat_id: str, user_id: str):
    markdown_text = export_chat_to_markdown(chat_id)
    html = markdown.markdown(markdown_text)
    return HTMLResponse(content=html)

@app.get("/resume")
async def resume_chat(chat_id: str, user_id: str):
    _, history = get_chat(user_id, chat_id)
    return JSONResponse(content=history)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
