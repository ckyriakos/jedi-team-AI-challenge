import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate
import uuid
import json
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun,DuckDuckGoSearchResults
from langchain.agents import Tool

DB_FILE = "conversations.json"
BASE_URL = "localhost:11434"
from fastapi.responses import StreamingResponse
from guardrails import is_chitchat
from transformers import pipeline
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from uuid import uuid4
import json
import os
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from feedback_manager import add_feedback
#from your_model_code import agent_stream  # wherever your generator lives
from utils import decide_action, evaluate_response, is_relevant
from terminalui import get_chat, add_message, load_conversations, save_conversations, get_langchain_messages, export_chat_to_markdown, show_chat_history, list_user_chats

# --- Data Ingestion ---
def data_collection():
    data = pd.read_csv("data.md", sep="|", usecols=[1], names=["text"], skiprows=2, engine="python")
    data["text"] = data["text"].str.strip()
    return data

def ingest_to_vectorstore():
    df = data_collection()
    docs = [Document(page_content=row) for row in df["text"]]
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    #embedding_model = OllamaEmbeddings(
    #    model="mxbai-embed-large:335m",
    #    base_url=BASE_URL,
    #    num_gpu=1
    #)
    embedding_model = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={"device": "cuda", "trust_remote_code": True}
    )
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    return vectorstore



print("Initializing chat model")
llm = ChatOllama(
    model = "llama3.1:8b",
    temperature=0.3,
    num_predict=256,
    num_gpu=0,
    extract_reasoning=TRue,
    verbose=True
    )
# --- LLM Setup ---
    #llm_pipeline = HuggingFacePipeline.from_model_id(
    #    model_id="Qwen/Qwen3-1.7B",
    #    task="text-generation",
    #    device=-1,  # or -1 for CPU
    #    pipeline_kwargs=dict(
    #        max_new_tokens=256,
    #        do_sample=True,
    #        repetition_penalty=1.03
    #    )
    #)
    #chat_model = ChatHuggingFace(llm=llm_pipeline)


# --- Run Retrieval + Planner ---
print("Loading vectorstore")
vectorstore = ingest_to_vectorstore()
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
#qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

#query = "What are some gen z buying habits based on gwi insights?"
#response = handle_query(query, planner_llm=chat_model, qa_chain=qa_chain, llm=chat_model)
#print("\n🧠 Response:\n", response)
#while True:
#    query = input("🧠 Ask your question:\n>>> ")
#    if query=='exit':
#        break
#    response = handle_query(query, retriever, chat_model)
#    print("\n💬 Response:\n", response)

def run_chat(user_id, chat_model, retriever, chat_id=None, min_answer_length=50):
    chat_id, history = get_chat(user_id, chat_id)
    #chat_id, history = get_chat(user_id)

    # Convert saved history into LangChain message objects
    messages = []
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    # Get new input
    query = input("💬 You: ").strip()
    if query == "q":
        return
    if query.lower() == "/history":
        show_chat_history(user_id, chat_id)
        return

    if query.lower() == "/list":
        print(list_user_chats(user_id))
        return

    if query.lower().startswith("/resume "):
        new_chat_id = query.split("/resume ")[-1].strip()
        run_chat(user_id, chat_model, retriever, chat_id=new_chat_id)
        return

    messages.append(HumanMessage(content=query))
    add_message(user_id, chat_id, "user", query)

        # Check if this is chitchat
    if is_chitchat(query):
        print("[🧭 Plan]: Chitchat detected — skipping retrieval.")

        # if we want to let llm handle this
        # response = chat_model.invoke(messages)
        # answer = clean_output(response.content)
        # print(f"\n🤖 Assistant:\n{answer}")
        # add_message(user_id, chat_id, "assistant",answer) 
        
        # if not but we still want to keep the repsone in messages

        answer="Hi, How can I help you?"
        add_message(user_id, chat_id, "assistant",answer)
        print(f"\n🤖 Assistant:\n{answer}")
        

    else:
        # Retrieve documents
        retrieved_docs = retriever.invoke(query)

        # Determine if retrieved docs are actually relevant
    
        #Use same model as your vectorstore
        embedding_model = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs={"device": "cuda","trust_remote_code":True}
        )
        if retrieved_docs and is_relevant(retrieved_docs, query,embedding_model):
            print("[🧭 Plan]: Using vectorstore context.")
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            prompt = f"""You are a helpful assistant. Use the following internal GWI insights to answer the user's question.
    Only use the context. If irrelevant, say "I don't know".
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""
            #response = chat_model.invoke(prompt)
            #print("\n🤖 Assistant:")
            stream = chat_model.stream(prompt)
            # answer = ""
            # for chunk in stream:
            #     token = chunk.get("content", "")  # .stream() yields dicts
            #     print(token, end="", flush=True)
            #     answer += token
            # print()

        else:
            print("[🧭 Plan]: No useful context. Answering freely.")
            #response = chat_model.invoke(messages)
            stream = chat_model.stream(messages)
        
        answer = ""
        print("\n🤖 Assistant:")
        for chunk in stream:
            token = chunk.content  #.get("content", "")
            print(token, end="", flush=True)
            answer += token
        print()

        # Save and return

        #answer = clean_output(response.content)
        #print(f"\n🤖 Assistant:\n{answer}")

        add_message(user_id, chat_id, "assistant",answer)

        eval_result = evaluate_response(query, answer, retrieved_docs)
        #        model_eval = evaluate_with_nli_model(query, answer)

        print("\n🧪 Rule-Based Evaluation:")
        for f in eval_result["feedback"]:
            print("-", f)
        print(f"Score: {eval_result['score']} → {eval_result['status'].upper()}")
        if eval_result['status'] == 'fail':


            duckduckgo_tool = Tool(
                name="duckduckgo_search",
                func=DuckDuckGoSearchResults().run,
                description="Search the web for current and factual information."
            )
             #context = "\n".join(snippets)
            web_results = DuckDuckGoSearchResults().run(query)
            prompt = f"""You are a helpful assistant. Use the following web search results to answer the user's question.

                Web Search Results:
                {web_results}

                Question: {query}

                Answer:"""
            # if using invoke uncomment
            #response = chat_model.invoke(prompt)
            #answer = clean_output(response.content)
            # print(f"\n🤖 Assistant:\n{answer}")
            #add_message(user_id, chat_id, "assistant",answer)
            # streaming tokens
            #response = chat_model.invoke(prompt)
            #answer = clean_output(response.content)



            print("\n🤖 Assistant:")
            stream = chat_model.stream(prompt)
            answer = ""
            for chunk in stream:
                token = chunk.content  #.get("content", "")  # .stream() yields dicts
                print(token, end="", flush=True)
                answer += token
            print()
            add_message(user_id, chat_id, "assistant",answer)



def clean_output(raw_output: str) -> str:
    """
    Clean and extract the assistant's final response from a chat-formatted string.
    """
    if "<|im_start|>assistant" in raw_output:
        parts = raw_output.split("<|im_start|>assistant")
        last_block = parts[-1]
        return last_block.split("<|im_end|>")[0].strip()

    # Fallback to alternate formats
    match = re.search(r"<\|assistant\|>(.*?)<\|", raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no special tokens, just return the whole thing
    return raw_output.strip()






def chat_menu(user_id, chat_model, retriever):
    while True:
        print("\n🧠 Chat Assistant Menu")
        print("1. Start a new chat")
        print("2. Resume a previous chat")
        print("3. List all your chats")
        print("4. Show chat history")
        print("5. Export chat to Markdown")
        print("6. Clear a chat")
        print("7. Exit")

        choice = input("\nChoose an option (1–7): ").strip()

        if choice == "1":
            run_chat(user_id, chat_model, retriever)  # Starts new chat
        elif choice == "2":
            chat_ids = list_user_chats(user_id)
            if not chat_ids:
                print("⚠️ No chats found.")
                continue
            print("Available chats:")
            for cid in chat_ids:
                print("-", cid)
            chat_id = input("Enter chat ID to resume: ").strip()
            run_chat(user_id, chat_model, retriever, chat_id=chat_id)
        elif choice == "3":
            chat_ids = list_user_chats(user_id)
            print("\n🗂️ Your Chats:")
            for cid in chat_ids:
                print("-", cid)
        elif choice == "4":
            chat_id = input("Enter chat ID to show: ").strip()
            show_chat_history(user_id, chat_id)
        elif choice == "5":
            chat_id = input("Enter chat ID to export: ").strip()
            export_chat_to_markdown(user_id, chat_id, f"{chat_id}.md")
            print(f"✅ Exported to {chat_id}.md")
        elif choice == "6":
            chat_id = input("Enter chat ID to clear: ").strip()
            db = load_conversations()
            if user_id in db and chat_id in db[user_id]:
                db[user_id][chat_id] = []
                save_conversations(db)
                print("🧹 Chat cleared.")
            else:
                print("⚠️ Chat not found.")
        elif choice == "7":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Try again.")


if __name__ == "__main__":
    user_id = input("Enter your username: ").strip()
    chat_menu(user_id, chat_model=llm, retriever=retriever)

#while True:
#    run_chat(user_id="user_001", chat_model=chat_model, retriever=retriever)
