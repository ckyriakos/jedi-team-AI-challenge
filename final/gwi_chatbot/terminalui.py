from langchain_core.messages import HumanMessage, AIMessage
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
import duckduckgo_search
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
from utils import decide_action, evaluate_response, is_relevant, generate_chat_title
def load_conversations():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_conversations(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_chat(user_id, chat_id=None):
    db = load_conversations()
    if user_id not in db:
        db[user_id] = {}

        #if chat_id is None or chat_id not in db[user_id]:
        #chat_id = chat_id or generate_chat_title(user_input) or "chat_session"
        #db[user_id][chat_id] = []
    if not chat_id:
        chat_id = str(uuid.uuid4())
        db[user_id][chat_id] = []
        save_conversations(db)
    return chat_id, db[user_id][chat_id]

def add_message(user_id, chat_id, role, content):
    db = load_conversations()
    db[user_id][chat_id].append({"role": role, "content": content})
    save_conversations(db)



def get_langchain_messages(user_id, chat_id):
    db = load_conversations()
    messages = db.get(user_id, {}).get(chat_id, [])
    lc_msgs = []
    for m in messages:
        if m["role"] == "user":
            lc_msgs.append(HumanMessage(content=m["content"]))
        else:
            lc_msgs.append(AIMessage(content=m["content"]))
    return lc_msgs

def export_chat_to_markdown(user_id, chat_id, filename):
    db = load_conversations()
    messages = db.get(user_id, {}).get(chat_id, [])
    with open(filename, "w") as f:
        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            f.write(f"**{role}:** {m['content']}\n\n")


#@app.get("/chats/{user_id}")
def list_user_chats(user_id):
    db = load_conversations()
    return list(db.get(user_id, {}).keys())

def show_chat_history(user_id, chat_id):
    db = load_conversations()

    if user_id not in db:
        print(f"⚠️ No data found for user: {user_id}")
        return

    if chat_id not in db[user_id]:
        print(f"⚠️ No chat found with ID: {chat_id}")
        return

    messages = db[user_id][chat_id]
    print(f"\n📜 Chat History ({chat_id}):\n")
    for i, msg in enumerate(messages):
        role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
        print(f"{role}: {msg['content']}\n")

def delete_chat(user_id: str, chat_id: str) -> bool:
    db = load_conversations()
    if user_id in db and chat_id in db[user_id]:
        del db[user_id][chat_id]
        save_conversations(db)
        return True
    return False
