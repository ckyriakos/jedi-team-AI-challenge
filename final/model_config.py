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
    num_predict=512,
    num_gpu=0,
    extract_reasoning=True,
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
