
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


def decide_action(query, planner_llm):

    # --- Planning Prompt ---

    PLANNER_PROMPT = """
        You are a planning assistant.

        You have access to a knowledge base of GWI insights. If a user asks a question that could be answered from those insights, you should choose to 'retrieve'.

        Only respond with one word: 'retrieve' or 'generate'.

        Question:
    {question}
    """

    prompt = PromptTemplate.from_template(PLANNER_PROMPT)
    full_prompt = prompt.format(question=query)
    return planner_llm.invoke(full_prompt).content.strip().lower()

    #def handle_query(query, planner_llm, qa_chain, llm):
    #    action = decide_action(query, planner_llm)
    #    print("[🧭 Plan]:", action)
    #
    #    if action == "retrieve":
    #        print("→ Using internal vectorstore.")
    #        return qa_chain.run(query)
    #    else:
    #        print("→ Generating directly.")
    #        return llm.invoke(query)
    #
    #def is_chitchat(query):
    #    chitchat_keywords = ["hello", "hi", "how are you", "what's up", "who are you", "good morning"]
    #    return any(word in query.lower() for word in chitchat_keywords)
    #def is_chitchat(query: str) -> bool:
    #    chitchat_keywords = [
    #        "hello", "hi", "hey", "how are you", "what's up", 
    #        "good morning", "good afternoon", "who are you", 
    #        "are you a bot", "tell me a joke", "thank you", "bye"
    #    ]
    #    return any(phrase in query.lower() for phrase in chitchat_keywords)


def is_relevant(docs, query, embed_model, threshold=0.7):
    if not docs:
        return False
    query_embedding = embed_model.embed_query(query)
    
    top_docs_text = " ".join([doc.page_content.replace("\n", " ") for doc in docs[:3]])
    #doc_embedding = embed_model.embed_query(doc.page_content for doc in docs[:3])  # Take top 3 doc
    doc_embedding = embed_model.embed_query(top_docs_text)
    scores = cosine_similarity([query_embedding], [doc_embedding])[0][0] #add this if you use doc[0].page_content instead
    # return any(score >= threshold for score in scores)
    return np.mean(scores) >= threshold # supposedly avoids false positives


def evaluate_response(query: str, answer: str, retrieved_docs: list = None) -> dict:
    score = 0
    feedback = []

    # Rule 1: Minimum answer length
    if len(answer.strip()) < 30:
        feedback.append("Answer is too short.")
        score -= 1

    # Rule 2: Answer contains "I don't know" or similar uncertainty
    if "i don't know" in answer.lower():
        feedback.append("Agent was uncertain or failed to answer.")
        score -= 1

    # Rule 3: Relevance to query
    if retrieved_docs:
        relevant_terms = query.lower().split()
        context = " ".join([doc.page_content.lower() for doc in retrieved_docs])
        match_count = sum(term in context for term in relevant_terms)
        match_ratio = match_count / len(relevant_terms) if relevant_terms else 0
        if match_ratio > 0.7:
            score += 1
            feedback.append("Answer is semantically related to retrieved context.")
        else:
            feedback.append("Answer may not be relevant to retrieved context.")
            score -= 1

    # Final status
    status = "pass" if score >= 0 else "fail"

    return {
        "score": score,
        "status": status,
        "feedback": feedback
    }

#import duckduckgo_search

#def run_web_search(query):
#    results = duckduckgo_search.ddg(query, max_results=3)
#    return [res["body"] for res in results]

import re
def generate_chat_title(message: str) -> str:
    # Lowercase, remove special chars except spaces, truncate words
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', message).strip()
    words = sanitized.split()
    title = "_".join(words[:6])  # first 6 words joined by _
    if not title:
        title = "chat_session"
    return title

# Load once
#nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",device=-1)

#def evaluate_with_nli_model(query: str, answer: str) -> dict:
#    candidate_labels = ["fully answers the question", "partially answers", "irrelevant or hallucinated"]
#    result = nli(answer, candidate_labels, hypot`hesis_template=f"This text {{}}, given the question: '{query}'.")
#
#    label = result["labels"][0]
#    score = result["scores"][0]
#
#   return {
#        "best_label": label,
#        "score": score,
#        "raw": result
#    }

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
