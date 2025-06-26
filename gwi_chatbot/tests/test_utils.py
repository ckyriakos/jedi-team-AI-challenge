import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import decide_action, is_relevant, evaluate_response, generate_chat_title

class TestUtils(unittest.TestCase):

    #def test_decide_action_generate(self):
    #    mock_llm = MagicMock()
    #    mock_llm.invoke.return_value.content = "generate"
    #    result = decide_action("Tell me a joke", mock_llm)
    #    self.assertEqual(result, "generate")

    #def test_decide_action_retrieve(self):
    #    mock_llm = MagicMock()
    #    mock_llm.invoke.return_value.content = "retrieve"
    #    result = decide_action("What is the latest trend in marketing?", mock_llm)
    #    self.assertEqual(result, "retrieve")

    def test_is_relevant_positive(self):
        mock_embed_model = MagicMock()
        mock_embed_model.embed_query.side_effect = [
            np.array([1, 0]),  # query embedding
            np.array([1, 0])   # document embedding
        ]
        docs = [Document(page_content="This is relevant information.")]
        result = is_relevant(docs, "relevant", mock_embed_model, threshold=0.5)
        self.assertTrue(result)

    def test_is_relevant_negative(self):
        mock_embed_model = MagicMock()
        mock_embed_model.embed_query.side_effect = [
            np.array([1, 0]),  # query embedding
            np.array([0, 1])   # document embedding
        ]
        docs = [Document(page_content="Unrelated content.")]
        result = is_relevant(docs, "completely different", mock_embed_model, threshold=0.9)
        self.assertFalse(result)

    def test_evaluate_response_short(self):
        result = evaluate_response("Tell me about GWI", "No idea")
        self.assertEqual(result["status"], "fail")
        self.assertIn("too short", result["feedback"][0].lower())

    def test_evaluate_response_uncertain(self):
        answer = "I don't know the exact details, but I can find out."
        result = evaluate_response("What is GWI?", answer)
        self.assertIn("uncertain", " ".join(result["feedback"]).lower())
    
    def test_evaluate_response_relevance(self):
        docs = [Document(page_content="GWI is a consumer data platform that collects marketing insights.")]
        query = "What is a consumer data platform for marketing insights?"
        answer = "GWI is a platform for consumer data and marketing insights."
        result = evaluate_response(query, answer, docs)
        self.assertIn("semantically related", " ".join(result["feedback"]).lower())

    #    @patch("utils.ddg")
    #    def test_run_web_search(self, mock_ddg):
    #        mock_ddg.return_value = [
    #            {"body": "First result"}, {"body": "Second result"}, {"body": "Third result"}
    #        ]
    #        result = run_web_search("openai")
    #        self.assertEqual(result, ["First result", "Second result", "Third result"])

    def test_generate_chat_title(self):
        message = "Hello! This is a test title."
        title = generate_chat_title(message)
        self.assertEqual(title, "Hello_This_is_a_test_title")

    def test_generate_chat_title_empty(self):
        message = "!@#$%^&*"
        title = generate_chat_title(message)
        self.assertEqual(title, "chat_session")

if __name__ == "__main__":
    unittest.main()

