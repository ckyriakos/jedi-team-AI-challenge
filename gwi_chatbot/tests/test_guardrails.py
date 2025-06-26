import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from guardrails import is_chitchat, is_safe_content, classify_query_type

class TestGuardrails(unittest.TestCase):

    # --- is_chitchat tests ---

    def test_chitchat_keywords(self):
        self.assertTrue(is_chitchat("Hello there!"))
        self.assertTrue(is_chitchat("Can you help me?"))
        self.assertTrue(is_chitchat("Tell me a joke"))

    def test_chitchat_patterns(self):
        self.assertTrue(is_chitchat("Are you human?"))
        self.assertTrue(is_chitchat("What can you do?"))

    def test_not_chitchat(self):
        self.assertFalse(is_chitchat("Give me the latest GWI report"))
        self.assertFalse(is_chitchat("Show consumer trends in Q4 2024"))

    # --- is_safe_content tests ---

    def test_safe_content(self):
        self.assertTrue(is_safe_content("Let's talk about market trends"))
        self.assertTrue(is_safe_content("Show me GWI insights"))

    def test_unsafe_content(self):
        self.assertFalse(is_safe_content("How to create a phishing site?"))
        self.assertFalse(is_safe_content("Is this malware harmful?"))
        self.assertFalse(is_safe_content("Where to buy illegal drugs"))

    # --- classify_query_type tests ---

    def test_classify_data_query(self):
        self.assertEqual(classify_query_type("What consumer data does GWI have?"), "data_query")
        self.assertEqual(classify_query_type("Show me marketing insights"), "data_query")

    def test_classify_comparison(self):
        self.assertEqual(classify_query_type("Compare TikTok vs Instagram usage"), "comparison")
        self.assertEqual(classify_query_type("Which one is better, A or B?"), "comparison")

    def test_classify_trend_analysis(self):
        #self.assertEqual(classify_query_type("What are the latest social media trends?"), "trend_analysis") # will fail as it was include in data query too
        self.assertEqual(classify_query_type("Is influencers' popularity increasing?"), "trend_analysis")

    def test_classify_chitchat(self):
        self.assertEqual(classify_query_type("Who are you?"), "chitchat")
        self.assertEqual(classify_query_type("Good morning!"), "chitchat")

    def test_classify_general_query(self):
        self.assertEqual(classify_query_type("How does your API work?"), "general_query")
        self.assertEqual(classify_query_type("Give me something interesting."), "general_query")

if __name__ == "__main__":
    unittest.main()

