import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class EvaluationResult:
    score: float
    status: str
    feedback: List[str]
    confidence: float
    improvement_suggestions: List[str]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class EnhancedEvaluator:
    def __init__(self):
        self.evaluation_history = []
        self.performance_metrics = defaultdict(list)
        
    def evaluate_response(self, query: str, answer, retrieved_docs: list = None, 
                         source: str = "unknown") -> EvaluationResult:
        """Enhanced evaluation with more sophisticated scoring"""
        score = 0
        feedback = []
        improvement_suggestions = []
        
        # Handle answer as string or list
        if isinstance(answer, list):
            answer_text = " ".join(str(item) for item in answer)
        else:
            answer_text = str(answer)
        
        # Rule 1: Answer length and completeness
        word_count = len(answer_text.split())
        if word_count < 10:
            feedback.append("Answer is too short and lacks detail.")
            improvement_suggestions.append("Provide more comprehensive responses")
            score -= 2
        elif word_count < 30:
            feedback.append("Answer could be more detailed.")
            score -= 0.5
        else:
            feedback.append("Answer has appropriate length.")
            score += 1
            
        # Rule 2: Uncertainty patterns (more nuanced)
        uncertainty_phrases = [
            "i don't know", "not sure", "unclear", "unable to", 
            "can't find", "no information", "sorry, but"
        ]
        uncertainty_count = sum(1 for phrase in uncertainty_phrases 
                              if phrase in answer_text.lower())
        
        if uncertainty_count > 2:
            feedback.append("Answer shows high uncertainty.")
            improvement_suggestions.append("Try alternative search strategies or rephrase query")
            score -= 2
        elif uncertainty_count == 1:
            feedback.append("Answer shows some uncertainty.")
            score -= 0.5
        else:
            feedback.append("Answer shows confidence.")
            score += 1
            
        # Rule 3: Query relevance (enhanced)
        query_terms = set(query.lower().split())
        answer_terms = set(answer_text.lower().split())
        
        # Direct term overlap
        overlap = len(query_terms.intersection(answer_terms))
        overlap_ratio = overlap / len(query_terms) if query_terms else 0
        
        if overlap_ratio > 0.5:
            feedback.append("Answer directly addresses query terms.")
            score += 1
        elif overlap_ratio > 0.2:
            feedback.append("Answer partially addresses query.")
            score += 0.5
        else:
            feedback.append("Answer may not be relevant to query.")
            improvement_suggestions.append("Ensure response directly addresses the question")
            score -= 1
            
        # Rule 4: Context utilization (if retrieved docs exist)
        if retrieved_docs:
            # Handle different document formats
            context_parts = []
            for doc in retrieved_docs:
                if hasattr(doc, 'page_content'):
                    # Document object with page_content attribute
                    context_parts.append(doc.page_content.lower())
                elif isinstance(doc, dict) and 'page_content' in doc:
                    # Dictionary with page_content key
                    context_parts.append(doc['page_content'].lower())
                elif isinstance(doc, str):
                    # Plain string
                    context_parts.append(doc.lower())
                else:
                    # Convert to string as fallback
                    context_parts.append(str(doc).lower())
            
            context = " ".join(context_parts)
            context_terms = set(context.split())
            
            # How well does the answer use the retrieved context?
            answer_context_overlap = len(answer_terms.intersection(context_terms))
            context_utilization = answer_context_overlap / len(context_terms) if context_terms else 0
            
            if context_utilization > 0.3:
                feedback.append("Answer effectively uses retrieved context.")
                score += 1
            elif context_utilization > 0.1:
                feedback.append("Answer partially uses retrieved context.")
                score += 0.5
            else:
                feedback.append("Answer doesn't utilize retrieved context well.")
                improvement_suggestions.append("Better integrate retrieved information into response")
                score -= 1
                
        # Rule 5: Hallucination detection (basic)
        definitive_phrases = ["according to", "research shows", "studies indicate", 
                             "data reveals", "statistics show"]
        if any(phrase in answer_text.lower() for phrase in definitive_phrases) and not retrieved_docs:
            feedback.append("Answer makes definitive claims without supporting context.")
            improvement_suggestions.append("Avoid unsupported claims or clearly indicate uncertainty")
            score -= 1
            
        # Rule 6: Actionability and usefulness
        actionable_phrases = ["you can", "try", "consider", "recommend", "suggest"]
        if any(phrase in answer_text.lower() for phrase in actionable_phrases):
            feedback.append("Answer provides actionable insights.")
            score += 0.5
            
        # Calculate confidence based on various factors
        confidence = min(1.0, max(0.0, (score + 3) / 6))  # Normalize to 0-1
        
        # Determine status with more nuanced thresholds
        if score >= 2:
            status = "excellent"
        elif score >= 1:
            status = "good"
        elif score >= 0:
            status = "acceptable"
        elif score >= -1:
            status = "poor"
        else:
            status = "fail"
            
        result = EvaluationResult(
            score=score,
            status=status,
            feedback=feedback,
            confidence=confidence,
            improvement_suggestions=improvement_suggestions
        )
        
        # Store for learning
        self.evaluation_history.append({
            'query': query,
            'answer': answer_text,  # Store the processed text version
            'source': source,
            'result': result,
            'timestamp': time.time()
        })
        
        self.performance_metrics[source].append(score)
        
        return result
    
    def get_performance_insights(self) -> Dict:
        """Analyze performance patterns for iterative improvement"""
        if not self.evaluation_history:
            return {"message": "No evaluation history available"}
            
        insights = {
            "total_evaluations": len(self.evaluation_history),
            "average_score": sum(h['result'].score for h in self.evaluation_history) / len(self.evaluation_history),
            "source_performance": {},
            "common_issues": defaultdict(int),
            "improvement_trends": [],
            "recommendations": []
        }
        
        # Performance by source
        for source, scores in self.performance_metrics.items():
            insights["source_performance"][source] = {
                "average_score": sum(scores) / len(scores),
                "total_queries": len(scores),
                "success_rate": len([s for s in scores if s >= 0]) / len(scores)
            }
            
        # Common issues analysis
        for eval_record in self.evaluation_history:
            for suggestion in eval_record['result'].improvement_suggestions:
                insights["common_issues"][suggestion] += 1
                
        # Generate recommendations based on patterns
        source_perf = insights["source_performance"]
        
        if "vectorstore" in source_perf and source_perf["vectorstore"]["average_score"] < 0:
            insights["recommendations"].append("Consider improving vectorstore relevance threshold")
            
        if "knowledge_base" in source_perf and source_perf["knowledge_base"]["average_score"] < -0.5:
            insights["recommendations"].append("Knowledge base responses may need better prompting")
            
        if "web_search" in source_perf and source_perf["web_search"]["success_rate"] < 0.7:
            insights["recommendations"].append("Web search queries may need refinement")
            
        return insights

# Usage in your chat endpoint
def adaptive_agent_stream(user_input: str, user_id: str, chat_id: str, evaluator: EnhancedEvaluator):
    """Modified agent stream with iterative improvement"""
    
    # Get performance insights to adapt strategy
    insights = evaluator.get_performance_insights()
    
    # Adapt strategy based on historical performance
    vectorstore_threshold = 0.7  # Default
    if "vectorstore" in insights.get("source_performance", {}):
        vs_perf = insights["source_performance"]["vectorstore"]["average_score"]
        if vs_perf < -0.5:
            vectorstore_threshold = 0.5  # Lower threshold if vectorstore usually fails
            yield f"data: [ADAPTATION] Lowering vectorstore threshold based on performance\n\n"
    
    # Step 1: Vectorstore with adaptive threshold
    yield f"data: [THOUGHT] Searching vectorstore (threshold: {vectorstore_threshold})...\n\n"
    retrieved_docs = retriever.invoke(user_input)
    
    if retrieved_docs and is_relevant(retrieved_docs, user_input, embedding_model, threshold=vectorstore_threshold):
        # Generate response
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"""Based on performance analysis, provide a comprehensive answer using this context.
        
Context: {context}
Question: {user_input}
Answer:"""
        
        stream = llm.stream(prompt)
        answer = ""
        for chunk in stream:
            token = chunk.content
            answer += token
            yield f"data: {token}\n\n"
        
        # Enhanced evaluation
        eval_result = evaluator.evaluate_response(user_input, answer, retrieved_docs, "vectorstore")
        yield f"data: [EVALUATION] {eval_result.status.upper()} (Score: {eval_result.score:.1f}, Confidence: {eval_result.confidence:.2f})\n\n"
        
        # Show improvement suggestions if any
        if eval_result.improvement_suggestions:
            yield f"data: [LEARNING] Suggestions: {'; '.join(eval_result.improvement_suggestions)}\n\n"
        
        # Decide whether to continue based on enhanced evaluation
        if eval_result.status in ["excellent", "good"]:
            yield "data: [END]\n\n"
            return
        elif eval_result.status == "acceptable" and eval_result.confidence > 0.6:
            yield "data: [END]\n\n"
            return
        else:
            yield f"data: [THOUGHT] Response quality insufficient ({eval_result.status}). Trying next approach...\n\n"
    
    # Continue with knowledge base and web search with similar adaptive logic...
    # (Similar pattern for other sources)

# Example usage
evaluator = EnhancedEvaluator()

# In your chat endpoint, replace the simple evaluate_response call with:
# eval_result = evaluator.evaluate_response(user_input, answer, retrieved_docs, "vectorstore")

# Periodically check insights:
# insights = evaluator.get_performance_insights()
# print(f"System performance: {insights}")
