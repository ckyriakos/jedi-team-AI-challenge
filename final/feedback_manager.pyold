import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime, timedelta
import re
from copy import deepcopy

@dataclass
class AdaptiveConfig:
    """Configuration that adapts based on performance"""
    vectorstore_threshold: float = 0.7
    web_search_threshold: float = 0.5
    response_length_target: int = 100
    uncertainty_tolerance: int = 1
    context_utilization_threshold: float = 0.2
    
    # Strategy weights (0-1, higher = prefer this source)
    vectorstore_weight: float = 1.0
    knowledge_base_weight: float = 0.8
    web_search_weight: float = 0.6
    
    # Adaptation parameters
    min_samples_for_adaptation: int = 5
    adaptation_rate: float = 0.1  # How quickly to adapt (0-1)
    
    def to_dict(self):
        return asdict(self)
    
    def __eq__(self, other):
        if not isinstance(other, AdaptiveConfig):
            return False
        return asdict(self) == asdict(other)

class AdaptiveFeedbackManager:
    def __init__(self, feedback_file="feedback_log.jsonl", config_file="adaptive_config.json"):
        self.feedback_file = feedback_file
        self.config_file = config_file
        self.config = self.load_config()
        self.performance_history = defaultdict(list)
        self.query_patterns = defaultdict(lambda: {"success": 0, "total": 0, "strategies": []})
        self.last_adaptation_time = time.time()
        self.adaptation_cooldown = 300  # 5 minutes between adaptations
        
    def load_config(self) -> AdaptiveConfig:
        """Load adaptive configuration or create default"""
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                return AdaptiveConfig(**data)
        except FileNotFoundError:
            config = AdaptiveConfig()
            self.save_config(config)
            return config
    
    def save_config(self, config: AdaptiveConfig):
        """Save current configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        print(f"🔄 Configuration updated and saved: {config.to_dict()}")
    
    def add_feedback_with_context(self, user_id: str, chat_id: str, query: str, 
                                answer: str, was_helpful: bool, eval_result: dict,
                                source: str, response_time: float):
        """Enhanced feedback with more context"""
        feedback_entry = {
            "timestamp": time.time(),
            "user_id": user_id,
            "chat_id": chat_id,
            "query": query,
            "answer": answer,
            "was_helpful": was_helpful,
            "source": source,
            "response_time": response_time,
            "eval_score": eval_result.get('score', 0),
            "eval_status": eval_result.get('status', 'unknown'),
            "confidence": eval_result.get('confidence', 0),
            "query_length": len(query.split()),
            "answer_length": len(answer.split()),
            "query_type": self.classify_query_complexity(query)
        }
        
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        
        # Update patterns immediately
        self.update_query_patterns(query, was_helpful, source, eval_result)
        
        # Check if we should adapt configuration
        if time.time() - self.last_adaptation_time > self.adaptation_cooldown:
            self._maybe_adapt_config()
    
    def classify_query_complexity(self, query: str) -> str:
        """Classify query by complexity and type"""
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Check for specific patterns
        if any(word in query_lower for word in ['what', 'who', 'when', 'where']):
            complexity = 'factual'
        elif any(word in query_lower for word in ['how', 'why', 'explain']):
            complexity = 'analytical'
        elif any(word in query_lower for word in ['compare', 'analyze', 'evaluate']):
            complexity = 'complex'
        elif word_count > 15:
            complexity = 'complex'
        elif word_count > 8:
            complexity = 'analytical'
        else:
            complexity = 'simple'
            
        return complexity
    
    def update_query_patterns(self, query: str, was_helpful: bool, source: str, eval_result: dict):
        """Track patterns for different query types"""
        query_type = self.classify_query_complexity(query)
        pattern_key = f"{query_type}_{source}"
        
        self.query_patterns[pattern_key]["total"] += 1
        if was_helpful:
            self.query_patterns[pattern_key]["success"] += 1
        
        self.query_patterns[pattern_key]["strategies"].append({
            "helpful": was_helpful,
            "score": eval_result.get('score', 0),
            "timestamp": time.time()
        })
    
    def analyze_performance_trends(self, days_back: int = 7) -> Dict:
        """Analyze recent performance trends"""
        cutoff_time = time.time() - (days_back * 24 * 3600)
        recent_feedback = []
        
        try:
            with open(self.feedback_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get('timestamp', 0) > cutoff_time:
                        recent_feedback.append(entry)
        except FileNotFoundError:
            return {"error": "No feedback data available"}
        
        if not recent_feedback:
            return {"error": "No recent feedback data"}
        
        # Analyze by source
        source_performance = defaultdict(lambda: {"helpful": 0, "total": 0, "avg_score": 0, "avg_time": 0})
        
        for entry in recent_feedback:
            source = entry.get('source', 'unknown')
            source_performance[source]['total'] += 1
            if entry.get('was_helpful', False):
                source_performance[source]['helpful'] += 1
            source_performance[source]['avg_score'] += entry.get('eval_score', 0)
            source_performance[source]['avg_time'] += entry.get('response_time', 0)
        
        # Calculate averages and rates
        for source, data in source_performance.items():
            if data['total'] > 0:
                data['success_rate'] = data['helpful'] / data['total']
                data['avg_score'] /= data['total']
                data['avg_time'] /= data['total']
        
        return dict(source_performance)
    
    def _maybe_adapt_config(self):
        """Check if configuration should be adapted based on recent performance"""
        performance_trends = self.analyze_performance_trends()
        
        if not performance_trends or 'error' in performance_trends:
            return
        
        config_changed = False
        original_config = deepcopy(self.config)
        
        # Adapt based on performance trends
        for source, perf in performance_trends.items():
            if perf['total'] < self.config.min_samples_for_adaptation:
                continue
                
            success_rate = perf['success_rate']
            adaptation_factor = self.config.adaptation_rate
            
            if source == 'vectorstore':
                if success_rate < 0.4:  # Poor performance
                    new_threshold = max(0.3, self.config.vectorstore_threshold - adaptation_factor)
                    new_weight = max(0.3, self.config.vectorstore_weight - adaptation_factor)
                    
                    if abs(new_threshold - self.config.vectorstore_threshold) > 0.05:
                        self.config.vectorstore_threshold = new_threshold
                        config_changed = True
                        print(f"📉 Lowered vectorstore threshold to {new_threshold:.2f} (success rate: {success_rate:.2f})")
                    
                    if abs(new_weight - self.config.vectorstore_weight) > 0.05:
                        self.config.vectorstore_weight = new_weight
                        config_changed = True
                        print(f"📉 Lowered vectorstore weight to {new_weight:.2f}")
                        
                elif success_rate > 0.8:  # Excellent performance
                    new_threshold = min(0.9, self.config.vectorstore_threshold + adaptation_factor)
                    new_weight = min(1.5, self.config.vectorstore_weight + adaptation_factor)
                    
                    if abs(new_threshold - self.config.vectorstore_threshold) > 0.05:
                        self.config.vectorstore_threshold = new_threshold
                        config_changed = True
                        print(f"📈 Raised vectorstore threshold to {new_threshold:.2f} (success rate: {success_rate:.2f})")
                    
                    if abs(new_weight - self.config.vectorstore_weight) > 0.05:
                        self.config.vectorstore_weight = new_weight
                        config_changed = True
                        print(f"📈 Raised vectorstore weight to {new_weight:.2f}")
            
            elif source == 'web_search':
                if success_rate > 0.7:  # Good performance
                    new_weight = min(1.2, self.config.web_search_weight + adaptation_factor)
                    
                    if abs(new_weight - self.config.web_search_weight) > 0.05:
                        self.config.web_search_weight = new_weight
                        config_changed = True
                        print(f"📈 Increased web search weight to {new_weight:.2f} (success rate: {success_rate:.2f})")
                
                elif success_rate < 0.3:  # Poor performance
                    new_weight = max(0.2, self.config.web_search_weight - adaptation_factor)
                    
                    if abs(new_weight - self.config.web_search_weight) > 0.05:
                        self.config.web_search_weight = new_weight
                        config_changed = True
                        print(f"📉 Decreased web search weight to {new_weight:.2f} (success rate: {success_rate:.2f})")
            
            elif source == 'knowledge_base':
                if success_rate > 0.7:  # Good performance
                    new_weight = min(1.2, self.config.knowledge_base_weight + adaptation_factor)
                    
                    if abs(new_weight - self.config.knowledge_base_weight) > 0.05:
                        self.config.knowledge_base_weight = new_weight
                        config_changed = True
                        print(f"📈 Increased knowledge base weight to {new_weight:.2f} (success rate: {success_rate:.2f})")
                
                elif success_rate < 0.3:  # Poor performance
                    new_weight = max(0.2, self.config.knowledge_base_weight - adaptation_factor)
                    
                    if abs(new_weight - self.config.knowledge_base_weight) > 0.05:
                        self.config.knowledge_base_weight = new_weight
                        config_changed = True
                        print(f"📉 Decreased knowledge base weight to {new_weight:.2f} (success rate: {success_rate:.2f})")
        
        if config_changed:
            self.save_config(self.config)
            self.last_adaptation_time = time.time()
            print(f"🔄 Configuration adapted based on performance data")
        else:
            print(f"✅ Configuration stable - no changes needed")
    
    def get_adaptive_strategy(self, query: str) -> Tuple[AdaptiveConfig, Dict]:
        """Get current adaptive strategy - no longer modifies config here"""
        query_type = self.classify_query_complexity(query)
        performance_trends = self.analyze_performance_trends()
        
        # Return current config (modifications happen in _maybe_adapt_config)
        strategy_explanation = []
        
        # Provide explanation of current strategy
        if performance_trends and 'error' not in performance_trends:
            for source, perf in performance_trends.items():
                if perf['total'] >= self.config.min_samples_for_adaptation:
                    success_rate = perf['success_rate']
                    if source == 'vectorstore':
                        if success_rate > 0.8:
                            strategy_explanation.append(f"Vectorstore performing excellently ({success_rate:.1%} success)")
                        elif success_rate < 0.4:
                            strategy_explanation.append(f"Vectorstore underperforming ({success_rate:.1%} success)")
                    elif source == 'web_search':
                        if success_rate > 0.7:
                            strategy_explanation.append(f"Web search performing well ({success_rate:.1%} success)")
                        elif success_rate < 0.3:
                            strategy_explanation.append(f"Web search underperforming ({success_rate:.1%} success)")
        
        # Adjust strategy based on query type patterns
        temp_config = deepcopy(self.config)
        for pattern_key, pattern_data in self.query_patterns.items():
            if pattern_key.startswith(query_type) and pattern_data['total'] >= 3:
                success_rate = pattern_data['success'] / pattern_data['total']
                source = pattern_key.split('_', 1)[1]
                
                if success_rate < 0.3:
                    # Temporarily reduce preference for this source for this query type
                    if source == 'vectorstore':
                        temp_config.vectorstore_weight *= 0.8
                    elif source == 'knowledge_base':
                        temp_config.knowledge_base_weight *= 0.8
                    elif source == 'web_search':
                        temp_config.web_search_weight *= 0.8
                    
                    strategy_explanation.append(f"Reduced {source} preference for {query_type} queries ({success_rate:.1%} success)")
        
        return temp_config, {
            "strategy_explanation": strategy_explanation,
            "query_type": query_type,
            "performance_trends": performance_trends,
            "config_source": "current" if temp_config == self.config else "adapted"
        }
    
    def suggest_prompt_improvements(self) -> List[str]:
        """Analyze feedback to suggest prompt improvements"""
        suggestions = []
        
        try:
            with open(self.feedback_file, 'r') as f:
                feedback_data = [json.loads(line) for line in f]
        except FileNotFoundError:
            return ["No feedback data available for analysis"]
        
        if not feedback_data:
            return ["No feedback data available"]
        
        # Analyze common failure patterns
        failed_responses = [f for f in feedback_data if not f.get('was_helpful', False)]
        
        if len(failed_responses) > len(feedback_data) * 0.3:  # More than 30% failure rate
            suggestions.append("Consider revising base prompts - high failure rate detected")
        
        # Analyze by source
        source_failures = defaultdict(list)
        for failure in failed_responses:
            source_failures[failure.get('source', 'unknown')].append(failure)
        
        for source, failures in source_failures.items():
            if len(failures) > 5:  # Significant number of failures
                common_words = Counter()
                for failure in failures:
                    common_words.update(failure.get('query', '').lower().split())
                
                most_common = [word for word, count in common_words.most_common(3) 
                             if len(word) > 3]  # Ignore short words
                
                if most_common:
                    suggestions.append(f"Improve {source} prompts for queries about: {', '.join(most_common)}")
        
        return suggestions or ["No specific improvements identified"]
    
    def force_adaptation(self):
        """Force an immediate adaptation check (useful for testing)"""
        print("🔄 Forcing configuration adaptation...")
        self.last_adaptation_time = 0  # Reset cooldown
        self._maybe_adapt_config()
    
    def reset_config(self):
        """Reset configuration to defaults"""
        self.config = AdaptiveConfig()
        self.save_config(self.config)
        print("🔄 Configuration reset to defaults")
    
    def get_config_history(self) -> Dict:
        """Get configuration change history"""
        # In a real implementation, you'd track config changes over time
        return {
            "current_config": self.config.to_dict(),
            "last_adaptation": self.last_adaptation_time,
            "adaptation_cooldown_remaining": max(0, self.adaptation_cooldown - (time.time() - self.last_adaptation_time))
        }

# Integration with your existing system
class EnhancedChatEndpoint:
    def __init__(self):
        self.feedback_manager = AdaptiveFeedbackManager()
        self.evaluator = None  # Your existing EnhancedEvaluator
    
    def adaptive_agent_stream(self, user_input: str, user_id: str, chat_id: str):
        """Your existing agent_stream but with adaptive behavior"""
        start_time = time.time()
        
        # Get adaptive strategy
        adaptive_config, strategy_info = self.feedback_manager.get_adaptive_strategy(user_input)
        
        yield f"data: [ADAPTATION] Using adaptive strategy for {strategy_info['query_type']} query\n\n"
        
        if strategy_info['strategy_explanation']:
            for explanation in strategy_info['strategy_explanation']:
                yield f"data: [STRATEGY] {explanation}\n\n"
        
        # Step 1: Vectorstore with adaptive threshold
        yield f"data: [THOUGHT] Searching vectorstore (threshold: {adaptive_config.vectorstore_threshold:.2f}, weight: {adaptive_config.vectorstore_weight:.2f})...\n\n"
        
        # Your existing vectorstore logic here, but use adaptive_config.vectorstore_threshold
        # retrieved_docs = retriever.invoke(user_input)
        # if retrieved_docs and is_relevant(retrieved_docs, user_input, embedding_model, threshold=adaptive_config.vectorstore_threshold):
        #     ... generate response ...
        #     eval_result = self.evaluator.evaluate_response(user_input, answer, retrieved_docs, "vectorstore")
        #     
        #     # Store feedback with context
        #     response_time = time.time() - start_time
        #     self.feedback_manager.add_feedback_with_context(
        #         user_id, chat_id, user_input, answer, 
        #         eval_result.score > 0.7,  # Improved helpfulness heuristic
        #         eval_result.__dict__, "vectorstore", response_time
        #     )
        #     
        #     if eval_result.score >= adaptive_config.vectorstore_weight:  # Use adaptive weight
        #         yield "data: [END]\n\n"
        #         return
        
        # Continue with knowledge base and web search using similar adaptive logic...
        
    def get_performance_dashboard(self) -> Dict:
        """Get comprehensive performance insights"""
        trends = self.feedback_manager.analyze_performance_trends()
        suggestions = self.feedback_manager.suggest_prompt_improvements()
        config_history = self.feedback_manager.get_config_history()
        
        return {
            "performance_trends": trends,
            "improvement_suggestions": suggestions,
            "current_config": self.feedback_manager.config.to_dict(),
            "query_patterns": dict(self.feedback_manager.query_patterns),
            "config_history": config_history
        }
    
    def force_adaptation(self):
        """Force adaptation for testing purposes"""
        return self.feedback_manager.force_adaptation()
    
    def reset_config(self):
        """Reset configuration to defaults"""
        return self.feedback_manager.reset_config()

# Utility functions for testing and debugging
def simulate_feedback_data(feedback_manager: AdaptiveFeedbackManager, num_entries: int = 50):
    """Generate sample feedback data for testing"""
    import random
    
    sources = ['vectorstore', 'knowledge_base', 'web_search']
    query_types = ['factual', 'analytical', 'complex', 'simple']
    
    for i in range(num_entries):
        source = random.choice(sources)
        query_type = random.choice(query_types)
        
        # Simulate different performance patterns
        if source == 'vectorstore':
            success_rate = 0.85 if query_type in ['factual', 'simple'] else 0.6
        elif source == 'knowledge_base':
            success_rate = 0.75
        else:  # web_search
            success_rate = 0.7 if query_type in ['analytical', 'complex'] else 0.5
        
        was_helpful = random.random() < success_rate
        
        feedback_manager.add_feedback_with_context(
            user_id=f"user_{i%10}",
            chat_id=f"chat_{i%20}",
            query=f"Sample {query_type} query {i}",
            answer=f"Sample answer {i}",
            was_helpful=was_helpful,
            eval_result={"score": random.uniform(0.3, 0.9), "confidence": random.uniform(0.5, 1.0)},
            source=source,
            response_time=random.uniform(0.5, 3.0)
        )
    
    print(f"✅ Generated {num_entries} sample feedback entries")

# Usage example for integration:
"""
# In your main.py:

chat_system = EnhancedChatEndpoint()

# Optional: Generate test data
simulate_feedback_data(chat_system.feedback_manager, 100)

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    user_id = data.get("user_id", "default")
    chat_id = data.get("chat_id", None)
    
    # ... safety checks ...
    
    return StreamingResponse(
        chat_system.adaptive_agent_stream(user_input, user_id, chat_id), 
        media_type="text/event-stream"
    )

@app.get("/performance")
async def get_performance():
    return JSONResponse(content=chat_system.get_performance_dashboard())

@app.post("/force-adaptation")
async def force_adaptation():
    chat_system.force_adaptation()
    return {"message": "Adaptation forced"}

@app.post("/reset-config")
async def reset_config():
    chat_system.reset_config()
    return {"message": "Configuration reset"}
"""

feedback_store = "feedback_log.jsonl"

def add_feedback(user_id, chat_id, query, answer, was_helpful: bool):
    with open(feedback_store, "a") as f:
        f.write(json.dumps({
            "user_id": user_id,
            "chat_id": chat_id,
            "query": query,
            "answer": answer,
            "was_helpful": was_helpful
        }) + "\n")
