import os
import json
from typing import Dict, List, Any, Optional
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Dataset, Example
from langchain_anthropic import ChatAnthropic
from langgraph_agent import get_agent  # Your enhanced agent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBobRossEvaluator:
    """Enhanced evaluation with confidence-based metrics"""
    
    def __init__(self):
        self.client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        self.agent = get_agent()  # Your enhanced agent
        self.llm = ChatAnthropic(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name="claude-sonnet-4-20250514"
        )
    
    def create_enhanced_evaluation_dataset(self) -> str:
        """Create enhanced dataset with confidence expectations"""
        dataset_name = "bob-ross-confidence-enhanced-eval"
        
        test_cases = [
            {
                "input": "from langchain import LLMChain",
                "expected_query_type": "code_explanation",
                "expected_keywords": ["LLMChain", "sequence", "prompt", "model"],
                "expected_confidence_range": [0.7, 1.0],  # High confidence expected
                "difficulty": "easy",
                "confidence_rationale": "Clear import statement should have high classification confidence"
            },
            {
                "input": "What does this thing do?",
                "expected_query_type": "general_help",
                "expected_keywords": ["help", "explanation"],
                "expected_confidence_range": [0.2, 0.5],  # Low confidence expected
                "difficulty": "hard",
                "confidence_rationale": "Vague question should trigger low confidence and clarification request"
            },
            {
                "input": "ImportError: No module named 'langchain'",
                "expected_query_type": "error_help",
                "expected_keywords": ["install", "pip", "module", "import"],
                "expected_confidence_range": [0.8, 1.0],  # Very high confidence expected
                "difficulty": "easy",
                "confidence_rationale": "Clear error message should have very high confidence"
            },
            {
                "input": "chain = prompt | model | output_parser",
                "expected_query_type": "code_explanation",
                "expected_keywords": ["LCEL", "chain", "pipe", "prompt", "model", "parser"],
                "expected_confidence_range": [0.6, 0.9],  # Good confidence expected
                "difficulty": "medium",
                "confidence_rationale": "LCEL pattern should be recognized with good confidence"
            },
            {
                "input": "How do I use streaming with ChatAnthropic for real-time responses?",
                "expected_query_type": "api_usage",
                "expected_keywords": ["ChatAnthropic", "streaming", "async", "callback"],
                "expected_confidence_range": [0.7, 0.9],  # High confidence expected
                "difficulty": "medium",
                "confidence_rationale": "Specific API question should have high confidence"
            }
        ]
        
        examples = []
        for i, case in enumerate(test_cases):
            example = Example(
                inputs={"text": case["input"]},
                outputs={
                    "expected_query_type": case["expected_query_type"],
                    "expected_keywords": case["expected_keywords"],
                    "expected_confidence_range": case["expected_confidence_range"],
                    "confidence_rationale": case["confidence_rationale"],
                    "difficulty": case["difficulty"]
                }
            )
            examples.append(example)
        
        try:
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Enhanced evaluation dataset with confidence scoring expectations"
            )
            logger.info(f"Created enhanced dataset: {dataset_name}")
        except Exception:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            logger.info(f"Using existing enhanced dataset: {dataset_name}")
        
        for example in examples:
            try:
                self.client.create_example(
                    inputs=example.inputs,
                    outputs=example.outputs,
                    dataset_id=dataset.id
                )
            except Exception as e:
                logger.warning(f"Example might already exist: {e}")
        
        return dataset_name
    
    def confidence_calibration_evaluator(self, run, example) -> Dict[str, Any]:
        """Evaluate how well-calibrated the confidence scores are"""
        try:
            # Get agent outputs
            overall_confidence = run.outputs.get("overall_confidence", 0.0)
            classification_confidence = run.outputs.get("classification_confidence", 0.0)
            context_confidence = run.outputs.get("context_confidence", 0.0)
            
            # Get expected confidence range
            expected_range = example.outputs.get("expected_confidence_range", [0.0, 1.0])
            min_expected, max_expected = expected_range
            
            # Check if confidence is within expected range
            confidence_appropriate = min_expected <= overall_confidence <= max_expected
            
            # Calculate calibration score
            if confidence_appropriate:
                calibration_score = 1.0
            else:
                # Penalize based on how far off we are
                if overall_confidence < min_expected:
                    distance = min_expected - overall_confidence
                elif overall_confidence > max_expected:
                    distance = overall_confidence - max_expected
                calibration_score = max(0.0, 1.0 - distance * 2)  # Penalty factor of 2
            
            return {
                "key": "confidence_calibration",
                "score": calibration_score,
                "comment": f"Overall confidence: {overall_confidence:.2f}, Expected: [{min_expected:.2f}-{max_expected:.2f}], Appropriate: {confidence_appropriate}"
            }
            
        except Exception as e:
            logger.error(f"Error in confidence calibration evaluator: {e}")
            return {"key": "confidence_calibration", "score": 0, "comment": f"Evaluation error: {e}"}
    
    def context_quality_evaluator(self, run, example) -> Dict[str, Any]:
        """Evaluate quality of context retrieval and categorization"""
        try:
            context_category = run.outputs.get("context_category", "unknown")
            context_confidence = run.outputs.get("context_confidence", 0.0)
            query_type = run.outputs.get("query_type", "")
            
            # Score based on context specificity
            context_quality_score = 0.5  # Base score
            
            # Bonus for specific context categories
            specific_categories = [
                "langgraph_specific", "lcel_specific", "vector_store_specific", 
                "chain_specific", "agent_specific", "troubleshooting_specific"
            ]
            
            if context_category in specific_categories:
                context_quality_score += 0.3
            elif context_category == "general_langchain":
                context_quality_score += 0.1
            
            # Factor in context confidence
            context_quality_score = (context_quality_score + context_confidence) / 2
            
            return {
                "key": "context_quality",
                "score": context_quality_score,
                "comment": f"Context category: {context_category}, Context confidence: {context_confidence:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Error in context quality evaluator: {e}")
            return {"key": "context_quality", "score": 0, "comment": f"Evaluation error: {e}"}
    
    def response_appropriateness_evaluator(self, run, example) -> Dict[str, Any]:
        """Evaluate if response appropriately handles confidence level"""
        try:
            bob_response = run.outputs.get("analysis", "")
            overall_confidence = run.outputs.get("overall_confidence", 0.5)
            expected_range = example.outputs.get("expected_confidence_range", [0.0, 1.0])
            
            # Check if low confidence responses appropriately ask for clarification
            has_uncertainty_acknowledgment = any(phrase in bob_response.lower() for phrase in [
                "not entirely certain", "might be", "if you could clarify", 
                "more context", "not sure", "uncertain", "clarification"
            ])
            
            # Check if high confidence responses are assertive
            has_confident_language = any(phrase in bob_response.lower() for phrase in [
                "exactly", "definitely", "clearly", "precisely", "certainly"
            ])
            
            appropriateness_score = 0.5  # Base score
            
            # Low confidence should acknowledge uncertainty
            if overall_confidence < 0.6 and has_uncertainty_acknowledgment:
                appropriateness_score += 0.4
            elif overall_confidence < 0.6 and not has_uncertainty_acknowledgment:
                appropriateness_score -= 0.2
            
            # High confidence should be more assertive
            if overall_confidence > 0.8 and has_confident_language:
                appropriateness_score += 0.3
            elif overall_confidence > 0.8 and has_uncertainty_acknowledgment:
                appropriateness_score -= 0.1  # Shouldn't be uncertain if confident
            
            appropriateness_score = max(0.0, min(1.0, appropriateness_score))
            
            return {
                "key": "response_appropriateness",
                "score": appropriateness_score,
                "comment": f"Confidence: {overall_confidence:.2f}, Has uncertainty: {has_uncertainty_acknowledgment}, Has confidence: {has_confident_language}"
            }
            
        except Exception as e:
            logger.error(f"Error in response appropriateness evaluator: {e}")
            return {"key": "response_appropriateness", "score": 0, "comment": f"Evaluation error: {e}"}
    
    def enhanced_accuracy_evaluator(self, run, example) -> Dict[str, Any]:
        """Enhanced accuracy evaluation with confidence weighting"""
        try:
            agent_output = run.outputs.get("analysis", "")
            expected_keywords = example.outputs.get("expected_keywords", [])
            overall_confidence = run.outputs.get("overall_confidence", 0.5)
            
            # Keyword matching score
            keywords_found = sum(1 for keyword in expected_keywords 
                                if keyword.lower() in agent_output.lower())
            keyword_score = keywords_found / len(expected_keywords) if expected_keywords else 0
            
            # Query type accuracy
            expected_query_type = example.outputs.get("expected_query_type")
            actual_query_type = run.outputs.get("query_type")
            query_type_correct = expected_query_type == actual_query_type
            
            # Base accuracy
            base_accuracy = (keyword_score + (1 if query_type_correct else 0)) / 2
            
            # Confidence-weighted accuracy: penalize overconfidence on wrong answers
            if base_accuracy < 0.5 and overall_confidence > 0.7:
                # Overconfident and wrong - penalty
                final_accuracy = base_accuracy * 0.5
                confidence_note = "Penalized for overconfidence on incorrect answer"
            elif base_accuracy > 0.8 and overall_confidence > 0.7:
                # Confident and correct - bonus
                final_accuracy = min(1.0, base_accuracy * 1.1)
                confidence_note = "Bonus for appropriate confidence on correct answer"
            else:
                final_accuracy = base_accuracy
                confidence_note = "Standard scoring"
            
            return {
                "key": "enhanced_accuracy",
                "score": final_accuracy,
                "comment": f"Keywords: {keywords_found}/{len(expected_keywords)}, Query type: {query_type_correct}, {confidence_note}"
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced accuracy evaluator: {e}")
            return {"key": "enhanced_accuracy", "score": 0, "comment": f"Evaluation error: {e}"}
    
    def bob_ross_authenticity_evaluator(self, run, example) -> Dict[str, Any]:
        """Evaluate Bob Ross style authenticity with confidence awareness"""
        try:
            response = run.outputs.get("analysis", "")
            overall_confidence = run.outputs.get("overall_confidence", 0.5)
            
            # Bob Ross style indicators
            bob_indicators = [
                "friend", "happy little", "beautiful", "wonderful", "gentle",
                "paint", "canvas", "brush", "color", "masterpiece", 
                "accident", "mistake", "believe", "confidence"
            ]
            
            indicators_found = sum(1 for indicator in bob_indicators 
                                 if indicator.lower() in response.lower())
            
            # Base style score
            style_score = min(1.0, indicators_found / 5)
            
            # Check for appropriate confidence messaging
            if overall_confidence < 0.6:
                # Low confidence should have gentle uncertainty
                gentle_uncertainty = any(phrase in response.lower() for phrase in [
                    "i think", "might be", "not entirely sure", "let me know if"
                ])
                if gentle_uncertainty:
                    style_score = min(1.0, style_score + 0.2)
            
            return {
                "key": "bob_ross_authenticity", 
                "score": style_score,
                "comment": f"Bob Ross indicators: {indicators_found}, Style score: {style_score:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Error in authenticity evaluator: {e}")
            return {"key": "bob_ross_authenticity", "score": 0, "comment": f"Evaluation error: {e}"}
    
    def agent_factory(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced factory function for evaluation"""
        try:
            text = inputs.get("text", "")
            result = self.agent.process_highlighted_text(text)
            return result
        except Exception as e:
            logger.error(f"Error in enhanced agent factory: {e}")
            return {
                "analysis": f"Error processing text: {e}",
                "error": str(e),
                "overall_confidence": 0.1
            }
    
    def run_enhanced_evaluation(self, dataset_name: str) -> str:
        """Run the enhanced evaluation experiment"""
        try:
            logger.info(f"Starting enhanced evaluation on dataset: {dataset_name}")
            
            # Enhanced evaluator suite
            evaluators = [
                self.confidence_calibration_evaluator,
                self.context_quality_evaluator,
                self.response_appropriateness_evaluator,
                self.enhanced_accuracy_evaluator,
                self.bob_ross_authenticity_evaluator
            ]
            
            # Run evaluation with enhanced metrics
            experiment_results = evaluate(
                self.agent_factory,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix="bob-ross-confidence-enhanced",
                description="Enhanced evaluation with confidence scoring and context categorization"
            )
            
            experiment_name = experiment_results.experiment_name
            logger.info(f"Enhanced evaluation completed! Experiment: {experiment_name}")
            
            return experiment_name
            
        except Exception as e:
            logger.error(f"Error running enhanced evaluation: {e}")
            raise
    
    def analyze_enhanced_results(self, experiment_name: str) -> Dict[str, Any]:
        """Analyze enhanced evaluation results with confidence insights"""
        try:
            # Get experiment results
            runs = list(self.client.list_runs(project_name=experiment_name))
            
            # Enhanced metrics collection
            metrics = {
                "total_runs": len(runs),
                "confidence_calibration": [],
                "context_quality": [],
                "response_appropriateness": [],
                "enhanced_accuracy": [],
                "bob_ross_authenticity": [],
                "confidence_distribution": {"low": 0, "medium": 0, "high": 0}
            }
            
            for run in runs:
                if hasattr(run, 'feedback_stats') and run.feedback_stats:
                    feedback = run.feedback_stats
                    
                    # Collect all metrics
                    for metric_name in ["confidence_calibration", "context_quality", 
                                       "response_appropriateness", "enhanced_accuracy", "bob_ross_authenticity"]:
                        if metric_name in feedback:
                            metrics[metric_name].append(feedback[metric_name].get('avg', 0))
                
                # Analyze confidence distribution
                if hasattr(run, 'outputs') and run.outputs:
                    confidence = run.outputs.get('overall_confidence', 0.5)
                    if confidence < 0.4:
                        metrics["confidence_distribution"]["low"] += 1
                    elif confidence < 0.7:
                        metrics["confidence_distribution"]["medium"] += 1
                    else:
                        metrics["confidence_distribution"]["high"] += 1
            
            # Calculate enhanced results
            def safe_avg(scores): 
                return sum(scores) / len(scores) if scores else 0
            
            results = {
                "experiment_name": experiment_name,
                "total_test_cases": metrics["total_runs"],
                "confidence_calibration": safe_avg(metrics["confidence_calibration"]),
                "context_quality": safe_avg(metrics["context_quality"]),
                "response_appropriateness": safe_avg(metrics["response_appropriateness"]),
                "enhanced_accuracy": safe_avg(metrics["enhanced_accuracy"]),
                "bob_ross_authenticity": safe_avg(metrics["bob_ross_authenticity"]),
                "confidence_distribution": metrics["confidence_distribution"],
                "overall_score": safe_avg([
                    safe_avg(metrics["confidence_calibration"]),
                    safe_avg(metrics["context_quality"]),
                    safe_avg(metrics["enhanced_accuracy"]),
                    safe_avg(metrics["bob_ross_authenticity"])
                ])
            }
            
            logger.info("🎨 Enhanced Evaluation Results:")
            logger.info(f"📊 Confidence Calibration: {results['confidence_calibration']:.2f}")
            logger.info(f"📚 Context Quality: {results['context_quality']:.2f}")
            logger.info(f"🎯 Response Appropriateness: {results['response_appropriateness']:.2f}")
            logger.info(f"✅ Enhanced Accuracy: {results['enhanced_accuracy']:.2f}")
            logger.info(f"🎨 Bob Ross Authenticity: {results['bob_ross_authenticity']:.2f}")
            logger.info(f"📈 Overall Score: {results['overall_score']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing enhanced results: {e}")
            return {"error": str(e)}

def main():
    """Enhanced evaluation main function"""
    try:
        evaluator = EnhancedBobRossEvaluator()
        
        # Create enhanced dataset
        logger.info("Creating enhanced evaluation dataset with confidence expectations...")
        dataset_name = evaluator.create_enhanced_evaluation_dataset()
        
        # Run enhanced evaluation
        logger.info("Running enhanced evaluation with confidence metrics...")
        experiment_name = evaluator.run_enhanced_evaluation(dataset_name)
        
        # Analyze enhanced results
        logger.info("Analyzing enhanced results...")
        results = evaluator.analyze_enhanced_results(experiment_name)
        
        # Save enhanced results
        with open("enhanced_evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎨 Enhanced Bob Ross Agent Evaluation Complete!")
        print(f"🔬 Experiment: {experiment_name}")
        print(f"📊 Confidence Calibration: {results.get('confidence_calibration', 0):.2f}")
        print(f"📚 Context Quality: {results.get('context_quality', 0):.2f}")
        print(f"🎯 Overall Score: {results.get('overall_score', 0):.2f}")
        print(f"📁 Results saved to: enhanced_evaluation_results.json")
        print(f"🌐 View in LangSmith: https://smith.langchain.com")
        
    except Exception as e:
        logger.error(f"Enhanced evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()