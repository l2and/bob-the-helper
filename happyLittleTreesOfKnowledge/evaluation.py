import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langsmith.schemas import Dataset, Example
from langchain_anthropic import ChatAnthropic
from langgraph_agent import BobRossDocumentationAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BobRossEvaluator:
    """Evaluation suite for the Bob Ross Documentation Agent"""
    
    def __init__(self):
        self.client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        self.agent = BobRossDocumentationAgent()
        self.llm = ChatAnthropic(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name="claude-sonnet-4-20250514"
        )
    
    def create_evaluation_dataset(self) -> str:
        """Create a dataset for evaluation"""
        dataset_name = "bob-ross-documentation-assistant-eval"
        
        # Test cases covering different query types
        test_cases = [
            {
                "input": "from langchain import LLMChain",
                "expected_query_type": "code_explanation",
                "expected_content_keywords": ["LLMChain", "sequence", "prompt", "model"],
                "difficulty": "easy"
            },
            {
                "input": "ImportError: No module named 'langchain'",
                "expected_query_type": "error_help",
                "expected_content_keywords": ["install", "pip", "module", "import"],
                "difficulty": "easy"
            },
            {
                "input": "What is LCEL and how do I use the pipe operator?",
                "expected_query_type": "concept_learning",
                "expected_content_keywords": ["LCEL", "pipe", "operator", "chain", "compose"],
                "difficulty": "medium"
            },
            {
                "input": "chain = prompt | model | output_parser",
                "expected_query_type": "code_explanation", 
                "expected_content_keywords": ["LCEL", "chain", "pipe", "prompt", "model", "parser"],
                "difficulty": "medium"
            },
            {
                "input": "How do I use ChatAnthropic with streaming?",
                "expected_query_type": "api_usage",
                "expected_content_keywords": ["ChatAnthropic", "streaming", "async", "callback"],
                "difficulty": "medium"
            },
            {
                "input": "vectorstore.similarity_search() returns empty results",
                "expected_query_type": "error_help",
                "expected_content_keywords": ["vector", "similarity", "embeddings", "empty", "troubleshoot"],
                "difficulty": "hard"
            },
            {
                "input": "def create_agent_executor(tools, llm, memory=None):",
                "expected_query_type": "code_explanation",
                "expected_content_keywords": ["agent", "executor", "tools", "memory"],
                "difficulty": "hard"
            },
            {
                "input": "LangGraph state management best practices",
                "expected_query_type": "concept_learning",
                "expected_content_keywords": ["LangGraph", "state", "management", "TypedDict", "workflow"],
                "difficulty": "hard"
            }
        ]
        
        # Create examples for LangSmith
        examples = []
        for i, case in enumerate(test_cases):
            example = Example(
                inputs={"text": case["input"]},
                outputs={
                    "expected_query_type": case["expected_query_type"],
                    "expected_keywords": case["expected_content_keywords"],
                    "difficulty": case["difficulty"]
                }
            )
            examples.append(example)
        
        # Create or update dataset
        try:
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Test cases for Bob Ross Documentation Assistant evaluation"
            )
            logger.info(f"Created dataset: {dataset_name}")
        except Exception:
            # Dataset might already exist
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            logger.info(f"Using existing dataset: {dataset_name}")
        
        # Add examples
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
    
    def custom_accuracy_evaluator(self, run, example) -> Dict[str, Any]:
        """Custom evaluator for accuracy of responses"""
        try:
            # Get the agent's response
            agent_output = run.outputs.get("analysis", "")
            expected_keywords = example.outputs.get("expected_keywords", [])
            
            # Check if expected keywords are present
            keywords_found = sum(1 for keyword in expected_keywords 
                                if keyword.lower() in agent_output.lower())
            keyword_score = keywords_found / len(expected_keywords) if expected_keywords else 0
            
            # Check query type classification
            expected_query_type = example.outputs.get("expected_query_type")
            actual_query_type = run.outputs.get("query_type")
            query_type_correct = expected_query_type == actual_query_type
            
            return {
                "key": "accuracy",
                "score": (keyword_score + (1 if query_type_correct else 0)) / 2,
                "comment": f"Keywords found: {keywords_found}/{len(expected_keywords)}, "
                          f"Query type correct: {query_type_correct}"
            }
            
        except Exception as e:
            logger.error(f"Error in accuracy evaluator: {e}")
            return {"key": "accuracy", "score": 0, "comment": f"Evaluation error: {e}"}
    
    def custom_helpfulness_evaluator(self, run, example) -> Dict[str, Any]:
        """Custom evaluator using LLM-as-judge for helpfulness"""
        try:
            agent_response = run.outputs.get("analysis", "")
            user_input = run.inputs.get("text", "")
            
            helpfulness_prompt = f"""
            Evaluate how helpful this Bob Ross-style technical explanation is for a user who highlighted this text:

            User's highlighted text: "{user_input}"
            
            Bob Ross response: "{agent_response}"
            
            Rate the helpfulness on a scale of 1-10 considering:
            1. Technical accuracy
            2. Clarity of explanation  
            3. Appropriate use of Bob Ross style/metaphors
            4. Actionable guidance provided
            5. Encouraging and supportive tone
            
            Respond with just a number from 1-10.
            """
            
            response = self.llm.invoke([{"role": "user", "content": helpfulness_prompt}])
            score = float(response.content.strip()) / 10.0
            
            return {
                "key": "helpfulness",
                "score": score,
                "comment": f"LLM-judged helpfulness score: {score}"
            }
            
        except Exception as e:
            logger.error(f"Error in helpfulness evaluator: {e}")
            return {"key": "helpfulness", "score": 0.5, "comment": f"Evaluation error: {e}"}
    
    def custom_bob_ross_style_evaluator(self, run, example) -> Dict[str, Any]:
        """Evaluate how well the response captures Bob Ross's style"""
        try:
            response = run.outputs.get("analysis", "")
            
            # Check for Bob Ross characteristics
            bob_ross_indicators = [
                "friend", "happy little", "beautiful", "wonderful", "gentle",
                "paint", "canvas", "brush", "color", "masterpiece", 
                "accident", "mistake", "believe", "confidence"
            ]
            
            indicators_found = sum(1 for indicator in bob_ross_indicators 
                                 if indicator.lower() in response.lower())
            
            style_score = min(indicators_found / 5, 1.0)  # Cap at 1.0
            
            return {
                "key": "bob_ross_style",
                "score": style_score,
                "comment": f"Bob Ross style indicators found: {indicators_found}"
            }
            
        except Exception as e:
            logger.error(f"Error in style evaluator: {e}")
            return {"key": "bob_ross_style", "score": 0, "comment": f"Evaluation error: {e}"}
    
    def agent_factory(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Factory function to run the agent for evaluation"""
        try:
            text = inputs.get("text", "")
            result = self.agent.process_highlighted_text(text)
            return result
        except Exception as e:
            logger.error(f"Error in agent factory: {e}")
            return {
                "analysis": f"Error processing text: {e}",
                "error": str(e)
            }
    
    def run_evaluation(self, dataset_name: str) -> str:
        """Run the evaluation experiment"""
        try:
            logger.info(f"Starting evaluation on dataset: {dataset_name}")
            
            # Define evaluators
            evaluators = [
                self.custom_accuracy_evaluator,
                self.custom_helpfulness_evaluator,
                self.custom_bob_ross_style_evaluator
            ]
            
            # Run evaluation
            experiment_results = evaluate(
                self.agent_factory,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix="bob-ross-agent-v2",
                description="Evaluation of Bob Ross Documentation Assistant with LangGraph"
            )
            
            experiment_name = experiment_results.experiment_name
            logger.info(f"Evaluation completed! Experiment: {experiment_name}")
            
            return experiment_name
            
        except Exception as e:
            logger.error(f"Error running evaluation: {e}")
            raise
    
    def analyze_results(self, experiment_name: str) -> Dict[str, Any]:
        """Analyze evaluation results"""
        try:
            # Get experiment results
            experiment = self.client.read_project(project_name=experiment_name)
            runs = list(self.client.list_runs(project_name=experiment_name))
            
            # Calculate aggregate metrics
            metrics = {
                "total_runs": len(runs),
                "accuracy_scores": [],
                "helpfulness_scores": [],
                "bob_ross_style_scores": []
            }
            
            for run in runs:
                if hasattr(run, 'feedback_stats'):
                    feedback = run.feedback_stats or {}
                    
                    if 'accuracy' in feedback:
                        metrics["accuracy_scores"].append(feedback['accuracy'].get('avg', 0))
                    if 'helpfulness' in feedback:
                        metrics["helpfulness_scores"].append(feedback['helpfulness'].get('avg', 0))
                    if 'bob_ross_style' in feedback:
                        metrics["bob_ross_style_scores"].append(feedback['bob_ross_style'].get('avg', 0))
            
            # Calculate averages
            results = {
                "experiment_name": experiment_name,
                "total_test_cases": metrics["total_runs"],
                "avg_accuracy": sum(metrics["accuracy_scores"]) / len(metrics["accuracy_scores"]) if metrics["accuracy_scores"] else 0,
                "avg_helpfulness": sum(metrics["helpfulness_scores"]) / len(metrics["helpfulness_scores"]) if metrics["helpfulness_scores"] else 0,
                "avg_bob_ross_style": sum(metrics["bob_ross_style_scores"]) / len(metrics["bob_ross_style_scores"]) if metrics["bob_ross_style_scores"] else 0
            }
            
            logger.info(f"Evaluation Results Summary:")
            logger.info(f"- Total test cases: {results['total_test_cases']}")
            logger.info(f"- Average accuracy: {results['avg_accuracy']:.2f}")
            logger.info(f"- Average helpfulness: {results['avg_helpfulness']:.2f}")
            logger.info(f"- Average Bob Ross style: {results['avg_bob_ross_style']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing results: {e}")
            return {"error": str(e)}

def main():
    """Main evaluation function"""
    try:
        evaluator = BobRossEvaluator()
        
        # Create evaluation dataset
        logger.info("Creating evaluation dataset...")
        dataset_name = evaluator.create_evaluation_dataset()
        
        # Run evaluation
        logger.info("Running evaluation...")
        experiment_name = evaluator.run_evaluation(dataset_name)
        
        # Analyze results
        logger.info("Analyzing results...")
        results = evaluator.analyze_results(experiment_name)
        
        # Save results
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎨 Bob Ross Agent Evaluation Complete!")
        print(f"Experiment: {experiment_name}")
        print(f"Results saved to: evaluation_results.json")
        print(f"View in LangSmith: https://smith.langchain.com")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()