import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional
from langsmith import Client, traceable
from langsmith.evaluation import evaluate
from langsmith.schemas import Dataset, Example
from langchain_anthropic import ChatAnthropic
from langgraph_agent import get_agent  # Your enhanced agent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedBobRossEvaluator:
    """Enhanced evaluation with confidence-based metrics"""
    
    def __init__(self):
        self.client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        self.agent = get_agent()  # Your enhanced agent
        self.llm = ChatAnthropic(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name="claude-sonnet-4-20250514"
        )
    
    def load_test_cases_from_csv(self, csv_file_path: str, dataset_name: str):
        """Load test cases from CSV file using LangSmith upload_csv method"""
        confidence_ranges = {
            "low": [0.0, 0.4],
            "medium": [0.4, 0.7], 
            "high": [0.7, 1.0]
        }
        
        try:
            # Check if dataset already exists
            try:
                existing_dataset = self.client.read_dataset(dataset_name=dataset_name)
                logger.info(f"Dataset {dataset_name} already exists. Skipping CSV upload to avoid duplicates.")
                return dataset_name
            except Exception:
                logger.info(f"Dataset {dataset_name} doesn't exist. Creating new dataset from CSV.")
            
            # Use LangSmith's upload_csv method
            dataset = self.client.upload_csv(
                csv_file=csv_file_path,
                input_keys=['input'],  # Column that contains the input text
                output_keys=['expected_query_type', 'expected_confidence', 'confidence_rationale', 
                           'category', 'severity', 'difficulty'],  # Expected output columns
                name=dataset_name,
                description="Support ticket evaluation dataset uploaded from CSV"
            )
            
            logger.info(f"Successfully uploaded CSV to dataset: {dataset_name}")
            logger.info(f"Dataset ID: {dataset.id}")
            
            return dataset_name
            
        except Exception as e:
            logger.error(f"Failed to upload CSV {csv_file_path}: {e}")
            raise

    
    def confidence_calibration(self, inputs: dict, outputs: dict, reference_outputs: dict) -> float:
        """
        Evaluate how well-calibrated the agent's confidence scores are for understanding user intent.
        
        Tests whether the agent's overall_confidence appropriately reflects the clarity/ambiguity 
        of the user's request:
        - High confidence for clear, specific questions (e.g., explicit error messages)
        - Low confidence for vague, ambiguous questions (e.g., "this doesn't work")
        - Medium confidence for questions with some context but missing details
        
        Returns a calibration score from 0.0 to 1.0:
        - 1.0 = perfectly calibrated (confidence matches expected range)
        - 0.0 = completely miscalibrated (confidence far from expected range)
        """
        try:
            # Get agent outputs
            overall_confidence = outputs.get("overall_confidence", 0.0)
            
            # Get expected confidence from reference
            expected_confidence = reference_outputs.get("expected_confidence", "medium")
            confidence_ranges = {
                "low": [0.0, 0.4],
                "medium": [0.4, 0.7], 
                "high": [0.7, 1.0]
            }
            
            # Convert label to range
            expected_range = confidence_ranges.get(expected_confidence, [0.0, 1.0])
            min_expected, max_expected = expected_range
            
            # Calculate calibration score
            if min_expected <= overall_confidence <= max_expected:
                # Perfect calibration
                calibration_score = 1.0
            else:
                # Penalize based on distance from expected range
                if overall_confidence < min_expected:
                    distance = min_expected - overall_confidence
                else:  # overall_confidence > max_expected
                    distance = overall_confidence - max_expected
                
                # Convert distance to penalty (0.0 to 1.0 scale)
                # Maximum penalty when distance >= 0.5 (e.g., expecting low 0.2 but got high 0.9)
                calibration_score = max(0.0, 1.0 - (distance * 2.0))
            
            return calibration_score
            
        except Exception as e:
            logger.error(f"Error in confidence calibration evaluator: {e}")
            return 0.0
    
    def query_classification(self, inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        """
        Evaluate accuracy of query type classification.
        
        Tests whether the agent correctly identified the type of help the user needs
        (e.g., error_help, api_usage, code_explanation, etc.).
        
        Returns True if classification is correct, False otherwise.
        """
        try:
            expected_query_type = reference_outputs.get("expected_query_type")
            actual_query_type = outputs.get("query_type")
            
            # Direct classification accuracy
            classification_correct = expected_query_type == actual_query_type
            
            if not classification_correct:
                logger.debug(f"Classification mismatch - Expected: {expected_query_type}, Actual: {actual_query_type}")
            
            return classification_correct
            
        except Exception as e:
            logger.error(f"Error in query classification evaluator: {e}")
            return False
    
    
    
    @traceable
    def agent_factory(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced factory function for evaluation with @traceable decorator"""
        try:
            # CSV upload uses column name as key, try both "text" and "input"
            text = inputs.get("text", "") or inputs.get("input", "")
            if not text:
                logger.warning(f"No text found in inputs. Available keys: {list(inputs.keys())}")
            
            result = self.agent.process_highlighted_text(text)
            
            # Return in the expected format for evaluators
            return {
                "query_type": result.get("query_type", "unknown"),
                "overall_confidence": result.get("overall_confidence", 0.0),
                "analysis": result.get("bob_ross_response", ""),
                # Include other fields that evaluators might need
                "classification_confidence": result.get("classification_confidence", 0.0),
                "context_confidence": result.get("context_confidence", 0.0)
            }
        except Exception as e:
            logger.error(f"Error in enhanced agent factory: {e}")
            return {
                "query_type": "error",
                "analysis": f"Error processing text: {e}",
                "error": str(e),
                "overall_confidence": 0.1
            }
    
    def run_enhanced_evaluation(self, dataset_name: str) -> str:
        """Run the enhanced evaluation experiment"""
        try:
            logger.info("="*60)
            logger.info(f"🚀 STARTING EVALUATION")
            logger.info(f"📊 Dataset: {dataset_name}")
            logger.info("="*60)
            
            # Get dataset info for logging
            try:
                dataset = self.client.read_dataset(dataset_name=dataset_name)
                example_count = len(list(self.client.list_examples(dataset_id=dataset.id)))
                logger.info(f"📝 Processing {example_count} test cases")
            except Exception as e:
                logger.warning(f"Could not count examples: {e}")
            
            # Simplified evaluator suite - only classification accuracy and confidence calibration
            evaluators = [
                self.query_classification,
                self.confidence_calibration
            ]
            logger.info(f"🔧 Running 2 evaluators:")
            logger.info(f"   1️⃣ Query Classification Accuracy")
            logger.info(f"   2️⃣ Confidence Calibration")
            
            logger.info("⏳ Running evaluation... (this may take a few minutes)")
            
            # Run evaluation using LangSmith Client method (as per docs)
            experiment_results = self.client.evaluate(
                self.agent_factory,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix="bob-ross-simplified",
                description="Simplified evaluation with query classification and confidence calibration",
                max_concurrency=1,  # Process one at a time for debugging
                blocking=True  # Wait for completion before returning
            )
            
            experiment_name = experiment_results.experiment_name
            logger.info(f"🔬 Experiment results object: {type(experiment_results)}")
            logger.info(f"🔬 Experiment results attributes: {dir(experiment_results)}")
            
            # Try to get results directly from experiment_results if available
            if hasattr(experiment_results, '_results'):
                logger.info(f"🔬 Direct results available: {len(experiment_results._results) if experiment_results._results else 0}")
            if hasattr(experiment_results, '_summary_results'):
                logger.info(f"🔬 Summary results available: {experiment_results._summary_results}")
            if hasattr(experiment_results, 'to_pandas'):
                logger.info("🔬 Pandas conversion available")
            
            logger.info("="*60)
            logger.info(f"✅ EVALUATION COMPLETE!")
            logger.info(f"🔬 Experiment Name: {experiment_name}")
            logger.info("="*60)
            
            return experiment_name, experiment_results  # Return both for direct analysis
            
        except Exception as e:
            logger.error("="*60)
            logger.error(f"❌ EVALUATION FAILED: {e}")
            logger.error("="*60)
            raise
    
    def analyze_enhanced_results(self, experiment_name: str, experiment_results=None) -> Dict[str, Any]:
        """Analyze enhanced evaluation results with confidence insights"""
        try:
            logger.info("🔍 ANALYZING RESULTS...")
            logger.info(f"📈 Fetching experiment data: {experiment_name}")
            
            # Try to use experiment_results directly first
            metrics = {
                "total_runs": 0,
                "query_classification": [],
                "confidence_calibration": [],
                "confidence_distribution": {"low": 0, "medium": 0, "high": 0}
            }
            
            if experiment_results and hasattr(experiment_results, 'to_pandas'):
                logger.info("🔬 Using direct experiment results via pandas...")
                try:
                    # Wait for results to complete if needed
                    if hasattr(experiment_results, 'wait'):
                        logger.info("⏳ Waiting for experiment to complete...")
                        experiment_results.wait()
                    
                    # Convert to pandas DataFrame
                    df = experiment_results.to_pandas()
                    logger.info(f"📊 Found {len(df)} evaluation runs in DataFrame")
                    
                    if len(df) > 0:
                        logger.info(f"🔍 DataFrame columns: {list(df.columns)}")
                        
                        # Extract metrics from DataFrame
                        metrics["total_runs"] = len(df)
                        
                        # Look for evaluator results in the DataFrame
                        for col in df.columns:
                            if 'query_classification' in col.lower():
                                # Boolean evaluator - convert True/False to 1.0/0.0 scores
                                bool_scores = df[col].dropna().tolist()
                                scores = [1.0 if score else 0.0 for score in bool_scores]
                                metrics["query_classification"].extend(scores)
                                logger.info(f"🔍 Found query_classification results: {bool_scores} -> scores: {scores}")
                            elif 'confidence_calibration' in col.lower():
                                # Float evaluator - use scores directly
                                float_scores = df[col].dropna().tolist()
                                metrics["confidence_calibration"].extend(float_scores)
                                logger.info(f"🔍 Found confidence_calibration scores: {float_scores}")
                        
                        # Analyze confidence distribution from outputs
                        if 'outputs.overall_confidence' in df.columns:
                            for confidence in df['outputs.overall_confidence'].dropna():
                                if confidence < 0.4:
                                    metrics["confidence_distribution"]["low"] += 1
                                elif confidence < 0.7:
                                    metrics["confidence_distribution"]["medium"] += 1
                                else:
                                    metrics["confidence_distribution"]["high"] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to use pandas approach: {e}")
                    logger.info("🔄 Falling back to direct run query...")
            
            # Fallback: Try to query runs directly if pandas approach failed
            if metrics["total_runs"] == 0:
                logger.info("🔄 Using fallback method to get runs...")
                import time
                time.sleep(3)  # Brief delay
                
                try:
                    runs = list(self.client.list_runs(project_name=experiment_name))
                    logger.info(f"📊 Found {len(runs)} evaluation runs via direct query")
                    
                    if len(runs) > 0:
                        metrics["total_runs"] = len(runs)
                        # Process runs for metrics (simplified version)
                        for run in runs:
                            if hasattr(run, 'feedback_stats') and run.feedback_stats:
                                feedback = run.feedback_stats
                                for metric_name in ["query_classification", "confidence_calibration"]:
                                    if metric_name in feedback:
                                        score = feedback[metric_name].get('avg', 0)
                                        metrics[metric_name].append(score)
                                        
                except Exception as e:
                    logger.warning(f"Fallback method also failed: {e}")
            
            # Calculate simplified results
            def safe_avg(scores): 
                return sum(scores) / len(scores) if scores else 0
            
            results = {
                "experiment_name": experiment_name,
                "total_test_cases": metrics["total_runs"],
                "query_classification": safe_avg(metrics["query_classification"]),
                "confidence_calibration": safe_avg(metrics["confidence_calibration"]),
                "confidence_distribution": metrics["confidence_distribution"],
                "overall_score": safe_avg([
                    safe_avg(metrics["query_classification"]),
                    safe_avg(metrics["confidence_calibration"])
                ])
            }
            
            logger.info("="*60)
            logger.info("🎨 EVALUATION RESULTS:")
            logger.info("="*60)
            logger.info(f"🎯 Query Classification: {results['query_classification']:.3f} ({results['query_classification']*100:.1f}%)")
            logger.info(f"📊 Confidence Calibration: {results['confidence_calibration']:.3f} ({results['confidence_calibration']*100:.1f}%)")
            logger.info(f"📈 Overall Score: {results['overall_score']:.3f} ({results['overall_score']*100:.1f}%)")
            logger.info(f"📋 Test Cases Processed: {results['total_test_cases']}")
            logger.info("="*60)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing enhanced results: {e}")
            return {"error": str(e)}

def load_all_sample_datasets(evaluator, sample_dir="sample_datasets") -> str:
    """Load CSV files from sample_datasets directory into single dataset"""
    dataset_name = "bob-ross-support-tickets"
    
    # Find all CSV files in the sample_datasets directory
    csv_files = glob.glob(os.path.join(os.getcwd(), sample_dir, "*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {sample_dir} directory")
        return None
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file (though typically you'll have one main file)
    for csv_file in csv_files:
        try:
            logger.info(f"Processing {csv_file}...")
            evaluator.load_test_cases_from_csv(csv_file, dataset_name)
            
        except Exception as e:
            logger.error(f"Failed to load {csv_file}: {e}")
    
    return dataset_name

def main():
    """Simplified evaluation main function"""
    try:
        print("="*80)
        print("🎨 BOB ROSS SUPPORT TICKETS EVALUATION")
        print("="*80)
        
        evaluator = SimplifiedBobRossEvaluator()
        
        # Load all sample datasets into single dataset
        logger.info("📂 STEP 1: Loading CSV datasets...")
        dataset_name = load_all_sample_datasets(evaluator)
        
        if not dataset_name:
            logger.error("❌ No datasets loaded. Please add CSV files to the sample_datasets directory.")
            return
        
        logger.info(f"✅ Dataset loaded: {dataset_name}")
        
        # Run evaluation on the consolidated dataset
        logger.info("🔄 STEP 2: Running evaluation...")
        
        try:
            # Run evaluation
            experiment_name, experiment_results = evaluator.run_enhanced_evaluation(dataset_name)
            
            # Analyze results
            logger.info("📊 STEP 3: Analyzing results...")
            results = evaluator.analyze_enhanced_results(experiment_name, experiment_results)
            
            # Save results in timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = ".evaluation_results"
            os.makedirs(results_dir, exist_ok=True)
            
            results_filename = os.path.join(results_dir, f"{timestamp}.json")
            with open(results_filename, "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"💾 Results saved to: {results_filename}")
            
            print("\n" + "="*80)
            print("🎉 EVALUATION COMPLETE!")
            print("="*80)
            print(f"🔬 Experiment: {experiment_name}")
            print(f"📊 Dataset: {dataset_name}")
            print(f"🎯 Query Classification: {results.get('query_classification', 0)*100:.1f}%")
            print(f"📊 Confidence Calibration: {results.get('confidence_calibration', 0)*100:.1f}%")
            print(f"📈 Overall Score: {results.get('overall_score', 0)*100:.1f}%")
            print(f"📁 Results: {results_filename}")
            print(f"🌐 LangSmith: https://smith.langchain.com")
            print("="*80)
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate dataset {dataset_name}: {e}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation setup failed: {e}")
        raise

if __name__ == "__main__":
    main()