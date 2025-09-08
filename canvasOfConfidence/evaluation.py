import os
import sys
import json
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional

from langsmith import Client, traceable
from langsmith.evaluation import evaluate

# Add happyLittleTreesOfKnowledge directory to path to import langgraph_agent and logger_config
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'happyLittleTreesOfKnowledge'))
from langgraph_agent import get_agent  # Your enhanced agent
from logger_config import setup_logger

from evaluators import confidence_calibration, query_classification

logger = setup_logger(__name__)

class SimplifiedBobRossEvaluator:
    """Enhanced evaluation with confidence-based metrics"""
    
    def __init__(self):
        self.client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        self.agent = get_agent()  # Your enhanced agent
    
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

    
    
    
    
    @traceable(name="agent_factory")
    def agent_factory(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced factory function for evaluation with @traceable decorator"""
        try:
            # CSV upload uses column name as key, try both "text" and "input"
            text = inputs.get("text", "") or inputs.get("input", "")
            if not text:
                logger.warning(f"No text found in inputs. Available keys: {list(inputs.keys())}")
                return {
                    "query_type": "error",
                    "analysis": "No input text provided",
                    "overall_confidence": 0.0,
                    "classification_confidence": 0.0,
                    "context_confidence": 0.0
                }
            
            logger.info(f"Processing text: {text[:100]}...")  # Log first 100 chars for debugging
            
            # Use the process_highlighted_text method which is the correct interface
            result = self.agent.process_highlighted_text(text)
            
            # The result is already the final outputs dictionary
            final_outputs = result if isinstance(result, dict) else {}
            
            # Sanitize and truncate outputs to prevent UI issues
            def sanitize_output(text, max_length=2000):
                if not isinstance(text, str):
                    return str(text)
                # Remove problematic characters that might cause UI issues
                text = text.replace('\x00', '').replace('\x01', '').replace('\x02', '')
                # Truncate if too long
                if len(text) > max_length:
                    text = text[:max_length] + "... [truncated]"
                return text
            
            # Return in the expected format for evaluators with sanitized outputs
            return {
                "query_type": final_outputs.get("query_type", "unknown"),
                "overall_confidence": final_outputs.get("overall_confidence", 0.0),
                "analysis": sanitize_output(final_outputs.get("bob_ross_response", "")),
                # Include other fields that evaluators might need
                "classification_confidence": final_outputs.get("classification_confidence", 0.0),
                "context_confidence": final_outputs.get("context_confidence", 0.0)
            }
        except Exception as e:
            logger.error(f"Error in enhanced agent factory: {e}")
            import traceback
            traceback.print_exc()
            return {
                "query_type": "error",
                "analysis": f"Error processing text: {e}",
                "error": str(e),
                "overall_confidence": 0.1,
                "classification_confidence": 0.0,
                "context_confidence": 0.0
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
                query_classification,
                confidence_calibration
            ]
            logger.info(f"🔧 Running 2 evaluators:")
            logger.info(f"   1️⃣ Query Classification Accuracy")
            logger.info(f"   2️⃣ Confidence Calibration")
            
            logger.info("⏳ Running evaluation... (this may take a few minutes)")
            
            # Use the evaluate function from langsmith.evaluation module
            from langsmith.evaluation import evaluate
            
            # The evaluate function will automatically create an experiment and link all runs
            experiment_results = evaluate(
                self.agent_factory,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix="bob-ross-simplified",
                description="Simplified evaluation with query classification and confidence calibration",
                max_concurrency=1,  # Process one at a time for debugging
                client=self.client,
                metadata={"evaluation_type": "bob_ross_confidence_calibration"}  # Add metadata for tracking
            )
            
            experiment_name = experiment_results.experiment_name
            
            # Debug: Check all attributes of experiment_results
            logger.info(f"🔬 Experiment attributes: {[attr for attr in dir(experiment_results) if not attr.startswith('_')]}")
            
            # Try to get experiment ID from different possible locations
            experiment_id = None
            if hasattr(experiment_results, 'experiment_id'):
                experiment_id = experiment_results.experiment_id
            elif hasattr(experiment_results, 'id'):
                experiment_id = experiment_results.id
            
            # Try to get it from the results manager
            if not experiment_id and hasattr(experiment_results, '_manager'):
                manager = experiment_results._manager
                if hasattr(manager, 'experiment_id'):
                    experiment_id = manager.experiment_id
                elif hasattr(manager, 'experiment'):
                    experiment_id = getattr(manager.experiment, 'id', None)
            
            logger.info("="*60)
            logger.info(f"✅ EVALUATION COMPLETE!")
            logger.info(f"🔬 Experiment Name: {experiment_name}")
            logger.info(f"🔬 Experiment ID: {experiment_id}")
            
            # Try to get direct experiment URL
            if experiment_id:
                experiment_url = f"https://smith.langchain.com/experiments/{experiment_id}"
            else:
                experiment_url = f"https://smith.langchain.com/experiments?name={experiment_name}"
            
            logger.info(f"🔗 Direct Experiment URL: {experiment_url}")
            logger.info("="*60)
            
            # Log detailed experiment info for debugging
            logger.info(f"🔬 Experiment results object: {type(experiment_results)}")
            
            # Try to get results directly from experiment_results if available
            if hasattr(experiment_results, '_results'):
                results_count = len(experiment_results._results) if experiment_results._results else 0
                logger.info(f"🔬 Direct results available: {results_count}")
                
                # Debug: Log details of first result if available
                if results_count > 0:
                    first_result = experiment_results._results[0]
                    logger.info(f"🔬 First result keys: {list(first_result.keys()) if hasattr(first_result, 'keys') else 'Not a dict'}")
                    if hasattr(first_result, 'get'):
                        logger.info(f"🔬 First result run_id: {first_result.get('id', 'No ID')}")
                        logger.info(f"🔬 First result example_id: {first_result.get('reference_example_id', 'No example ID')}")
            
            if hasattr(experiment_results, '_summary_results'):
                logger.info(f"🔬 Summary results available: {experiment_results._summary_results}")
            if hasattr(experiment_results, 'to_pandas'):
                logger.info("🔬 Pandas conversion available")
                
            # Additional debugging: Try to query runs directly by experiment name
            try:
                logger.info("🔍 Attempting to query runs by experiment name...")
                runs_by_name = list(self.client.list_runs(project_name=experiment_name, limit=5))
                logger.info(f"🔬 Found {len(runs_by_name)} runs by experiment name")
                if runs_by_name:
                    first_run = runs_by_name[0]
                    logger.info(f"🔬 First run ID: {first_run.id}")
                    logger.info(f"🔬 First run reference_example_id: {getattr(first_run, 'reference_example_id', 'None')}")
            except Exception as e:
                logger.warning(f"🔬 Could not query runs by experiment name: {e}")
            
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
                "experiment_id": getattr(experiment_results, 'experiment_id', None) if experiment_results else None,
                "langsmith_url": f"https://smith.langchain.com/experiments/{experiment_name}",
                "dataset_name": "bob-ross-support-tickets",
                "total_test_cases": metrics["total_runs"],
                "query_classification": safe_avg(metrics["query_classification"]),
                "confidence_calibration": safe_avg(metrics["confidence_calibration"]),
                "confidence_distribution": metrics["confidence_distribution"],
                "overall_score": safe_avg([
                    safe_avg(metrics["query_classification"]),
                    safe_avg(metrics["confidence_calibration"])
                ]),
                "evaluation_timestamp": datetime.now().isoformat()
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
            if results.get('experiment_id'):
                print(f"🆔 Experiment ID: {results.get('experiment_id')}")
            print(f"📊 Dataset: {dataset_name}")
            print(f"🎯 Query Classification: {results.get('query_classification', 0)*100:.1f}%")
            print(f"📊 Confidence Calibration: {results.get('confidence_calibration', 0)*100:.1f}%")
            print(f"📈 Overall Score: {results.get('overall_score', 0)*100:.1f}%")
            print(f"📁 Results: {results_filename}")
            print(f"🔗 LangSmith Experiment: {results.get('langsmith_url', 'https://smith.langchain.com')}")
            print(f"🌐 LangSmith Dashboard: https://smith.langchain.com")
            print("="*80)
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate dataset {dataset_name}: {e}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation setup failed: {e}")
        raise

if __name__ == "__main__":
    main()