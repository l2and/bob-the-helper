#!/usr/bin/env python3
"""
Debug script to investigate LangSmith 500 error
"""

import os
from langsmith import Client
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_experiment(experiment_name="bob-ross-simplified-cbb3bf79"):
    """Debug the experiment that's causing 500 errors"""
    try:
        client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        
        print(f"[DEBUG] Debugging experiment: {experiment_name}")
        print("=" * 60)
        
        # Try to get runs from the experiment
        runs = list(client.list_runs(project_name=experiment_name))
        print(f"[INFO] Found {len(runs)} runs")
        
        for i, run in enumerate(runs):
            print(f"\n[RUN] Run {i+1}:")
            print(f"   ID: {run.id}")
            print(f"   Status: {getattr(run, 'status', 'unknown')}")
            print(f"   Error: {getattr(run, 'error', 'None')}")
            
            # Check inputs
            if hasattr(run, 'inputs') and run.inputs:
                inputs_size = len(str(run.inputs))
                print(f"   Inputs size: {inputs_size} chars")
                if inputs_size > 10000:
                    print(f"   [WARNING] Large inputs detected!")
                    
            # Check outputs
            if hasattr(run, 'outputs') and run.outputs:
                outputs_size = len(str(run.outputs))
                print(f"   Outputs size: {outputs_size} chars")
                if outputs_size > 10000:
                    print(f"   [WARNING] Large outputs detected!")
                    
                # Check for problematic data types
                try:
                    json.dumps(run.outputs)
                    print(f"   [OK] Outputs are JSON serializable")
                except Exception as e:
                    print(f"   [ERROR] Outputs NOT JSON serializable: {e}")
            
            # Check feedback
            if hasattr(run, 'feedback_stats') and run.feedback_stats:
                print(f"   Feedback: {len(run.feedback_stats)} items")
                for key, value in run.feedback_stats.items():
                    print(f"     {key}: {value}")
        
        # Try to access the experiment directly
        print(f"\n[CHECK] Trying to access experiment metadata...")
        try:
            # This might fail if the experiment is corrupted
            project = client.read_project(project_name=experiment_name)
            print(f"   [OK] Project accessible: {project.name}")
        except Exception as e:
            print(f"   [ERROR] Project access failed: {e}")
        
        print("\n" + "=" * 60)
        print("[RECOMMENDATIONS]")
        
        # Check for common issues
        has_large_data = any(
            (hasattr(run, 'inputs') and len(str(run.inputs)) > 10000) or
            (hasattr(run, 'outputs') and len(str(run.outputs)) > 10000)
            for run in runs
        )
        
        if has_large_data:
            print("[WARNING] Large data detected - this might cause UI issues")
            print("   Consider truncating large outputs in the agent_factory")
        
        has_serialization_issues = False
        for run in runs:
            if hasattr(run, 'outputs') and run.outputs:
                try:
                    json.dumps(run.outputs)
                except:
                    has_serialization_issues = True
                    break
        
        if has_serialization_issues:
            print("[WARNING] JSON serialization issues detected")
            print("   Check for circular references or non-serializable objects")
        
        if not has_large_data and not has_serialization_issues:
            print("[OK] No obvious data issues found")
            print("   This might be a temporary LangSmith backend issue")
            print("   Try refreshing the page or waiting a few minutes")
            
    except Exception as e:
        print(f"[ERROR] Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use the experiment name from your logs
    debug_experiment("bob-ross-simplified-cbb3bf79")