import sys
import os
import json
import time

# Add project root to sys.path
sys.path.append(os.getcwd())

from agent.core.planner import Planner
from agent.core.validator import PlanValidator

def run_regression():
    print("Starting Planner Regression Test...")
    
    planner = Planner()
    validator = PlanValidator()
    
    tools = [
        {"name": "compute", "description": "Perform math calculations. Inputs: expression (str) or a (num), b (num)."},
        {"name": "search", "description": "Search the web. Inputs: query (str)."},
        {"name": "reason", "description": "Reasoning engine. Inputs: context (str)."}
    ]
    
    test_cases = [
        {
            "id": 1,
            "objective": "Calculate 50 * 3",
            "expected_literals": ["50", "3"],
            "description": "Numeric extraction"
        },
        {
            "id": 2,
            "objective": "Search for iPhone 16 and extract price",
            "expected_literals": ["iPhone 16 price"], 
            "description": "String entity extraction"
        },
        {
            "id": 3,
            "objective": "Summarize these notes: Note A, Note B, Note C",
            "expected_literals": ["Note A, Note B, Note C"],
            "description": "List/Long text extraction"
        }
    ]
    
    total_runs = 10
    results = {}
    
    for case in test_cases:
        print(f"\n--- Testing Case {case['id']}: {case['description']} ---")
        print(f"Objective: {case['objective']}")
        
        success_count = 0
        
        for i in range(total_runs):
            print(f"Run {i+1}/{total_runs}...", end="", flush=True)
            try:
                # We need to simulate the tools structure as expected by planner
                plan = planner.plan(case['objective'], tools)
                
                # 1. Validate structure
                val_res = validator.validate(plan)
                if not val_res.valid:
                    print(f" Invalid Plan: {val_res.errors}")
                    continue
                
                # 2. Check literals
                found_literals = []
                for step in plan.steps:
                    for binding in step.operation.inputs.values():
                        if binding.value_ref.source == "literal":
                            found_literals.append(binding.value_ref.name)
                
                # Check if all expected literals are present (partial match or exact match?)
                # The prompt asks for "name directly holds primitive value".
                # For "Calculate 50 * 3", we expect "50" and "3".
                # For "iPhone 16 price", we expect "iPhone 16 price" (based on few-shot).
                
                all_found = True
                for expected in case['expected_literals']:
                    # We look for the expected string in the found literals
                    # Strict check: expected must be in found_literals list
                    if expected not in found_literals:
                        # Fallback: maybe it combined them? e.g. "50 * 3"
                        # Or maybe "iPhone 16" and "price" separate?
                        # Let's be slightly flexible: is expected substring of any found literal?
                        match = False
                        for found in found_literals:
                            if expected in found:
                                match = True
                                break
                        if not match:
                            all_found = False
                            break
                
                if all_found:
                    print(" Success")
                    success_count += 1
                else:
                    print(f" Failed. Found: {found_literals}, Expected: {case['expected_literals']}")
                    
            except Exception as e:
                print(f" Error: {e}")
        
        accuracy = (success_count / total_runs) * 100
        results[case['id']] = accuracy
        print(f"Case {case['id']} Accuracy: {accuracy}%")

    print("\n=== Final Report ===")
    overall_sum = 0
    for case_id, acc in results.items():
        print(f"Case {case_id}: {acc}%")
        overall_sum += acc
    
    avg_acc = overall_sum / len(test_cases)
    print(f"Average Accuracy: {avg_acc}%")
    
    if avg_acc >= 95:
        print("RESULT: PASS (>= 95%)")
    else:
        print("RESULT: FAIL (< 95%)")

if __name__ == "__main__":
    run_regression()
