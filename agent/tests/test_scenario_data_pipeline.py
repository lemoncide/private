import asyncio
import os
import sys
import shutil
import pytest

sys.path.append(os.getcwd())

from agent.core.agent import LangGraphAgent
from agent.utils.config import config
from agent.utils.logger import logger

def test_data_pipeline():
    if os.environ.get("RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("integration test")

    print("Initializing Agent for Data Pipeline Scenario...")
    
    agent = LangGraphAgent()
    
    # 1. Setup Test Data
    target_dir = os.path.join(os.getcwd(), "sandbox", "data_pipeline")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    csv_file = os.path.join(target_dir, "sales.csv")
    csv_content = """product,category,amount,region
    Apple,Fruit, 100 ,North
    Banana,Fruit, 50 ,South
    Carrot,Veg, 30 ,North
    Apple,Fruit, 120 ,South
    """
    
    # Pre-create the file using python directly (simulating existing data)
    with open(csv_file, "w") as f:
        f.write(csv_content)
        
    print(f"Created test data at {csv_file}")
    
    # 2. Define Objective
    # - Read CSV (MCP: official_filesystem)
    # - Clean 'amount' column (Local Skill: data_processing.clean_data)
    # - Summarize stats (Local Skill: data_processing.read_csv_summary)
    # - Save report (MCP: official_filesystem)
    
    report_file = os.path.join(target_dir, "report.txt")
    
    objective = f"""
    1. Read the CSV file at {csv_file}.
    2. Clean the 'amount' column (strip whitespace).
    3. Generate a statistical summary of the cleaned data.
    4. Save the summary report to {report_file}.
    """
    
    print(f"\nRunning Agent with objective: {objective}")
    
    try:
        final_state = asyncio.run(agent.ainvoke(objective))
        
        print("\nExecution Completed.")
        
        # Validation
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                report_content = f.read()
            print("--- Report Content ---")
            print(report_content)
            print("----------------------")
            
            if "Avg=" in report_content:
                print("SUCCESS: Report contains statistics.")
            else:
                print("FAILURE: Report missing statistics.")
        else:
            print(f"FAILURE: Report file {report_file} not found.")
            
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_pipeline()
