import csv
import io
import statistics
from typing import Dict, Any, List

def read_csv_summary(file_path: str = None, content: str = None) -> str:
    """
    Parses CSV content and returns a summary.
    Summary includes: headers, row count, and basic stats for numeric columns.
    
    Args:
        file_path: Optional path to the CSV file (used for reference or reading if content is missing).
        content: The actual CSV string content. If None, tries to read from file_path.
    """
    try:
        # If content is missing but file_path is provided, try to read from file
        if not content and file_path:
            try:
                import os
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    # Maybe it's not a path but the content itself passed mistakenly?
                    # Check if it looks like CSV
                    if "," in file_path or "\n" in file_path:
                        content = file_path
                        file_path = "In-Memory Content"
            except Exception:
                pass

        if not content:
            return "Error: No content provided for CSV analysis."
            
        f = io.StringIO(content)
        reader = csv.DictReader(f)
        rows = list(reader)
        
        if not rows:
            return "CSV is empty."
            
        headers = reader.fieldnames
        row_count = len(rows)
        
        # Calculate stats for numeric columns
        stats_summary = []
        if rows:
            for header in headers:
                values = []
                is_numeric = True
                for row in rows:
                    try:
                        val = float(row[header])
                        values.append(val)
                    except ValueError:
                        is_numeric = False
                        break
                
                if is_numeric and values:
                    avg = statistics.mean(values)
                    stats_summary.append(f"- {header}: Avg={avg:.2f}, Min={min(values)}, Max={max(values)}")
        
        stats_str = "\n".join(stats_summary) if stats_summary else "No numeric columns found."
        
        return f"""CSV Summary for {file_path}:
Headers: {', '.join(headers) if headers else 'None'}
Row Count: {row_count}
Statistics:
{stats_str}
"""
    except Exception as e:
        return f"Error processing CSV: {str(e)}"

def clean_data(content: str, column_to_clean: str, operation: str = "strip") -> str:
    """
    Performs basic cleaning on a specific column in CSV content.
    Operations: 'strip' (whitespace), 'upper', 'lower'.
    Returns the new CSV content string.
    """
    try:
        f = io.StringIO(content)
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames
        
        if not headers or column_to_clean not in headers:
            return f"Error: Column '{column_to_clean}' not found in headers: {headers}"
            
        new_rows = []
        for row in rows:
            val = row.get(column_to_clean, "")
            if val is None:
                val = ""
            
            if operation == "strip":
                row[column_to_clean] = val.strip()
            elif operation == "upper":
                row[column_to_clean] = val.upper()
            elif operation == "lower":
                row[column_to_clean] = val.lower()
            new_rows.append(row)
            
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(new_rows)
        return output.getvalue()
        
    except Exception as e:
        return f"Error cleaning data: {str(e)}"
