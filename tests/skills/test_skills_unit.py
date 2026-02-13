import unittest
import os
import shutil
from skills.calculator import calculate
from skills.filesystem import read_file, write_file, _get_sandbox_path
from skills.text_processing import count_words, summarize_text
from skills.web_search import local_web_search
import json

class TestRealSkills(unittest.TestCase):

    def setUp(self):
        # Clean sandbox
        self.sandbox = _get_sandbox_path()
        if os.path.exists(self.sandbox):
            shutil.rmtree(self.sandbox)
        os.makedirs(self.sandbox)

    def tearDown(self):
        if os.path.exists(self.sandbox):
            shutil.rmtree(self.sandbox)

    # --- Calculator ---
    def test_calculator_basic(self):
        self.assertEqual(calculate("1 + 1"), 2.0)
        self.assertEqual(calculate("50 * 3"), 150.0)
        self.assertEqual(calculate("10 / 2"), 5.0)

    def test_calculator_math_functions(self):
        self.assertEqual(calculate("max(1, 5)"), 5.0)
        self.assertEqual(calculate("abs(-10)"), 10.0)

    def test_calculator_safety(self):
        with self.assertRaises(ValueError):
            calculate("__import__('os').system('ls')")
            
    # --- Filesystem ---
    def test_filesystem_write_read(self):
        filename = "test.txt"
        content = "Hello World"
        
        # Write
        msg = write_file(filename, content)
        self.assertIn("Successfully wrote", msg)
        
        # Read
        read_content = read_file(filename)
        self.assertEqual(read_content, content)

    def test_filesystem_security(self):
        with self.assertRaises(ValueError):
            read_file("../outside.txt")
            
    # --- Text Processing ---
    def test_text_processing(self):
        text = "Hello world this is a test"
        self.assertEqual(count_words(text), 6)
        
        long_text = "a" * 200
        summary = summarize_text(long_text, max_length=10)
        self.assertTrue(len(summary) <= 13) # 10 + "..."
        
    # --- Web Search ---
    def test_web_search(self):
        result_json = local_web_search("iPhone 16")
        data = json.loads(result_json)
        self.assertTrue(len(data) > 0)
        self.assertIn("iPhone 16", data[0]['title'])

if __name__ == '__main__':
    unittest.main()
