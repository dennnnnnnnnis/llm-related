# test_tools.py
import logging
from tools import create_file, str_replace, shell_exec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_create_file():
    print("Testing create_file...")
    result = create_file({"file_name": "test.txt", "file_contents": "This is a test"})
    print(f"Result: {result}")

def test_str_replace():
    print("Testing str_replace...")
    result = str_replace({"text": "Hello world", "old": "world", "new": "universe"})
    print(f"Result: {result}")

def test_shell_exec():
    print("Testing shell_exec...")
    result = shell_exec({"command": "ls -la"})
    print(f"Result: {result}")

if __name__ == "__main__":
    test_create_file()
    test_str_replace()
    test_shell_exec()