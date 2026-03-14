import os
import re

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix test functions
    content = re.sub(r'def test_([a-zA-Z0-9_]+)\(\):', r'def test_\1() -> None:', content)
    content = re.sub(r'def setup_module\(\):', r'def setup_module() -> None:', content)
    content = re.sub(r'def test_([a-zA-Z0-9_]+)\(self\):', r'def test_\1(self) -> None:', content)

    # Any other def without type hints in tests
    if 'test_main' in filepath:
        content = re.sub(r'def mock_input\(prompt\):', r'def mock_input(prompt: str) -> str:', content)
        content = re.sub(r'def test_([a-zA-Z0-9_]+)\(([^)]+)\):', lambda m: f'def test_{m.group(1)}({m.group(2)}) -> None:' if '->' not in m.group(0) else m.group(0), content)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for root, _, files in os.walk('tests'):
    for file in files:
        if file.endswith('.py'):
            fix_file(os.path.join(root, file))
