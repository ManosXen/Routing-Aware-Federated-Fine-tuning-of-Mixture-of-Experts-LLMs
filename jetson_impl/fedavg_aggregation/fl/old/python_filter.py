PYTHON_KEYWORDS = [
    '```python',
    'Python',
    'python'
]

def is_python_example(example):
    """
    Checks if an example is likely a Python example by searching
    for keywords in the instruction or output.
    """
    # Combine the instruction and output text for a comprehensive search
    full_text = example['instruction'].lower() + example['output'].lower()
    
    # Check if any of our keywords are present
    for keyword in PYTHON_KEYWORDS:
        if keyword.lower() in full_text:
            return True
            
    return False