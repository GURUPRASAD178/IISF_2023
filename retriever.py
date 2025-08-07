import os

def retrieve_context(query):
    # Placeholder: Load a sample context
    try:
        sample_file = os.path.join("data", "sample_bhuvan_text.txt")
        with open(sample_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Context not found. Please load Bhuvan data."
