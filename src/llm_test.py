# run_llm_test.py
from src.llm_absa import LLMABSA

if __name__ == "__main__":
    analyzer = LLMABSA(model="llama3")  # or "mistral" depends on what I pull
    text = "The laptop has a great screen but terrible battery life."
    results = analyzer.analyze(text)
    for r in results:
        print(r)
