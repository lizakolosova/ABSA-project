from src import (
    LexiconABSA, TransformerABSA, LLMABSA,
    setup_logging, get_logger
)

logger = get_logger(__name__)


def test_model(name, model_class, *args, **kwargs):
    print("\n" + "=" * 60)
    print(f"{name}")
    print("=" * 60)

    try:
        model = model_class(*args, **kwargs)
        text = "The food was amazing but the service was slow."
        results = model.analyze(text)

        if results:
            for result in results:
                print(f"  {result.aspect}: {result.sentiment} ({result.confidence:.2%})")
        else:
            print("  No aspects found")

        return model
    except ImportError as e:
        logger.error(f"{name} import failed: {e}")
        print(f"  ✗ Failed: {e}")
        if "keras" in str(e).lower():
            print("  → Fix: pip install torch transformers --upgrade")
        return None
    except RuntimeError as e:
        logger.error(f"{name} initialization failed: {e}")
        print(f"  ✗ Failed: {e}")
        if "ollama" in str(e).lower():
            print("  → Fix: Install Ollama from https://ollama.ai/")
            print("  → Then: ollama pull llama3")
        elif "spacy" in str(e).lower():
            print("  → Fix: python -m spacy download en_core_web_trf")
        return None
    except Exception as e:
        logger.error(f"{name} failed: {e}", exc_info=True)
        print(f"  ✗ Failed: {e}")
        return None


def main():
    setup_logging(level="INFO")

    text = "The food was amazing but the service was slow."
    logger.info(f"Testing models with: '{text}'")

    lexicon = test_model("LEXICON ABSA (Rule-based)", LexiconABSA)
    transformer = test_model("TRANSFORMER ABSA (Deep Learning)", TransformerABSA, device=-1)
    llm = test_model("LLM ABSA (Large Language Model)", LLMABSA, model="llama3")

    print("\n" + "=" * 60)
    print("BATCH PROCESSING")
    print("=" * 60)

    if lexicon:
        texts = [
            "Great ambiance and delicious pizza!",
            "The coffee was mediocre.",
            "Terrible experience."
        ]

        try:
            batch_results = lexicon.analyze_batch(texts)

            for i, (text, results) in enumerate(zip(texts, batch_results), 1):
                print(f"\n{i}. '{text}'")
                for result in results:
                    print(f"   - {result.aspect}: {result.sentiment}")
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    working = []
    if lexicon: working.append("LexiconABSA")
    if transformer: working.append("TransformerABSA")
    if llm: working.append("LLMABSA")

    if working:
        print(f"✓ Working models: {', '.join(working)}")
    else:
        print("✗ No models working - check error messages above")

    if not transformer:
        print("\nNote: TransformerABSA may need: pip install torch transformers --upgrade")
    if not llm:
        print("Note: LLMABSA needs Ollama installed: https://ollama.ai/")


if __name__ == "__main__":
    main()