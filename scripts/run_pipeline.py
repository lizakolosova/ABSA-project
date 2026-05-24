"""
End-to-end ABSA comparison pipeline.

Usage:
    python scripts/run_pipeline.py                 # all three models
    python scripts/run_pipeline.py --skip-llm      # skip LLMABSA (no Ollama needed)
    python scripts/run_pipeline.py --skip-transformer  # skip TransformerABSA
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def load_eval_data(path: Path):
    with open(path) as f:
        raw = json.load(f)
    return [
        (item["text"], [(a["aspect"], a["sentiment"]) for a in item["aspects"]])
        for item in raw
    ]


def run_evaluation(model, test_data):
    all_true, all_pred, all_conf = [], [], []
    total_time = 0.0

    for text, true_labels in test_data:
        start = time.time()
        outputs = model.analyze(text)
        total_time += time.time() - start

        pred_map = {o.aspect.lower(): o.sentiment.lower() for o in outputs}
        conf_map = {o.aspect.lower(): o.confidence for o in outputs}

        for aspect, true_sent in true_labels:
            all_true.append(true_sent.lower())
            all_pred.append(pred_map.get(aspect.lower(), "neutral"))
            all_conf.append(conf_map.get(aspect.lower(), 0.0))

    if not all_true:
        return {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0,
                "avg_confidence": 0, "avg_inference_time_s": 0}

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(all_true, all_pred)
    avg_conf = sum(all_conf) / len(all_conf) if all_conf else 0.0
    avg_time = total_time / len(test_data) if test_data else 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
        "avg_confidence": round(float(avg_conf), 4),
        "avg_inference_time_s": round(float(avg_time), 4),
    }


def print_report(results: dict) -> None:
    col = 22
    divider = "=" * (col * (len(results) + 1))
    print(f"\n{divider}")
    header = f"{'Metric':<{col}}" + "".join(f"{k:<{col}}" for k in results)
    print(header)
    print(divider)
    metrics = list(next(iter(results.values())).keys())
    for metric in metrics:
        row = f"{metric:<{col}}" + "".join(
            f"{vals[metric]:<{col}}" for vals in results.values()
        )
        print(row)
    print(divider)


def main():
    parser = argparse.ArgumentParser(description="ABSA comparison pipeline")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLMABSA (requires Ollama)")
    parser.add_argument("--skip-transformer", action="store_true",
                        help="Skip TransformerABSA (requires GPU/large download)")
    parser.add_argument("--data", default=str(ROOT / "data" / "evaluation_data.json"),
                        help="Path to evaluation data JSON")
    parser.add_argument("--output", default=str(ROOT / "data" / "evaluation_report.json"),
                        help="Path for the output report JSON")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: evaluation data not found at {data_path}")
        sys.exit(1)

    print(f"Loading evaluation data from {data_path} ...")
    test_data = load_eval_data(data_path)
    print(f"Loaded {len(test_data)} samples.\n")

    results = {}

    # --- Step 1: LexiconABSA ---
    print("Step 1 — LexiconABSA (spaCy + VADER)")
    from src.lexicon_absa import LexiconABSA
    lexicon = LexiconABSA()
    results["LexiconABSA"] = run_evaluation(lexicon, test_data)
    m = results["LexiconABSA"]
    print(f"  precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  latency={m['avg_inference_time_s']:.3f}s/sample")

    # --- Step 2: TransformerABSA ---
    if not args.skip_transformer:
        print("\nStep 2 — TransformerABSA (RoBERTa + DeBERTa)")
        from src.transformer_absa import TransformerABSA
        transformer = TransformerABSA()
        results["TransformerABSA"] = run_evaluation(transformer, test_data)
        m = results["TransformerABSA"]
        print(f"  precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  latency={m['avg_inference_time_s']:.3f}s/sample")
    else:
        print("\nStep 2 — TransformerABSA  [skipped]")

    # --- Step 3: LLMABSA ---
    if not args.skip_llm:
        print("\nStep 3 — LLMABSA (Ollama / Llama3)")
        from src.llm_absa import LLMABSA
        llm = LLMABSA()
        results["LLMABSA"] = run_evaluation(llm, test_data)
        m = results["LLMABSA"]
        print(f"  precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  latency={m['avg_inference_time_s']:.3f}s/sample")
    else:
        print("\nStep 3 — LLMABSA  [skipped]")

    print_report(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
