import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def extract_labels_from_absa_outputs(absa_outputs):
    """
    Converts model outputs (AspectSentiment objects or tuples)
    into a dictionary {aspect: sentiment}.
    Works safely across Lexicon, Transformer, and LLM models.
    """
    label_dict = {}

    for item in absa_outputs:
        # Case 1: model returned ("aspect", "sentiment") tuple
        if isinstance(item, tuple) and len(item) == 2:
            aspect, sentiment = item
            label_dict[aspect.lower()] = sentiment.lower()
        # Case 2: model returned AspectSentiment or similar object
        else:
            try:
                aspect = getattr(item, "aspect", None)
                sentiment = getattr(item, "sentiment", None)
                if aspect and sentiment:
                    label_dict[aspect.lower()] = sentiment.lower()
            except Exception:
                continue

    return label_dict


def extract_labels_and_confidence_from_llm_outputs(llm_outputs):
    label_dict, conf_dict = {}, {}
    for item in llm_outputs:
        aspect = getattr(item, "aspect", "").lower()
        sentiment = getattr(item, "sentiment", "").lower()
        confidence = getattr(item, "confidence", 0.0)
        if aspect:
            label_dict[aspect] = sentiment
            conf_dict[aspect] = confidence
    return label_dict, conf_dict


# ----------- METRICS AND EVALUATION -----------

def evaluate_predictions(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


def average_confidence(conf_dict, aspects):
    confidences = [conf_dict.get(a.lower(), 0) for a in aspects]
    return sum(confidences) / len(confidences) if confidences else 0.0


def run_evaluation_on_model(model, test_data, is_llm=False):
    all_true, all_pred, confidences = [], [], []
    total_time = 0.0
    all_outputs = []

    for text, true_labels in test_data:
        start = time.time()
        pred_outputs = model.analyze(text)
        total_time += time.time() - start
        all_outputs.append(pred_outputs)

        if not pred_outputs:
            pred_dict, conf_dict = {}, {}
        else:
            if is_llm:
                pred_dict, conf_dict = extract_labels_and_confidence_from_llm_outputs(pred_outputs)
            else:
                pred_dict = extract_labels_from_absa_outputs(pred_outputs)
                conf_dict = {}

        true_dict = {a.lower(): s.lower() for a, s in true_labels}
        for aspect in true_dict:
            all_true.append(true_dict[aspect])
            all_pred.append(pred_dict.get(aspect, 'missing'))
            if is_llm:
                confidences.append(conf_dict.get(aspect, 0.0))

    precision, recall, f1, accuracy = evaluate_predictions(all_true, all_pred)
    avg_conf = average_confidence(conf_dict, true_dict.keys()) if confidences else None

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "avg_confidence": avg_conf,
        "avg_inference_time": total_time / len(test_data) if len(test_data) else 0.0
    }
    return metrics, all_outputs