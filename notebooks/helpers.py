import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def extract_labels_from_outputs(absa_outputs):
    label_dict = {}
    for item in absa_outputs:
        try:
            aspect = getattr(item, "aspect", None)
            sentiment = getattr(item, "sentiment", None)
            if aspect and sentiment:
                label_dict[aspect.lower()] = sentiment.lower()
        except Exception:
            continue
    return label_dict


def extract_labels_and_confidence(absa_outputs):
    label_dict, conf_dict = {}, {}
    for item in absa_outputs:
        try:
            aspect = getattr(item, "aspect", "").lower()
            sentiment = getattr(item, "sentiment", "").lower()
            confidence = getattr(item, "confidence", 0.0)
            if aspect:
                label_dict[aspect] = sentiment
                conf_dict[aspect] = confidence
        except Exception:
            continue
    return label_dict, conf_dict


def evaluate_predictions(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


def run_evaluation(model, test_data):
    all_true, all_pred, confidences = [], [], []
    total_time = 0.0
    all_outputs = []

    for text, true_labels in test_data:
        start = time.time()
        pred_outputs = model.analyze(text)
        elapsed = time.time() - start
        total_time += elapsed
        all_outputs.append(pred_outputs)

        pred_dict, conf_dict = extract_labels_and_confidence(pred_outputs)
        true_dict = {a.lower(): s.lower() for a, s in true_labels}

        for aspect in true_dict:
            all_true.append(true_dict[aspect])
            all_pred.append(pred_dict.get(aspect, 'neutral'))
            confidences.append(conf_dict.get(aspect, 0.0))

    precision, recall, f1, accuracy = evaluate_predictions(all_true, all_pred)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "avg_confidence": avg_conf,
        "avg_inference_time": total_time / len(test_data) if test_data else 0.0
    }, all_outputs