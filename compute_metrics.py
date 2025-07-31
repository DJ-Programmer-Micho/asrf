# import jiwer

# def compute_metrics(pred):
#     pred_ids = pred.predictions
#     label_ids = pred.label_ids

#     pred_str = pred.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = pred.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

#     wer = jiwer.wer(label_str, pred_str)
#     return {"wer": wer}

import evaluate
from typing import Callable

wer_metric = evaluate.load("wer")

def get_compute_metrics_fn(processor) -> Callable:
    def compute_metrics(pred):
        pred_ids = pred.predictions.argmax(-1)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # 🛑 Filter out empty references to avoid zero division
        filtered_preds, filtered_refs = [], []
        for p, r in zip(pred_str, label_str):
            if r.strip():  # keep only if reference is not empty
                filtered_preds.append(p)
                filtered_refs.append(r)

        if len(filtered_refs) == 0:
            return {"wer": 1.0}  # Worst-case fallback if everything is empty

        wer = wer_metric.compute(predictions=filtered_preds, references=filtered_refs)
        return {"wer": wer}

    return compute_metrics
