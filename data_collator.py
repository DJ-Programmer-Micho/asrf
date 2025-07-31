import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Ensure 1D float arrays
        input_values = [np.array(f["input_values"], dtype=np.float32).squeeze() for f in features]
        labels = [str(f["labels"]) for f in features]  # Ensure labels are strings

        # Manual padding for input_values
        max_input_len = max(len(arr) for arr in input_values)
        padded_inputs = np.zeros((len(input_values), max_input_len), dtype=np.float32)
        for i, arr in enumerate(input_values):
            padded_inputs[i, :len(arr)] = arr

        input_values_tensor = torch.tensor(padded_inputs, dtype=torch.float32)

        # Pad labels using processor directly (no `as_target_processor`)
        batch_labels = self.processor(
            text=labels,
            padding=self.padding,
            return_tensors="pt"
        )
        batch_labels["input_ids"][batch_labels["input_ids"] == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_values": input_values_tensor,
            "labels": batch_labels["input_ids"]
        }
