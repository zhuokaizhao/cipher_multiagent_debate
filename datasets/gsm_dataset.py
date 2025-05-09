import numpy as np
from torch.utils.data import Dataset
import random
from models.utils import read_jsonl
import json
from glob import glob
from typing import Optional


class GSM8K(Dataset):
    def __init__(
        self,
        input_path: str,
        data_name: str,
        custom_range: Optional[str] = None,
        seed: int = 0,
    ):
        random.seed(seed)
        np.random.seed(seed)

        # all_questions = read_jsonl(input_path)
        # random.shuffle(all_questions)
        # self.questions = all_questions[:n_ques]

        jsons = sorted(glob(input_path))
        random.shuffle(jsons)
        test_problems = []

        if data_name == "math":
            for json_file in jsons:
                data = json.load(open(json_file, "r"))
                if (
                    ("1" in data["level"])
                    or ("2" in data["level"])
                    or ("3" in data["level"])
                ):
                    test_problems.append(data)

        elif data_name == "gsm8k":
            for json_file in jsons:
                with open(json_file, "r") as f:
                    lines = f.readlines()
                    test_problems.extend(lines)

            for i in range(len(test_problems)):
                test_problems[i] = json.loads(test_problems[i])

        else:
            raise NotImplementedError()

        random.shuffle(test_problems)

        # Handle custom range if specified
        if custom_range:
            try:
                start_idx, end_idx = map(int, custom_range.split("-"))

                # Convert from 1-indexed (human-friendly) to 0-indexed (Python)
                # But only if start_idx is > 0 to avoid subtracting from 0
                start_idx_0 = start_idx - 1 if start_idx > 0 else start_idx

                # Keep end_idx as is since slicing is exclusive on the end
                end_idx_0 = end_idx

                if (
                    start_idx <= 0
                    or end_idx > len(test_problems)
                    or start_idx > end_idx
                ):
                    print(
                        f"Invalid custom range {custom_range}, using default selection"
                    )
                else:
                    print(
                        f"Using problem range {custom_range} (converting to 0-indexed: {start_idx_0}:{end_idx_0})"
                    )
                    test_problems = test_problems[start_idx_0:end_idx_0]
                    print(f"Selected {len(test_problems)} problems")
                    # Adjust num_problems_to_process to match the custom range
                    num_problems_to_process = len(test_problems)
            except Exception as e:
                print(f"Error parsing custom range: {e}, using default selection")

        # Ensure we don't go out of bounds
        self.questions = test_problems[:num_problems_to_process]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx]
