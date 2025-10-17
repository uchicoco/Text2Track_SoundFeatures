import torch
from torch.utils.data import Dataset
import pandas as pd

class T5Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.df = df
        self.max_length = max_length

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                dict: {
                    'input_ids': list,
                    'attention_mask': list,
                    'labels': list
                }
        """
        row = self.df.iloc[index]
        prompt = str(row['prompt'])
        target = str(row['target'])

        # Tokenize prompt (no padding here - DataCollator will handle it)
        source_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
        )

        # Tokenize target (no padding here - DataCollator will handle it)
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            truncation=True,
        )

        return {
            "input_ids": source_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
            "labels": target_encoding["input_ids"],
        }