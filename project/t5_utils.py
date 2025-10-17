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
                    'input_ids': torch.Tensor,
                    'attention_mask': torch.Tensor,
                    'labels': torch.Tensor
                }
        """
        row = self.df.iloc[index]
        prompt = str(row['prompt'])
        target = str(row['target'])

        # tokenize prompt
        source_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # tokenize target
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        labels = target_encoding['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            "prompt": prompt,
            "target": target
        }