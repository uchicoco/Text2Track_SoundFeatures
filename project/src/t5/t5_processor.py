import multiprocessing
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import umap
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments
)

# Handle imports for both module usage and standalone execution
try:
    from ..data.dataset_processor import DatasetProcessor
    from .t5_utils import T5Dataset
except ImportError:
    # When run as standalone script, add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.data.dataset_processor import DatasetProcessor
    from src.t5.t5_utils import T5Dataset

class T5Processor:
    # Template strings for prompt generation
    PROMPT_TEMPLATES = (
        "recommend a song with {tags}",
        "find a track that is {tags}",
        "suggest music with {tags}",
        "what song has {tags}?",
        "which track is {tags}?",
        "play something {tags}",
        "show me {tags} music",
        "I want {tags} music",
        "looking for {tags} songs",
        "need {tags} tracks"
    )
    
    def __init__(self, model_name="t5-base"):
        self.dp = DatasetProcessor()
        self.output_dir = self.dp.output_dir
        self.current_dir = self.dp.current_dir
        self.data_path = Path(self.output_dir) / "data/dataset_with_semantic_ids.csv"
        self.train_path = Path(self.output_dir) / "data/train.csv"
        self.val_path = Path(self.output_dir) / "data/val.csv"
        self.test_path = Path(self.output_dir) / "data/test.csv"
        self.model_path = Path(self.current_dir) / "models/t5_semantic_id_model"
        self.model_name = model_name

    def create_prompt(self, row):
        """
        Create a prompt from the given row of the DataFrame.
        """
        tag_list = []

        if not pd.isna(row.mood):
            tag_list.append(row.mood)
        if not pd.isna(row.genre):
            tag_list.append(row.genre)
        if not pd.isna(row.instrument):
            tag_list.append(f'with {row.instrument}')

        if len(tag_list) == 0:
            return None  # No valid tags ??
        elif len(tag_list) == 1:
            tags_text = tag_list[0]
        elif len(tag_list) == 2:
            connectors = [" and ", " ", ", "]
            tags_text = random.choice(connectors).join(tag_list)
        else:
            # Handle 3 or more tags
            tags_text = f"{tag_list[0]} {tag_list[1]} {tag_list[2]}"

        template = random.choice(self.PROMPT_TEMPLATES)
        prompt = template.format(tags=tags_text)
        return prompt

    def prepare_data(self, df_ids=None, csv_path=None, upsample_threshold=16, max_replication=6):
        # Create prompt-target pairs
        print("Creating prompt-target pairs")
        if df_ids is not None:
            df = df_ids.copy()
        elif csv_path is not None:
            df = pd.read_csv(csv_path)
        else:
            df = pd.read_csv(self.data_path)

        # Upsampling
        target_counts = df['semantic_id'].value_counts()

        all_data = []
        for row in df.itertuples(index=False):
            target = row.semantic_id
            target_count = target_counts[target]

            if target_count < upsample_threshold:
                # Calculate replication factor
                sample_count = min(max_replication, upsample_threshold // target_count + 1)

                # Generate different prompt variations for each replication
                for _ in range(sample_count):
                    prompt = self.create_prompt(row)  # Random template each time
                    all_data.append({
                        'prompt': prompt,
                        'target': target
                    })
            else:
                # Data with >threshold occurrences: add once
                prompt = self.create_prompt(row)
                all_data.append({
                    'prompt': prompt,
                    'target': target
                })
        df_t5 = pd.DataFrame(all_data).dropna()

        # Split into train, val, test sets (80:10:10 %)
        print("Splitting dataset")
        train_df, temp_df = train_test_split(df_t5, test_size=0.2, random_state=42)
        train_df = train_df.reset_index(drop=True)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Save to files
        data_dir = Path(self.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(data_dir / "train.csv", index=False)
        val_df.to_csv(data_dir / "val.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

        return train_df, val_df, test_df

    def train_t5(self, train_df=None, val_df=None, train_path=None, val_path=None):
        """
        Train a T5 model for semantic ID generation.
        Args:
            train_df (pd.DataFrame): Training dataset with "prompt" and "target" columns
            val_df (pd.DataFrame): Validation dataset with "prompt" and "target" columns
        Returns:
            tokenizer: Trained T5 tokenizer
            model: Trained T5 model
        """
        tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        model = T5ForConditionalGeneration.from_pretrained(self.model_name)

        # Add special tokens for semantic IDs (<000> to <255>)
        special_tokens = [f"<{i:03d}>" for i in range(256)]
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            print(f"Added {num_added} special tokens to vocabulary")

        # Create datasets
        if train_df is not None:
            train_raw = train_df.copy()
        elif train_path is not None:
            train_raw = pd.read_csv(train_path)
        else:
            train_raw = pd.read_csv(self.train_path)

        if val_df is not None:
            val_raw = val_df.copy()
        elif val_path is not None:
            val_raw = pd.read_csv(val_path)
        else:
            val_raw = pd.read_csv(self.val_path)

        train_data = T5Dataset(train_raw, tokenizer)
        val_data = T5Dataset(val_raw, tokenizer)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )

        # Parallel processing
        num_workers = min(4, multiprocessing.cpu_count())

        training_args = TrainingArguments(
            output_dir=self.model_path,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            dataloader_num_workers=num_workers,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
        )

        print("Start training")
        trainer.train()
        print("Training finished. Saving the best model")

        # Create model directory if it doesn't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(self.model_path)
        tokenizer.save_pretrained(self.model_path)

        return tokenizer, model

    def evaluate_t5(self,
                    tokenizer=None, model=None, test_df=None,
                    model_path=None, test_path=None):
        """
        Evaluate the T5 model using Hits@10 metric
        Args:
            tokenizer: Trained T5 tokenizer
            model: Trained T5 model
            test_df (pd.DataFrame): Test dataset with "prompt" and "target" columns
        Returns:
            hits_at_10 (float): Hits@10 score
        """
        if tokenizer is not None:
            tokenizer = tokenizer
        elif model_path is not None:
            tokenizer = T5Tokenizer.from_pretrained(model_path)
        else:
            tokenizer = T5Tokenizer.from_pretrained(self.model_path)

        if model is not None:
            model = model
        elif model_path is not None:
            model = T5ForConditionalGeneration.from_pretrained(model_path)
        else:
            model = T5ForConditionalGeneration.from_pretrained(self.model_path)

        if test_df is not None:
            test_df = test_df.copy()
        elif test_path is not None:
            test_df = pd.read_csv(test_path)
        else:
            test_df = pd.read_csv(self.test_path)
        
        test_dataset = T5Dataset(test_df, tokenizer)
        print("Calculating Hits@10")
        model.eval()
        hits = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Data collator for dynamic padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )

        # Batch processing
        batch_size = 16
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator
        )

        for batch_idx, batch in enumerate(tqdm(test_loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get true targets from original dataframe
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(test_df))
            true_targets = test_df.iloc[start_idx:end_idx]['target'].tolist()

            # Beam search to generate top 10 candidates
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                num_beams=10,
                num_return_sequences=10,
                max_length=64
            )

            # Check if the correct target is in the generated candidates
            for i in range(len(true_targets)):
                generated_targets = [tokenizer.decode(outputs[j], skip_special_tokens=True) 
                                     for j in range(i*10, (i+1)*10)]

                if true_targets[i] in generated_targets:
                    hits += 1
                
                # Debug: Print first few examples
                if batch_idx == 0 and i < 3:
                    print(f"\nExample {i+1}:")
                    print(f"  Prompt: {test_df.iloc[i]['prompt']}")
                    print(f"  True target: {true_targets[i]}")
                    print(f"  Generated (top 3): {generated_targets[:3]}")
                    print(f"  Match: {true_targets[i] in generated_targets}")

        return hits / len(test_dataset)

    def visualize_umap(self, 
                       tokenizer=None, model=None, test_df=None,
                       model_path=None, test_path=None):
        """
        Visualize the embeddings from the T5 encoder using UMAP.
        Args:
            model: Trained T5 model
            test_df (pd.DataFrame): Test dataset with "prompt" and "target" columns
        
        """
        if tokenizer is None:
            if model_path is not None:
                tokenizer = T5Tokenizer.from_pretrained(model_path)
            else:
                tokenizer = T5Tokenizer.from_pretrained(self.model_path)

        if model is None:
            if model_path is not None:
                model = T5ForConditionalGeneration.from_pretrained(model_path)
            else:
                model = T5ForConditionalGeneration.from_pretrained(self.model_path)

        if test_df is not None:
            test_df = test_df.copy()
        elif test_path is not None:
            test_df = pd.read_csv(test_path)
        else:
            test_df = pd.read_csv(self.test_path)

        test_dataset = T5Dataset(test_df, tokenizer)

        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        prompt_embeddings = []
        labels = []

        # Calculate embeddings from the encoder
        for i in tqdm(range(len(test_dataset))):
            data = test_dataset[i]
            # Convert lists to tensors and add batch dimension
            input_ids = torch.tensor(data["input_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(data["attention_mask"]).unsqueeze(0).to(device)

            with torch.no_grad():
                encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                embedding = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy() # mean pooling

            prompt_embeddings.append(embedding)
            labels.append(test_df.iloc[i]["target"].split(">")[1]) # color by the first token

        # Reduce to 3D
        reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_3d = reducer.fit_transform(prompt_embeddings)

        print("Creating UMAP plot")
        df_3d = pd.DataFrame({
            "UMAP_1": embedding_3d[:, 0],
            "UMAP_2": embedding_3d[:, 1],
            "UMAP_3": embedding_3d[:, 2],
            "semantic_token": [label + ">" for label in labels],
            "prompt": test_df["prompt"]
        })

        fig = px.scatter_3d(
            df_3d,
            x="UMAP_1",
            y="UMAP_2",
            z="UMAP_3",
            color="semantic_token",
            hover_data=["prompt"],
            title="T5 Encoder UMAP Visualization"
        )
        fig.write_html(Path(self.output_dir) / "figures/umap.html")
        print("UMAP plot saved to outputs/figures/umap.html")

        fig.show()

#!!! Run this after generating csv file with semantic IDs !!!
def main():
    t5p = T5Processor(model_name="t5-small")

    try:
        df_ids = pd.read_csv(t5p.data_path)
        train_df, val_df, test_df = t5p.prepare_data(df_ids=df_ids)
    except FileNotFoundError:
        print(f"File not found: {t5p.data_path}")
        return

    tokenizer, model = t5p.train_t5(train_df=train_df, val_df=val_df)
    hits_at_10 = t5p.evaluate_t5(tokenizer=tokenizer, model=model, test_df=test_df)
    
    print(f"Hits@10: {hits_at_10:.4f}")
    t5p.visualize_umap(model=model, test_df=test_df)

if __name__ == "__main__":
    main()