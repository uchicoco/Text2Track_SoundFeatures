"""
Lightweight test script for RVQ-based pipeline.

Quick validation with:
- First 100 tracks only
- Smaller RVQ configuration (L=2, K=32)
- 1 training epoch
- Minimal evaluation
"""

import warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
import re

from config import settings
from src.data.dataset_processor import DatasetProcessor
from src.data.feature_extractor import FeatureExtractor
from src.models.pca_processor import PCAProcessor
from src.models.kmeans_clusterer import KMeansProcessor
from src.t5.t5_processor import T5Processor


# Lightweight RVQ config
RVQ_CONFIG = {
    "pca_components": 24,  # Fewer components
    "L": 2,                # 2 levels
    "K": 32,               # Fewer clusters
    "n_init": 8,           # Fewer initializations
    "max_iter": 100,       # Fewer iterations
}

# Template prompts (simplified)
TEMPLATES = {
    ("genre", "mood"): ["a {mood} {genre} track"],
    ("genre", "instrument"): ["{genre} with {instrument}"],
    ("mood", "instrument"): ["{mood} music with {instrument}"],
    ("genre", "mood", "instrument"): ["a {mood} {genre} track with {instrument}"],
}


def create_prompt_target_pairs(df):
    """Create prompt-target pairs (simplified version)."""
    def row_tags(r):
        vals = {}
        g = str(r.get("genre", "")).strip()
        m = str(r.get("mood", "")).strip()
        i = str(r.get("instrument", "")).strip()
        if g: vals["genre"] = g
        if m: vals["mood"] = m
        if i: vals["instrument"] = i
        return vals
    
    records = []
    for _, r in df.iterrows():
        vals = row_tags(r)
        for k in (2, 3):
            if len(vals) >= k:
                for combo_keys in combinations(sorted(vals.keys()), k):
                    tmpls = TEMPLATES.get(combo_keys, [])
                    for tpl in tmpls:
                        records.append({
                            "track_id": r["track_id"],
                            "prompt": tpl.format(**{key: vals[key] for key in combo_keys}),
                            "target": r["semantic_id"],
                        })
    
    return pd.DataFrame.from_records(records, columns=["track_id", "prompt", "target"])


def main():
    print("=" * 60)
    print("RVQ TEST MODE: Lightweight validation")
    print("=" * 60)
    
    # STEP 1: Load limited data
    print("\nSTEP 1: Loading first 100 tracks...")
    dp = DatasetProcessor()
    mood_df = dp.load_tag_tsv("autotagging_moodtheme.tsv", "mood").head(100)
    genre_df = dp.load_tag_tsv("autotagging_genre.tsv", "genre").head(100)
    inst_df = dp.load_tag_tsv("autotagging_instrument.tsv", "instrument").head(100)
    
    tags_merged = (mood_df
                   .merge(genre_df, on=["track_id", "path"], how="outer")
                   .merge(inst_df, on=["track_id", "path"], how="outer"))
    tags_merged[["genre", "mood", "instrument"]] = tags_merged[["genre", "mood", "instrument"]].fillna("")
    print(f"✓ Loaded {len(tags_merged)} tracks")
    
    # STEP 2: Extract features (minimal)
    print("\nSTEP 2: Extracting features...")
    fe = FeatureExtractor()
    df = fe.build_features_dataframe(
        tags_merged,
        use_mel_alt=False,
        use_energy_alt=False,
        add_valley=False,
        add_timbre_dist=False,
        add_tonality=False,
        add_rhythm_struct=False
    )
    df_two = fe.filter_min2tags(df)
    print(f"✓ {len(df_two)} tracks with 2+ tags")
    
    # STEP 3: PCA
    print("\nSTEP 3: PCA...")
    pp = PCAProcessor()
    feat_matrix = pp.build_feature_matrix(df_two)
    feat_pca, _, _, _ = pp.run_pca_components(feat_matrix, n_components=RVQ_CONFIG["pca_components"])
    print(f"✓ Reduced to {RVQ_CONFIG['pca_components']} dimensions")
    
    # STEP 4: RVQ
    print("\nSTEP 4: RVQ Semantic IDs...")
    kmp = KMeansProcessor()
    codebooks, codes, L, K = kmp.run_rvq(
        feat_pca,
        n_tokens=RVQ_CONFIG["L"],
        n_clusters=RVQ_CONFIG["K"],
        max_iter=RVQ_CONFIG["max_iter"],
        n_init=RVQ_CONFIG["n_init"]
    )
    
    # Convert codes to semantic IDs
    semantic_ids = []
    for row in codes:
        semantic_id = "".join([f"<Q{int(c):03d}>" for c in row])
        semantic_ids.append(semantic_id)
    
    df_two["semantic_id"] = semantic_ids
    
    vc = pd.Series(semantic_ids).value_counts()
    print(f"✓ {vc.size} unique IDs, avg {vc.mean():.1f} tracks/ID")
    
    # STEP 5: Prompts
    print("\nSTEP 5: Creating prompts...")
    pairs_df = create_prompt_target_pairs(df_two)
    print(f"✓ {len(pairs_df)} pairs")
    
    # STEP 6: Split (simple 80/10/10)
    print("\nSTEP 6: Splitting...")
    pairs_df = pairs_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(pairs_df)
    train_df = pairs_df[:int(0.8*n)]
    val_df = pairs_df[int(0.8*n):int(0.9*n)]
    test_df = pairs_df[int(0.9*n):]
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Show some training examples
    print("\n📝 Sample training data:")
    for _, row in train_df.head(3).iterrows():
        print(f"  Prompt: '{row['prompt']}' -> Target: '{row['target']}'")
    
    # STEP 7: Train (1 epoch)
    print("\nSTEP 7: Training T5 (1 epoch)...")
    from transformers import TrainingArguments, Trainer
    from src.t5.t5_utils import T5Dataset
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch
    
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    
    special_tokens = [f"<Q{i:03d}>" for i in range(256)]
    added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"✓ Added {added} special tokens")
    print(f"✓ Tokenizer vocab size: {len(tokenizer)}")
    
    # Verify special token encoding
    test_target = train_df.iloc[0]["target"]
    test_encoded = tokenizer(test_target, return_tensors="pt")
    test_decoded = tokenizer.decode(test_encoded["input_ids"][0], skip_special_tokens=False)
    print(f"\n🔍 Token verification:")
    print(f"  Original target: {test_target}")
    print(f"  Token IDs: {test_encoded['input_ids'][0].tolist()}")
    print(f"  Decoded back: {test_decoded}")
    print(f"  Match: {'✓' if test_target in test_decoded else '✗'}")
    
    train_data = T5Dataset(train_df, tokenizer)
    val_data = T5Dataset(val_df, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=str(settings.MODEL_DIR / "t5_rvq_test"),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-4,
        logging_steps=10,
        evaluation_strategy="no",
        save_strategy="no",
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    
    trainer.train()
    print("✓ Training complete")
    
    # STEP 8: Quick test
    print("\nSTEP 8: Testing predictions...")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print("\n⚠️  Note: With only 100 tracks and 1 epoch, predictions may be poor.")
    print("This is just a quick test to verify the pipeline works.\n")
    
    for _, row in test_df.head(3).iterrows():
        inputs = tokenizer(row["prompt"], return_tensors="pt", max_length=64, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=16,  # Increase to allow full semantic ID
                num_beams=1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode with and without special tokens for debugging
        predicted_full = tokenizer.decode(outputs[0], skip_special_tokens=False)
        predicted_clean = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_ids = "".join(re.findall(r'<Q\d{3}>', predicted_full))
        
        print(f"\nPrompt: {row['prompt']}")
        print(f"Target: {row['target']}")
        print(f"Raw output: {predicted_full}")
        print(f"Clean output: {predicted_clean}")
        print(f"Extracted IDs: {predicted_ids}")
        print(f"Match: {'✓' if predicted_ids == row['target'] else '✗'}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
