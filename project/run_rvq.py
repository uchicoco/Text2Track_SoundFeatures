"""
Main script using RVQ (Residual Vector Quantization) approach from the notebook.

This script follows the notebook's approach:
1. Load and merge tag data
2. Extract audio features
3. Filter tracks with 2+ tags
4. Apply PCA
5. Use RVQ for semantic ID generation
6. Train T5 model with prompt-target pairs
"""

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

import warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

# RVQ Configuration (from notebook's best config)
RVQ_CONFIG = {
    "pca_components": 36,  # PCA dimensions
    "L": 2,                # Number of RVQ levels (tokens per track)
    "K": 64,               # Clusters per level
    "n_init": 16,          # K-means initializations
    "max_iter": 200,       # K-means max iterations
}

# Template prompts (from notebook)
TEMPLATES = {
    ("genre", "mood"): [
        "a {mood} {genre} track",
        "{mood} {genre} music",
        "{genre} style with {mood} vibe",
        "looking for {mood} {genre}",
    ],
    ("genre", "instrument"): [
        "{genre} with {instrument}",
        "{genre} music featuring {instrument}",
        "a {genre} track focused on {instrument}",
        "{genre} with prominent {instrument}",
    ],
    ("mood", "instrument"): [
        "a {mood} track with {instrument}",
        "{mood} music featuring {instrument}",
        "{mood} vibe {instrument}",
        "looking for {mood} with {instrument}",
    ],
    ("genre", "mood", "instrument"): [
        "a {mood} {genre} track with {instrument}",
        "{mood} {genre} featuring {instrument}",
        "{genre} music, {mood} mood, {instrument} lead",
        "looking for {mood} {genre} with {instrument}",
    ],
}


def create_prompt_target_pairs(df):
    """
    Create (prompt, target) pairs from dataframe with tags and semantic IDs.
    Follows the notebook's template approach.
    """
    def row_tags(r):
        """Extract non-empty tags from a row."""
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
        
        # Create prompts for 2-tag and 3-tag combinations
        for k in (2, 3):
            if len(vals) >= k:
                for combo_keys in combinations(sorted(vals.keys()), k):
                    tmpls = TEMPLATES.get(
                        combo_keys,
                        ["; ".join([f"{key}={{{key}}}" for key in combo_keys])]
                    )
                    for tpl in tmpls:
                        records.append({
                            "track_id": r["track_id"],
                            "prompt": tpl.format(**{key: vals[key] for key in combo_keys}),
                            "target": r["semantic_id"],
                        })
    
    pairs_df = pd.DataFrame.from_records(records, columns=["track_id", "prompt", "target"])
    
    # Normalize semantic IDs
    pairs_df["target"] = pairs_df["target"].map(
        lambda s: re.sub(r'<(\d{3})>', r'<Q\1>', str(s))
    )
    
    return pairs_df


def main():
    """Main execution function"""
    print("=" * 70)
    print("RVQ-based Semantic ID Generation and T5 Training")
    print("(Following notebook configuration)")
    print("=" * 70)
    
    # ====================================================================
    # STEP 1: Load and merge tag data
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Loading tag data...")
    print("=" * 70)
    
    dp = DatasetProcessor()
    mood_df = dp.load_tag_tsv("autotagging_moodtheme.tsv", "mood")
    genre_df = dp.load_tag_tsv("autotagging_genre.tsv", "genre")
    inst_df = dp.load_tag_tsv("autotagging_instrument.tsv", "instrument")
    
    # Merge all tags
    tags_merged = (mood_df
                   .merge(genre_df, on=["track_id", "path"], how="outer")
                   .merge(inst_df, on=["track_id", "path"], how="outer"))
    tags_merged[["genre", "mood", "instrument"]] = tags_merged[["genre", "mood", "instrument"]].fillna("")
    
    print(f"✓ Loaded {len(tags_merged)} tracks with tags")
    
    # ====================================================================
    # STEP 2: Extract features
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Extracting audio features...")
    print("=" * 70)
    
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
    
    print(f"✓ Extracted features for {len(df)} tracks")
    
    # ====================================================================
    # STEP 3: Filter to tracks with 2+ tags
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Filtering to tracks with 2+ tags...")
    print("=" * 70)
    
    df_two = fe.filter_min2tags(df)
    print(f"✓ Kept {len(df_two)} tracks with 2+ tags")
    
    # Save intermediate result
    output_dir = settings.OUTPUT_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_two.to_csv(output_dir / "features_min2tags_rvq.csv", index=False)
    
    # ====================================================================
    # STEP 4: PCA dimensionality reduction
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Applying PCA...")
    print("=" * 70)
    
    pp = PCAProcessor()
    feat_matrix = pp.build_feature_matrix(df_two)
    
    n_components = RVQ_CONFIG["pca_components"]
    feat_pca, cum_var, expl_var, pca = pp.run_pca_components(
        feat_matrix,
        n_components=n_components
    )
    
    print(f"✓ Reduced to {n_components} dimensions")
    print(f"  Explained variance: {cum_var[-1]:.3f}")
    
    # ====================================================================
    # STEP 5: RVQ Semantic ID Generation
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Generating Semantic IDs with RVQ...")
    print("=" * 70)
    
    kmp = KMeansProcessor()
    codebooks, codes, L, K = kmp.run_rvq(
        feat_pca,
        n_tokens=RVQ_CONFIG["L"],
        n_clusters=RVQ_CONFIG["K"],
        max_iter=RVQ_CONFIG["max_iter"],
        n_init=RVQ_CONFIG["n_init"]
    )
    
    # Convert codes to semantic IDs
    def codes_to_semantic_ids(codes):
        """Convert integer codes to semantic ID strings."""
        semantic_ids = []
        for row in codes:
            semantic_id = "".join([f"<Q{int(c):03d}>" for c in row])
            semantic_ids.append(semantic_id)
        return semantic_ids
    
    semantic_ids = codes_to_semantic_ids(codes)
    
    # Attach to dataframe
    df_two = df_two.copy()
    for l in range(RVQ_CONFIG["L"]):
        df_two[f"rvq_code_{l}"] = codes[:, l]
    df_two["semantic_id"] = semantic_ids
    
    # Statistics
    vc = pd.Series(semantic_ids).value_counts()
    print(f"\n✓ Generated semantic IDs:")
    print(f"  Unique IDs: {vc.size}")
    print(f"  Mean tracks per ID: {vc.mean():.2f}")
    print(f"  Median tracks per ID: {vc.median():.0f}")
    print(f"  Singleton IDs: {(vc == 1).sum()} ({(vc == 1).mean():.1%})")
    
    # Save results
    df_two.to_csv(output_dir / "dataset_with_semantic_ids_rvq.csv", index=False)
    np.save(output_dir / f"rvq_codes_L{L}_K{K}.npy", codes)
    
    # Save codebooks
    for level, km in enumerate(codebooks):
        import joblib
        path = output_dir / f"rvq_level{level}_K{K}.joblib"
        joblib.dump(km, path)
    print(f"Saved {L} RVQ codebooks to {output_dir}")
    
    # ====================================================================
    # STEP 6: Create prompt-target pairs
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Creating prompt-target pairs...")
    print("=" * 70)
    
    pairs_df = create_prompt_target_pairs(df_two)
    
    used_tracks = pairs_df["track_id"].nunique()
    avg_per_track = len(pairs_df) / used_tracks
    print(f"✓ Created {len(pairs_df)} pairs from {used_tracks} tracks")
    print(f"  Average pairs per track: {avg_per_track:.2f}")
    print(f"  Unique targets: {pairs_df['target'].nunique()}")
    
    # ====================================================================
    # STEP 7: Split and prepare T5 training data
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Splitting data for T5 training...")
    print("=" * 70)
    
    t5p = T5Processor()
    
    # Shuffle
    pairs_df = pairs_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Split by track IDs (80/10/10)
    track_ids = pairs_df["track_id"].astype(str).unique()
    rng = np.random.default_rng(42)
    track_ids = rng.permutation(track_ids)
    n = len(track_ids)
    
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    split_map = {
        tid: ("train" if i < n_train
              else "validation" if i < n_train + n_val
              else "test")
        for i, tid in enumerate(track_ids)
    }
    pairs_df["split"] = pairs_df["track_id"].astype(str).map(split_map)
    
    # Create split dataframes
    train_df = pairs_df[pairs_df["split"] == "train"].copy()
    val_df = pairs_df[pairs_df["split"] == "validation"].copy()
    test_df = pairs_df[pairs_df["split"] == "test"].copy()
    
    print(f"✓ Split complete:")
    print(f"  Train: {len(train_df)} pairs")
    print(f"  Validation: {len(val_df)} pairs")
    print(f"  Test: {len(test_df)} pairs")
    
    # Apply upsampling to training set (like notebook: TARGET_MIN=20, REPEAT_CAP=6)
    TARGET_MIN = 20
    REPEAT_CAP = 6
    
    freq = train_df["target"].value_counts()
    repeats = np.clip(np.ceil(TARGET_MIN / freq).astype(int), 1, REPEAT_CAP)
    rep_map = repeats.to_dict()
    
    train_df_balanced = train_df.loc[
        np.repeat(train_df.index.values, train_df["target"].map(rep_map).values)
    ].sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"\n✓ Applied upsampling:")
    print(f"  Train size: {len(train_df)} -> {len(train_df_balanced)}")
    print(f"  Median frequency: {int(freq.median())} -> "
          f"{int(train_df_balanced['target'].value_counts().median())}")
    
    # Save splits
    train_df_balanced.to_csv(output_dir / "train_rvq.csv", index=False)
    val_df.to_csv(output_dir / "val_rvq.csv", index=False)
    test_df.to_csv(output_dir / "test_rvq.csv", index=False)
    
    # ====================================================================
    # STEP 8: Train T5 Model
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Training T5 Model...")
    print("=" * 70)
    
    from transformers import TrainingArguments, Trainer
    from src.t5.t5_utils import T5Dataset
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import multiprocessing
    import torch
    
    # Use smaller model for faster training (like notebook: flan-t5-small)
    model_name = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = [f"<Q{i:03d}>" for i in range(256)]
    added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    print(f"✓ Loaded model: {model_name}")
    print(f"  Added {added} special tokens")
    
    # Create datasets
    train_data = T5Dataset(train_df_balanced, tokenizer)
    val_data = T5Dataset(val_df, tokenizer)
    
    # Training arguments (following notebook config)
    training_args = TrainingArguments(
        output_dir=str(settings.MODEL_DIR / "t5_rvq_model"),
        num_train_epochs=6,  # Notebook uses 6 epochs
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch size = 32
        learning_rate=2e-4,  # Notebook uses 2e-4
        weight_decay=0.01,
        label_smoothing_factor=0.1,  # From notebook
        warmup_steps=100,
        logging_dir=str(settings.PROJECT_ROOT / "logs"),
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        dataloader_num_workers=min(4, multiprocessing.cpu_count() // 2),
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save model
    model_path = settings.MODEL_DIR / "t5_rvq_model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    print(f"\n✓ Model saved to: {model_path}")
    
    # ====================================================================
    # STEP 9: Evaluate
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 9: Evaluating model...")
    print("=" * 70)
    
    print("\nEvaluating on validation set...")
    metrics = trainer.evaluate(eval_dataset=val_data)
    print(f"Validation loss: {metrics.get('eval_loss', 'N/A')}")
    
    # Quick generation test
    print("\nTesting generation on sample prompts...")
    test_samples = test_df.sample(min(5, len(test_df)), random_state=42)
    
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    for _, row in test_samples.iterrows():
        inputs = tokenizer(
            row["prompt"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                num_beams=1,
                do_sample=False
            )
        
        predicted = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract only <Q###> tokens
        predicted_ids = "".join(re.findall(r'<Q\d{3}>', predicted))
        
        print(f"\nPrompt: {row['prompt']}")
        print(f"Target: {row['target']}")
        print(f"Predicted: {predicted_ids}")
        print(f"Match: {'✓' if predicted_ids == row['target'] else '✗'}")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"Model saved to: {model_path}")
    print("\nKey differences from original run_new.py:")
    print("- Uses RVQ (Residual Vector Quantization) instead of K-means/Dictionary Learning")
    print("- Uses template-based prompts following notebook approach")
    print("- Uses flan-t5-small with notebook's hyperparameters")
    print("- Upsampling with TARGET_MIN=20, REPEAT_CAP=6")


if __name__ == "__main__":
    main()
