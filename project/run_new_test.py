"""
Lightweight test script for quick validation (NEW MODULAR STRUCTURE).

This script runs a minimal version of the pipeline for testing purposes:
- Uses only first 100 tracks
- Reduces PCA components
- Uses smaller K-means parameters
- Trains T5 for fewer epochs
- Skips visualizations
"""

import warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

from pathlib import Path

import pandas as pd

from config import settings
from src.data.dataset_processor import DatasetProcessor
from src.data.feature_extractor import FeatureExtractor
from src.models.pca_processor import PCAProcessor
from src.semantic.semantic_id_generator import SemanticIDGenerator
from src.t5.t5_processor import T5Processor


def main():
    """Main test execution function"""
    print("=" * 60)
    print("TEST MODE: Running lightweight validation (NEW STRUCTURE)")
    print("=" * 60)
    
    # STEP 1: Load dataset (limited)
    print("\nSTEP 1: Loading dataset (first 100 tracks)...")
    dp = DatasetProcessor()
    mood_df = dp.load_tag_tsv("autotagging_moodtheme.tsv", "mood")
    genre_df = dp.load_tag_tsv("autotagging_genre.tsv", "genre")
    inst_df = dp.load_tag_tsv("autotagging_instrument.tsv", "instrument")
    
    # Limit to first 100 tracks for testing
    mood_df = mood_df.head(100)
    genre_df = genre_df.head(100)
    inst_df = inst_df.head(100)
    
    print(f"✓ Loaded {len(mood_df)} tracks")

    # STEP 2: Extract features (default settings only)
    print("\nSTEP 2: Extracting features (default settings)...")
    fe = FeatureExtractor()

    tags_merged = (mood_df
                   .merge(genre_df, on=["track_id", "path"], how="outer")
                   .merge(inst_df, on=["track_id", "path"], how="outer"))

    tags_merged[["genre", "mood", "instrument"]] = tags_merged[["genre", "mood", "instrument"]].fillna("")

    # Use default feature extraction only (fastest)
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
    print(f"✓ Extracted features for {len(df_two)} tracks with 2+ tags")

    # STEP 3: Run PCA (fewer components)
    print("\nSTEP 3: Running PCA (10 components)...")
    pp = PCAProcessor()
    feat_matrix = pp.build_feature_matrix(df_two)
    
    # Use fixed small number of components for speed
    feat_pca, cum_var, expl_var, pca = pp.run_pca_components(feat_matrix, n_components=10)
    print(f"✓ Reduced to {feat_pca.shape[1]} dimensions")

    # STEP 4: Generate Semantic IDs (smallest config)
    print("\nSTEP 4: Generating Semantic IDs (minimal config)...")
    sig = SemanticIDGenerator()
    
    # Use smallest K-means configuration for speed
    semantic_ids = sig.assign_sem_ids_kmean_manual(
        feat_pca,
        n_tokens=2,
        n_clusters=8,  # Very small for testing
        max_iter=50,   # Fewer iterations
        n_init=4       # Fewer initializations
    )
    print(f"✓ Generated {len(set(semantic_ids))} unique semantic IDs")

    df_ids = sig.integrate_with_dataset(df_two, semantic_ids)

    # STEP 5: Train T5 (minimal training)
    print("\nSTEP 5: Training T5 Model (minimal epochs)...")
    t5p = T5Processor()
    
    # Prepare data with minimal upsampling
    train_df, val_df, test_df = t5p.prepare_data(
        df_ids=df_ids,
        upsample_threshold=5,  # Lower threshold
        max_replication=2      # Less replication
    )
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Train with minimal epochs
    print("\nTraining T5 (1 epoch only)...")
    from transformers import TrainingArguments, Trainer
    from src.t5.t5_utils import T5Dataset
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import multiprocessing
    
    tokenizer = T5Tokenizer.from_pretrained("t5-small")  # Use smaller model
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # Create datasets
    train_data = T5Dataset(train_df, tokenizer)
    val_data = T5Dataset(val_df, tokenizer)
    
    # Minimal training arguments
    training_args = TrainingArguments(
        output_dir=str(t5p.model_path),
        num_train_epochs=1,  # Just 1 epoch
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=10,
        learning_rate=5e-5,
        logging_dir=str(settings.PROJECT_ROOT / "logs"),
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        dataloader_num_workers=min(2, multiprocessing.cpu_count()//4),  # Minimal workers
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    trainer.train()
    print("✓ T5 training completed")

    # STEP 6: Quick evaluation (small sample)
    print("\nSTEP 6: Evaluating (first 10 samples only)...")
    test_df_small = test_df.head(10)  # Only evaluate 10 samples
    
    model.eval()
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    test_dataset = T5Dataset(test_df_small, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    hits = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    for batch in test_loader:
        true_targets = batch['target']
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            num_beams=10,
            num_return_sequences=10,
            max_length=64
        )
        
        for i in range(len(true_targets)):
            generated_targets = [tokenizer.decode(outputs[j], skip_special_tokens=True) 
                                 for j in range(i*10, (i+1)*10)]
            if true_targets[i] in generated_targets:
                hits += 1
    
    hits_at_10 = hits / len(test_df_small)
    print(f"\n{'='*60}")
    print(f"Test Result (10 samples): Hits@10 = {hits_at_10:.4f}")
    print(f"{'='*60}\n")

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNote: This is a minimal test run.")
    print("For full training, use run_new.py with complete dataset.")


if __name__ == "__main__":
    main()
