"""
Main execution script for the Music Informatics pipeline.

This script orchestrates the entire workflow:
1. Data loading and preprocessing
2. Feature extraction
3. PCA dimensionality reduction
4. Semantic ID generation (K-means or Dictionary Learning)
5. T5 model training and evaluation
"""

import warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

from pathlib import Path

import pandas as pd

from src.data import DatasetProcessor, FeatureExtractor
from src.models import PCAProcessor
from src.semantic import SemanticIDGenerator
from src.t5 import T5Processor


def main():
    """Main execution function"""
    # Load and prepare data
    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)
    dp = DatasetProcessor()
    mood_df = dp.load_tag_tsv("autotagging_moodtheme.tsv", "mood")
    genre_df = dp.load_tag_tsv("autotagging_genre.tsv", "genre")
    inst_df = dp.load_tag_tsv("autotagging_instrument.tsv", "instrument")
    print("✓ Dataset loaded successfully")
    print(f"\nSample mood data:\n{mood_df.head()}\n")

    # Extract features and save df to csv
    print("=" * 60)
    print("STEP 2: Extracting features")
    print("=" * 60)
    fe = FeatureExtractor()

    tags_merged = (mood_df
                   .merge(genre_df, on=["track_id", "path"], how="outer")
                   .merge(inst_df, on=["track_id", "path"], how="outer"))

    tags_merged[["genre", "mood", "instrument"]] = tags_merged[["genre", "mood", "instrument"]].fillna("")

    # Configure feature extraction
    # You can change these flags to extract different features
    df = fe.build_features_dataframe(
        tags_merged,
        use_mel_alt=False,
        use_energy_alt=False,
        add_valley=True,
        add_timbre_dist=True,
        add_tonality=True,
        add_rhythm_struct=True
    )

    df_two = fe.filter_min2tags(df)
    fe.check_tag_distribution(df_two)

    out_csv_two = Path(fe.dp.output_dir) / "data/features_2tags.csv"
    out_csv_two.parent.mkdir(parents=True, exist_ok=True)
    df_two.to_csv(out_csv_two, index=False)
    print(f"✓ Features saved to: {out_csv_two}")
    print(f"\nSample features:\n{df_two.head()}\n")

    # Run PCA
    print("=" * 60)
    print("STEP 3: Running PCA")
    print("=" * 60)
    pp = PCAProcessor()
    feat_matrix = pp.build_feature_matrix(df_two)

    # Choose PCA method:
    # Option 1: Fixed number of components
    # feat_pca, cum_var, expl_var, pca = pp.run_pca_components(feat_matrix, n_components=32)
    
    # Option 2: Explained variance ratio (recommended)
    feat_pca, cum_var, expl_var, pca = pp.run_pca_ratio(feat_matrix, r_explained_var=0.80)

    pp.save_feat_pca(feat_pca, expl_var)
    print("✓ PCA completed and saved")

    # Optional: Visualize PCA results
    # Uncomment the following lines to generate visualizations
    # print("\nGenerating PCA visualizations...")
    # pp.plot_pca_with_tags(feat_pca, df_two)
    # pp.plot_explained_variance(expl_var, cum_var)

    # Generate Semantic IDs
    print("\n" + "=" * 60)
    print("STEP 4: Generating Semantic IDs")
    print("=" * 60)
    sig = SemanticIDGenerator()

    # Choose semantic ID generation method:
    
    # === Method 1: K-means with manual configuration ===
    # semantic_ids = sig.assign_sem_ids_kmean_manual(
    #     feat_pca,
    #     n_tokens=2,
    #     n_clusters=32,
    #     max_iter=200,
    #     n_init=16
    # )

    # === Method 2: K-means with grid search (recommended for first run) ===
    # semantic_ids = sig.search_best_kmean(
    #     feat_pca,
    #     n_tokens_list=[2, 3],
    #     n_clusters_list=[32, 64, 128],
    #     max_iter=200,
    #     n_init_values=[16],
    #     ideal_num_per_id=3
    # )

    # === Method 3: Dictionary Learning with manual configuration ===
    semantic_ids = sig.assign_sem_ids_dl_manual(
        feat_pca,
        n_nonzero_coefs=2,
        n_dict_components=16,
        max_iter=200,
        batch_size=256
    )

    # === Method 4: Dictionary Learning with grid search ===
    # semantic_ids = sig.search_best_dl(
    #     feat_pca,
    #     n_nonzero_coefs_list=[2, 3],
    #     n_dict_components_list=[32, 64, 128],
    #     batch_size_list=[256],
    #     max_iter=200,
    #     ideal_num_per_id=3
    # )

    # === Method 5: Load previously saved configuration ===
    # For K-means:
    # semantic_ids = sig.assign_sem_ids_kmean_load(feat_pca)
    # For Dictionary Learning:
    # semantic_ids = sig.assign_sem_ids_dl_load(feat_pca)

    print("✓ Semantic IDs generated")

    # Show semantic ID statistics (optional)
    # sig.print_stats(semantic_ids)

    # Integrate semantic IDs into the dataframe
    df_ids = sig.integrate_with_dataset(df_two, semantic_ids)
    print(f"\nSample semantic IDs:\n{df_ids[['track_id', 'genre', 'mood', 'instrument', 'semantic_id']].head()}\n")

    # Train T5 model
    print("=" * 60)
    print("STEP 5: Training T5 Model")
    print("=" * 60)
    t5p = T5Processor()
    
    train_df, val_df, test_df = t5p.prepare_data(df_ids=df_ids)
    print(f"\nDataset split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print(f"\nSample prompt-target pairs:")
    print(train_df.head())

    print("\nStarting T5 training...")
    tokenizer, model = t5p.train_t5(train_df, val_df)
    print("✓ T5 training completed")

    # Evaluate T5 model
    print("\n" + "=" * 60)
    print("STEP 6: Evaluating T5 Model")
    print("=" * 60)
    hits_at_10 = t5p.evaluate_t5(tokenizer=tokenizer, model=model, test_df=test_df)
    print(f"\n{'='*60}")
    print(f"Final Result: Hits@10 = {hits_at_10:.4f}")
    print(f"{'='*60}\n")

    # Visualize embeddings with UMAP
    print("Generating UMAP visualization...")
    t5p.visualize_umap(tokenizer=tokenizer, model=model, test_df=test_df)
    print("✓ UMAP visualization saved")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nOutputs saved to: {dp.output_dir}")
    print("  - data/: Processed datasets and PCA features")
    print("  - figures/: Visualizations")
    print("  - models/: Trained T5 model checkpoints")


if __name__ == "__main__":
    main()
