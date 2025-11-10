"""
Main execution script for genre+instrument classification (no mood).

This script uses 28 selected features for optimal discrimination:
- Genre and Instrument tags only
- Reduced feature set (28 features)
- Varied prompt templates to avoid syntax dependency
"""

import warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

from pathlib import Path
import pandas as pd

from src.data import DatasetProcessor
from src.data.feature_extractor_nomood import FeatureExtractorNoMood
from src.models import PCAProcessor
from src.semantic import SemanticIDGenerator
from src.t5.t5_processor_nomood import T5ProcessorNoMood


def main():
    """Main execution function for genre+instrument only pipeline"""
    
    # ========================================================================
    # STEP 1: Load dataset (genre and instrument tags only)
    # ========================================================================
    print("=" * 60)
    print("STEP 1: Loading dataset (genre + instrument)")
    print("=" * 60)
    
    dp = DatasetProcessor()
    genre_df = dp.load_tag_tsv("autotagging_genre.tsv", "genre")
    inst_df = dp.load_tag_tsv("autotagging_instrument.tsv", "instrument")
    print("✓ Dataset loaded successfully")
    
    # Merge genre and instrument tags
    tags_merged = genre_df.merge(inst_df, on=["track_id", "path"], how="inner")
    print(f"\nTracks with both genre and instrument tags: {len(tags_merged)}")
    print(f"\nSample data:\n{tags_merged.head()}\n")
    
    # ========================================================================
    # STEP 2: Extract 28 selected features
    # ========================================================================
    print("=" * 60)
    print("STEP 2: Extracting 28 selected features")
    print("=" * 60)
    print("\nFeature set:")
    print("  - GFCC (6): mean_0/1/3/5/7/11")
    print("  - Spectral Shape (8): flux, spread, skewness, kurtosis, RMS, ERB spread")
    print("  - Spectral Valley (4): valley_mean_0/3/4/5")
    print("  - ERB Bands (2): flatness, crest")
    print("  - HPCP/Tonal (2): dissonance, entropy")
    print("  - Other (6): loudness, danceability, centroid, complexity_var, entropy, tuning")
    
    fe = FeatureExtractorNoMood()
    df = fe.build_features_dataframe(tags_merged)
    
    # Filter to ensure both tags are present
    df_filtered = fe.filter_common_tags(df)
    fe.check_tag_distribution(df_filtered)
    
    # Save features
    out_csv = Path(fe.dp.output_dir) / "data/features_nomood.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(out_csv, index=False)
    print(f"\n✓ Features saved to: {out_csv}")
    print(f"\nSample features:\n{df_filtered.head()}\n")
    
    # ========================================================================
    # STEP 3: PCA dimensionality reduction
    # ========================================================================
    print("=" * 60)
    print("STEP 3: Running PCA")
    print("=" * 60)
    print("Configuration:")
    print("  - Input features: 28")
    print("  - Target components: ~20-24 (85-90% variance)")
    
    pp = PCAProcessor()
    feat_matrix = pp.build_feature_matrix(df_filtered)
    
    # Run PCA with variance ratio (recommended)
    feat_pca, cum_var, expl_var, pca = pp.run_pca_ratio(
        feat_matrix, 
        r_explained_var=0.85  # 85% variance
    )
    
    n_components = feat_pca.shape[1]
    print(f"\n✓ PCA completed: {n_components} components (85% variance)")
    
    pp.save_feat_pca(feat_pca, expl_var)
    print("✓ PCA results saved")
    
    # Optional: Visualize PCA results
    # Uncomment to generate visualizations
    # print("\nGenerating PCA visualizations...")
    # pp.plot_pca_with_tags(feat_pca, df_filtered)
    # pp.plot_explained_variance(expl_var, cum_var)
    
    # ========================================================================
    # STEP 4: Generate Semantic IDs
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Generating Semantic IDs")
    print("=" * 60)
    print("Configuration (optimized for reduced dimensions):")
    
    sig = SemanticIDGenerator()
    
    # === Method 1: Dictionary Learning (Recommended) ===
    print("  - Method: Dictionary Learning")
    print(f"  - PCA components: {n_components}")
    print("  - Dict components: 12-16 (adjusted for reduced dimensions)")
    print("  - Non-zero coefs: 2")
    
    semantic_ids = sig.assign_sem_ids_dl_manual(
        feat_pca,
        n_nonzero_coefs=2,
        n_dict_components=min(16, n_components),  # Adapt to PCA dims
        max_iter=200,
        batch_size=256
    )
    
    # === Alternative: K-means ===
    # Uncomment to use K-means instead
    # print("  - Method: K-means")
    # print(f"  - PCA components: {n_components}")
    # print("  - Clusters: 24-32")
    # print("  - Tokens: 2")
    # 
    # semantic_ids = sig.assign_sem_ids_kmean_manual(
    #     feat_pca,
    #     n_tokens=2,
    #     n_clusters=24,
    #     max_iter=200,
    #     n_init=16
    # )
    
    # === Alternative: Grid Search ===
    # Uncomment to search for best parameters
    # semantic_ids = sig.search_best_dl(
    #     feat_pca,
    #     n_nonzero_coefs_list=[2, 3],
    #     n_dict_components_list=[12, 16, 20],
    #     batch_size_list=[256],
    #     max_iter=200,
    #     ideal_num_per_id=3
    # )
    
    print("✓ Semantic IDs generated")
    
    # Show statistics
    sig.print_stats(semantic_ids)
    
    # Integrate with dataset
    df_ids = sig.integrate_with_dataset(df_filtered, semantic_ids)
    print(f"\nSample semantic IDs:\n{df_ids[['track_id', 'genre', 'instrument', 'semantic_id']].head()}\n")
    
    # ========================================================================
    # STEP 5: Train T5 Model
    # ========================================================================
    print("=" * 60)
    print("STEP 5: Training T5 Model (Genre + Instrument)")
    print("=" * 60)
    print("Prompt strategy:")
    print("  - Training: Varied templates (10 patterns, randomly selected)")
    print("  - Val/Test: Fixed template for consistency")
    print("  - Templates:")
    print("    * '{genre} with {instrument}'")
    print("    * '{instrument} playing {genre}'")
    print("    * 'recommend a {genre} track with {instrument}'")
    print("    * '{genre} music featuring {instrument}'")
    print("    * 'a {instrument}-based {genre} song'")
    print("    * ... and 5 more variations")
    
    t5p = T5ProcessorNoMood()
    
    train_df, val_df, test_df = t5p.prepare_data(df_ids=df_ids)
    print(f"\nDataset split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print(f"\nSample prompt-target pairs:")
    print(train_df.head())
    
    print("\nStarting T5 training...")
    tokenizer, model = t5p.train_t5(
        train_df, 
        val_df, 
        output_dir="models/t5_nomood_model"
    )
    print("✓ T5 training completed")
    
    # ========================================================================
    # STEP 6: Evaluate T5 Model
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Evaluating T5 Model")
    print("=" * 60)
    
    hits_at_10 = t5p.evaluate_t5(
        tokenizer=tokenizer, 
        model=model, 
        test_df=test_df,
        k=10
    )
    
    print(f"\n{'='*60}")
    print(f"Final Result: Hits@10 = {hits_at_10:.4f}")
    print(f"{'='*60}\n")
    
    # ========================================================================
    # STEP 7: Visualize Results
    # ========================================================================
    print("=" * 60)
    print("STEP 7: Generating Visualizations")
    print("=" * 60)
    
    print("Generating UMAP visualization...")
    t5p.visualize_umap(
        tokenizer=tokenizer, 
        model=model, 
        test_df=test_df,
        output_path="outputs/figures/umap_nomood.png"
    )
    print("✓ UMAP visualization saved")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nConfiguration summary:")
    print(f"  - Input features: 28 selected features")
    print(f"  - PCA components: {n_components} ({cum_var[-1]:.1%} variance)")
    print(f"  - Training samples: {len(train_df)}")
    print(f"  - Unique semantic IDs: {len(set(semantic_ids))}")
    print(f"  - Hits@10: {hits_at_10:.4f}")
    
    print(f"\nOutputs saved to: {dp.output_dir}")
    print("  - data/features_nomood.csv: Extracted features")
    print("  - data/pca/: PCA transformed features")
    print("  - figures/umap_nomood.png: Semantic ID visualization")
    print("  - models/t5_nomood_model/: Trained T5 model")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
