"""
Feature Discrimination Analysis for Tag Classification

This script analyzes which acoustic features are most effective for 
discriminating between different tags within each category (genre, mood, instrument).
Uses ANOVA and effect size to identify discriminative features.

Usage:
    python analyze_corr.py [options]
    
Options:
    --f               Extract all features and save to features_2tags_all.csv
    --name NAME       Custom name for saved features CSV (default: features_2tags_all.csv)
    --top-n N         Number of top tags to analyze (default: 10)
    --no-normalize    Disable feature normalization

Examples:
    python analyze_corr.py                    # Use cached CSV
    python analyze_corr.py --f                # Extract all features
    python analyze_corr.py --f --name custom  # Extract and save as features_2tags_custom.csv

Output:
    - Feature discrimination rankings (CSV)
    - 1D visualizations: violin plots for top features by tag (PNG)
    - 2D visualizations: scatter plots for feature pairs by tag (PNG)
    - Statistical summary (TXT)
"""

from pathlib import Path
import sys
import argparse

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from umap import UMAP

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset_processor import DatasetProcessor
from src.data.feature_extractor import FeatureExtractor
from src.data.feature_extractor_additional import FeatureExtractorAdditional


class FeatureDiscriminationAnalyzer:
    """Analyze which features discriminate between tags within each category"""
    
    def __init__(self, output_dir=None, top_n_tags=10):
        """
        Initialize the analyzer
        
        Args:
            output_dir: Output directory path (default: outputs/discrimination_analysis)
            top_n_tags: Number of top tags to analyze per category (default: 10)
        """
        self.dp = DatasetProcessor()
        self.fe = FeatureExtractor()
        
        if output_dir is None:
            self.output_dir = Path(self.dp.output_dir) / "discrimination_analysis"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.top_n_tags = top_n_tags
        
        # Features to exclude from analysis
        self.exclude_cols = ["json_full", "track_id", "path", "genre", "mood", "instrument"]
        
        # Set style for plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
    
    def load_data(self, csv_name='features_all.csv', extract_all=False, extract_comprehensive=False):
        """
        Load and prepare dataset with features and tags
        
        Args:
            csv_name: Name of CSV file to load/save (default: features_all.csv)
            extract_all: If True, extract ALL features (including alternatives) - original method
            extract_comprehensive: If True, extract ALL AcousticBrainz features (comprehensive analysis)
        
        Returns:
            df: DataFrame with features and tags
        """
        print("Loading dataset...")
        
        # Determine CSV path
        csv_path = Path(self.dp.output_dir) / f"data/{csv_name}"
        
        if csv_path.exists() and not extract_all and not extract_comprehensive:
            print(f"Loading from existing CSV: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            if extract_comprehensive:
                print(f"Extracting COMPREHENSIVE features (ALL AcousticBrainz features)...")
            elif extract_all:
                print(f"Extracting ALL features...")
            else:
                print("Building dataset from scratch...")
            
            # Load tags
            mood_df = self.dp.load_tag_tsv("autotagging_moodtheme.tsv", "mood")
            genre_df = self.dp.load_tag_tsv("autotagging_genre.tsv", "genre")
            inst_df = self.dp.load_tag_tsv("autotagging_instrument.tsv", "instrument")
            
            # Merge tags
            tags_merged = (mood_df
                          .merge(genre_df, on=["track_id", "path"], how="outer")
                          .merge(inst_df, on=["track_id", "path"], how="outer"))
            tags_merged[["genre", "mood", "instrument"]] = tags_merged[["genre", "mood", "instrument"]].fillna("")
            
            if extract_comprehensive:
                # Use FeatureExtractorAdditional for comprehensive analysis
                fe_add = FeatureExtractorAdditional()
                print("\nExtracting COMPREHENSIVE feature set (MFCC-based)...")
                df = fe_add.build_features_dataframe(
                    tags_merged,
                    use_mel_alt=False,
                    use_energy_alt=False,
                    extract_all=True  # Extract ALL features
                )
                print(f"Total features extracted: {len([c for c in df.columns if c not in self.exclude_cols])}")
                
            elif extract_all:
                # Extract ALL features by combining different options
                print("\nExtracting feature set 1/4: MFCC + Spectral Centroid + Optional features...")
                df_base = self.fe.build_features_dataframe(
                    tags_merged,
                    use_mel_alt=False,
                    use_energy_alt=False,
                    add_valley=True,
                    add_timbre_dist=True,
                    add_tonality=True,
                    add_rhythm_struct=True
                )
                
                print("\nExtracting feature set 2/4: MEL + Spectral Centroid...")
                df_mel = self.fe.build_features_dataframe(
                    tags_merged,
                    use_mel_alt=True,
                    use_energy_alt=False,
                    add_valley=False,  # Already in base
                    add_timbre_dist=False,
                    add_tonality=False,
                    add_rhythm_struct=False
                )
                
                print("\nExtracting feature set 3/4: MFCC + Spectral Energy...")
                df_energy = self.fe.build_features_dataframe(
                    tags_merged,
                    use_mel_alt=False,
                    use_energy_alt=True,
                    add_valley=False,
                    add_timbre_dist=False,
                    add_tonality=False,
                    add_rhythm_struct=False
                )
                
                print("\nExtracting feature set 4/4: MEL + Spectral Energy...")
                df_mel_energy = self.fe.build_features_dataframe(
                    tags_merged,
                    use_mel_alt=True,
                    use_energy_alt=True,
                    add_valley=False,
                    add_timbre_dist=False,
                    add_tonality=False,
                    add_rhythm_struct=False
                )
                
                # Merge all feature sets
                print("\nMerging all feature sets...")
                # Get feature columns only
                base_features = [c for c in df_base.columns if c not in self.exclude_cols]
                mel_features = [c for c in df_mel.columns if c not in self.exclude_cols and c not in base_features]
                energy_features = [c for c in df_energy.columns if c not in self.exclude_cols and c not in base_features]
                mel_energy_features = [c for c in df_mel_energy.columns if c not in self.exclude_cols 
                                      and c not in base_features and c not in mel_features and c not in energy_features]
                
                # Combine
                df = df_base.copy()
                for feat in mel_features:
                    df[feat] = df_mel[feat]
                for feat in energy_features:
                    df[feat] = df_energy[feat]
                for feat in mel_energy_features:
                    df[feat] = df_mel_energy[feat]
                
                print(f"Total features extracted: {len([c for c in df.columns if c not in self.exclude_cols])}")
                
            else:
                # Standard extraction (backward compatible)
                df = self.fe.build_features_dataframe(
                    tags_merged,
                    use_mel_alt=False,
                    use_energy_alt=False,
                    add_valley=False,
                    add_timbre_dist=False,
                    add_tonality=False,
                    add_rhythm_struct=False
                )
            
            # Filter to tracks with at least 1 tag (not 2 tags like in run.py)
            # This allows analyzing more tracks
            df = df[(df['genre'] != '') | (df['mood'] != '') | (df['instrument'] != '')]
            print(f"Filtered to {len(df)} tracks with at least 1 tag")
            
            # Save for future use
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"Saved features to: {csv_path}")
        
        print(f"Loaded {len(df)} tracks")
        return df
    
    def prepare_features(self, df, normalize=True):
        """
        Prepare feature matrix
        
        Args:
            df: DataFrame with features and tags
            normalize: Whether to normalize features (default: True)
            
        Returns:
            feature_df: DataFrame with feature columns only
            feature_names: List of feature names
        """
        # Get feature columns
        feature_cols = [c for c in df.columns if c not in self.exclude_cols]
        feature_df = df[feature_cols].copy()
        
        # Handle NaN values (fill with 0 before normalization)
        nan_count = feature_df.isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values, filling with 0")
            feature_df = feature_df.fillna(0)
        
        # Normalize if requested
        if normalize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            feature_df[feature_cols] = scaler.fit_transform(feature_df[feature_cols])
            print(f"Features normalized (StandardScaler: mean=0, std=1)")
        
        print(f"Prepared {len(feature_cols)} features")
        return feature_df, feature_cols
    
    def get_top_tags(self, df, tag_column):
        """
        Get top N most frequent tags and filter data
        
        Args:
            df: DataFrame with tag column
            tag_column: Name of tag column ('genre', 'mood', or 'instrument')
            
        Returns:
            filtered_df: DataFrame with only top tags
            top_tags: List of top tag names
        """
        # Split multi-label tags and count
        tag_series = df[tag_column].str.split('---')
        all_tags = []
        for tags in tag_series:
            if isinstance(tags, list):
                all_tags.extend(tags)
        
        tag_counts = pd.Series(all_tags).value_counts()
        top_tags = tag_counts.head(self.top_n_tags).index.tolist()
        
        # Add "sad" to mood tags if not already included
        if tag_column == 'mood' and 'sad' not in top_tags:
            if 'sad' in tag_counts.index:
                top_tags.append('sad')
                print(f"Added 'sad' to mood tags (n={tag_counts['sad']})")
        
        print(f"Top {len(top_tags)} {tag_column} tags: {top_tags}")
        
        # Filter to tracks that have exactly one of the top tags (single-label for discrimination)
        def has_single_top_tag(tags_str):
            if pd.isna(tags_str) or tags_str == "":
                return None
            tags = tags_str.split('---')
            top_matches = [t for t in tags if t in top_tags]
            return top_matches[0] if len(top_matches) == 1 else None
        
        df_copy = df.copy()
        df_copy[f'{tag_column}_single'] = df_copy[tag_column].apply(has_single_top_tag)
        filtered_df = df_copy[df_copy[f'{tag_column}_single'].notna()].copy()
        
        print(f"Filtered to {len(filtered_df)} tracks with single top {tag_column} tag")
        
        return filtered_df, top_tags
    
    def calculate_feature_discrimination(self, feature_df, tag_labels):
        """
        Calculate ANOVA F-statistic and effect size for each feature
        
        Args:
            feature_df: DataFrame with feature values
            tag_labels: Series with tag labels
            
        Returns:
            discrimination_df: DataFrame with F-statistic, p-value, and effect size
        """
        print("Calculating feature discrimination power...")
        
        results = []
        unique_tags = tag_labels.unique()
        
        for feat_col in tqdm(feature_df.columns):
            # Group feature values by tag
            groups = [feature_df.loc[tag_labels == tag, feat_col].values 
                     for tag in unique_tags]
            
            # Remove empty groups
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) < 2:
                continue
            
            # ANOVA
            f_stat, p_val = f_oneway(*groups)
            
            # Effect size (eta-squared)
            # SS_between / SS_total
            all_values = feature_df[feat_col].values
            grand_mean = all_values.mean()
            
            ss_between = sum([len(g) * (g.mean() - grand_mean)**2 for g in groups])
            ss_total = ((all_values - grand_mean)**2).sum()
            
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            results.append({
                'feature': feat_col,
                'f_statistic': f_stat,
                'p_value': p_val,
                'eta_squared': eta_squared,
                'significant': p_val < 0.05
            })
        
        discrimination_df = pd.DataFrame(results)
        discrimination_df = discrimination_df.sort_values('f_statistic', ascending=False)
        
        return discrimination_df
    
    def plot_violin_1d(self, df, feature_df, tag_column, tag_labels, top_tags, n_features=6):
        """
        Plot violin plots for top discriminative features
        
        Args:
            df: Original DataFrame
            feature_df: DataFrame with feature values
            tag_column: Tag category name
            tag_labels: Series with tag labels
            top_tags: List of top tags
            n_features: Number of features to plot
        """
        print(f"Plotting violin plots for {tag_column}...")
        
        # Get discrimination results
        discrimination_df = self.calculate_feature_discrimination(feature_df, tag_labels)
        top_features = discrimination_df.head(n_features)['feature'].tolist()
        
        # Create subplots
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feat in enumerate(top_features):
            if idx >= len(axes):
                break
            
            # Prepare data
            plot_df = pd.DataFrame({
                'feature': feature_df[feat],
                'tag': tag_labels
            })
            
            # Plot
            sns.violinplot(data=plot_df, x='tag', y='feature', ax=axes[idx], palette='Set2')
            axes[idx].set_title(f'{feat}\n(F={discrimination_df.iloc[idx]["f_statistic"]:.2f}, η²={discrimination_df.iloc[idx]["eta_squared"]:.3f})')
            axes[idx].set_xlabel('')
            axes[idx].set_ylabel('Normalized Value')
            axes[idx].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for idx in range(len(top_features), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Top Discriminative Features for {tag_column.capitalize()}', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / f"violin_{tag_column}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved violin plot to: {output_path}")
    
    def plot_scatter_2d(self, df, feature_df, tag_column, tag_labels, top_tags, feature_pairs=None):
        """
        Plot 2D scatter plots for feature pairs
        
        Args:
            df: Original DataFrame
            feature_df: DataFrame with feature values
            tag_column: Tag category name
            tag_labels: Series with tag labels
            top_tags: List of top tags
            feature_pairs: List of (feature1, feature2) tuples. If None, auto-select
        """
        print(f"Plotting 2D scatter plots for {tag_column}...")
        
        if feature_pairs is None:
            # Auto-select top discriminative features and pair them
            discrimination_df = self.calculate_feature_discrimination(feature_df, tag_labels)
            top_features = discrimination_df.head(6)['feature'].tolist()
            feature_pairs = [(top_features[i], top_features[i+1]) for i in range(0, len(top_features)-1, 2)]
        
        # Create subplots
        n_pairs = len(feature_pairs)
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten() if n_pairs > 1 else [axes]
        
        # Color palette
        colors = sns.color_palette('Set2', n_colors=len(top_tags))
        color_map = dict(zip(top_tags, colors))
        
        for idx, (feat1, feat2) in enumerate(feature_pairs):
            if idx >= len(axes):
                break
            
            # Plot each tag
            for tag in top_tags:
                mask = tag_labels == tag
                axes[idx].scatter(
                    feature_df.loc[mask, feat1],
                    feature_df.loc[mask, feat2],
                    c=[color_map[tag]],
                    label=tag,
                    alpha=0.6,
                    s=20
                )
            
            axes[idx].set_xlabel(feat1)
            axes[idx].set_ylabel(feat2)
            axes[idx].set_title(f'{feat1} vs {feat2}')
            axes[idx].legend(loc='best', fontsize=8)
            axes[idx].grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'2D Feature Space for {tag_column.capitalize()}', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / f"scatter2d_{tag_column}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved 2D scatter plot to: {output_path}")
    
    def plot_scatter_2d_specific(self, df, feature_df, tag_column, tag_labels, top_tags, 
                                 feature_pairs, suffix=''):
        """
        Plot 2D scatter plots for specific feature pairs (larger, cleaner plots)
        
        Args:
            df: Original DataFrame
            feature_df: DataFrame with feature values
            tag_column: Tag category name
            tag_labels: Series with tag labels
            top_tags: List of top tags
            feature_pairs: List of (feature1, feature2) tuples
            suffix: Suffix for output filename
        """
        print(f"Plotting specific 2D scatter plots for {tag_column}...")
        
        # Color palette
        colors = sns.color_palette('Set2', n_colors=len(top_tags))
        color_map = dict(zip(top_tags, colors))
        
        for feat1, feat2 in feature_pairs:
            # Create single large plot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot each tag
            for tag in top_tags:
                mask = tag_labels == tag
                n_samples = mask.sum()
                ax.scatter(
                    feature_df.loc[mask, feat1],
                    feature_df.loc[mask, feat2],
                    c=[color_map[tag]],
                    label=f'{tag} (n={n_samples})',
                    alpha=0.6,
                    s=30,
                    edgecolors='white',
                    linewidth=0.5
                )
            
            ax.set_xlabel(feat1, fontsize=12)
            ax.set_ylabel(feat2, fontsize=12)
            ax.set_title(f'{feat1} vs {feat2}\n{tag_column.capitalize()} Tags', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            
            # Clean feature names for filename
            feat1_clean = feat1.replace('_', '')
            feat2_clean = feat2.replace('_', '')
            output_path = self.output_dir / f"scatter2d_{tag_column}{suffix}_{feat1_clean}_vs_{feat2_clean}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved specific 2D scatter plot to: {output_path}")
    
    def plot_tonal_vector_3d(self, df, feature_df, tag_column, tag_labels, top_tags):
        """
        Plot 3D scatter plot for tonal vector (key_vector_x, key_vector_y, key_vector_z)
        
        Args:
            df: Original DataFrame
            feature_df: DataFrame with feature values
            tag_column: Tag category name
            tag_labels: Series with tag labels
            top_tags: List of top tags
        """
        # Check if tonal vector features exist
        required_features = ['key_vector_x', 'key_vector_y', 'key_vector_z']
        if not all(feat in feature_df.columns for feat in required_features):
            print(f"Skipping tonal vector 3D plot for {tag_column}: missing required features")
            return
        
        print(f"Plotting interactive 3D tonal vector for {tag_column}...")
        
        # Filter to top tags
        mask = tag_labels.isin(top_tags)
        filtered_feature_df = feature_df.loc[mask]
        filtered_labels = tag_labels.loc[mask]
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'X (Circle of Fifths)': filtered_feature_df['key_vector_x'].values,
            'Y (Circle of Fifths)': filtered_feature_df['key_vector_y'].values,
            'Z (Major/Minor)': filtered_feature_df['key_vector_z'].values,
            'Tag': filtered_labels.values
        })
        
        # Count samples per tag
        tag_counts = filtered_labels.value_counts()
        plot_df['Tag_with_count'] = plot_df['Tag'].apply(lambda x: f"{x} (n={tag_counts[x]})")
        
        # Create interactive 3D scatter plot
        fig = px.scatter_3d(
            plot_df,
            x='X (Circle of Fifths)',
            y='Y (Circle of Fifths)',
            z='Z (Major/Minor)',
            color='Tag',
            hover_data=['Tag_with_count'],
            title=f'Interactive Tonal Vector 3D Space - {tag_column.capitalize()} Tags',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        # Update layout for better visualization
        fig.update_traces(marker=dict(size=5, opacity=0.7, line=dict(width=0.5, color='white')))
        fig.update_layout(
            width=1200,
            height=900,
            font=dict(size=12),
            legend=dict(title=dict(text='Tags'), font=dict(size=10)),
            scene=dict(
                xaxis=dict(title='X - Circle of Fifths (cos)', backgroundcolor="rgb(230, 230, 230)"),
                yaxis=dict(title='Y - Circle of Fifths (sin)', backgroundcolor="rgb(230, 230, 230)"),
                zaxis=dict(title='Z - Major(+0.5) / Minor(-0.5)', backgroundcolor="rgb(230, 230, 230)"),
            )
        )
        
        # Save as interactive HTML
        output_path = self.output_dir / f"tonal3d_interactive_{tag_column}.html"
        fig.write_html(output_path)
        print(f"Saved interactive tonal vector 3D plot to: {output_path}")
        print(f"  Open in browser to interact with the plot!")
    
    def plot_umap_interactive_3d(self, df, feature_df, tag_column, tag_labels, top_tags):
        """
        Plot interactive 3D UMAP visualization using Plotly
        
        Args:
            df: Original DataFrame
            feature_df: DataFrame with feature values
            tag_column: Tag category name
            tag_labels: Series with tag labels
            top_tags: List of top tags
        """
        print(f"Computing UMAP 3D embedding for {tag_column}...")
        
        # Filter to top tags only
        mask = tag_labels.isin(top_tags)
        filtered_feature_df = feature_df.loc[mask]
        filtered_labels = tag_labels.loc[mask]
        
        # Compute UMAP
        try:
            umap_model = UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
            embedding = umap_model.fit_transform(filtered_feature_df)
        except Exception as e:
            print(f"Error computing UMAP for {tag_column}: {e}")
            return
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'UMAP3': embedding[:, 2],
            'Tag': filtered_labels.values
        })
        
        # Count samples per tag
        tag_counts = filtered_labels.value_counts()
        plot_df['Tag_with_count'] = plot_df['Tag'].apply(lambda x: f"{x} (n={tag_counts[x]})")
        
        # Create interactive 3D scatter plot
        fig = px.scatter_3d(
            plot_df,
            x='UMAP1',
            y='UMAP2',
            z='UMAP3',
            color='Tag',
            hover_data=['Tag_with_count'],
            title=f'UMAP 3D Projection - {tag_column.capitalize()} Tags',
            labels={'UMAP1': 'UMAP Dimension 1', 'UMAP2': 'UMAP Dimension 2', 'UMAP3': 'UMAP Dimension 3'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        # Update layout for better visualization
        fig.update_traces(marker=dict(size=5, opacity=0.7, line=dict(width=0.5, color='white')))
        fig.update_layout(
            width=1200,
            height=900,
            font=dict(size=12),
            legend=dict(title=dict(text='Tags'), font=dict(size=10)),
            scene=dict(
                xaxis=dict(title='UMAP Dimension 1', backgroundcolor="rgb(230, 230, 230)"),
                yaxis=dict(title='UMAP Dimension 2', backgroundcolor="rgb(230, 230, 230)"),
                zaxis=dict(title='UMAP Dimension 3', backgroundcolor="rgb(230, 230, 230)"),
            )
        )
        
        # Save as interactive HTML
        output_path = self.output_dir / f"umap3d_interactive_{tag_column}.html"
        fig.write_html(output_path)
        print(f"Saved interactive UMAP 3D plot to: {output_path}")
        print(f"  Open in browser to interact with the plot!")
    
    def save_discrimination_results(self, discrimination_dfs, tag_categories):
        """
        Save discrimination analysis results
        
        Args:
            discrimination_dfs: List of discrimination DataFrames
            tag_categories: List of tag category names
        """
        # Save individual CSVs
        for disc_df, category in zip(discrimination_dfs, tag_categories):
            output_path = self.output_dir / f"discrimination_{category}.csv"
            disc_df.to_csv(output_path, index=False)
            print(f"Saved discrimination results to: {output_path}")
        
        # Save summary text
        output_path = self.output_dir / "summary_statistics.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Feature Discrimination Analysis Summary\n")
            f.write("=" * 80 + "\n\n")
            
            for disc_df, category in zip(discrimination_dfs, tag_categories):
                f.write(f"\n{'=' * 80}\n")
                f.write(f"{category.upper()} CATEGORY\n")
                f.write(f"{'=' * 80}\n\n")
                
                # Overall statistics
                f.write(f"Number of features analyzed: {len(disc_df)}\n")
                f.write(f"Significant features (p<0.05): {disc_df['significant'].sum()}\n")
                f.write(f"Mean F-statistic: {disc_df['f_statistic'].mean():.2f}\n")
                f.write(f"Mean effect size (η²): {disc_df['eta_squared'].mean():.3f}\n\n")
                
                # ALL features ranked (from best to worst)
                f.write(f"ALL Features Ranked (Total: {len(disc_df)}):\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Rank':<6}{'Feature':<40}{'F-stat':<12}{'p-value':<12}{'η²':<8}\n")
                f.write("-" * 80 + "\n")
                
                for i, row in disc_df.iterrows():
                    f.write(f"{i+1:<6}{row['feature']:<40}{row['f_statistic']:<12.2f}"
                           f"{row['p_value']:<12.2e}{row['eta_squared']:<8.3f}\n")
                
                f.write("\n")
        
        print(f"Saved summary statistics to: {output_path}")
    
    def run_analysis(self, csv_name='features_all.csv', extract_all=False, extract_comprehensive=False, normalize=True):
        """
        Run complete discrimination analysis
        
        Args:
            csv_name: Name of CSV file to load/save (default: features_all.csv)
            extract_all: If True, extract ALL features (including alternatives)
            extract_comprehensive: If True, extract ALL AcousticBrainz features (comprehensive)
            normalize: Whether to normalize features before analysis
        """
        print("\n" + "=" * 80)
        print("Feature Discrimination Analysis")
        print("=" * 80 + "\n")
        
        # Load data
        df = self.load_data(csv_name=csv_name, extract_all=extract_all, extract_comprehensive=extract_comprehensive)
        
        # Prepare features
        feature_df, feature_names = self.prepare_features(df, normalize=normalize)
        
        # Analyze each tag category
        tag_categories = ['genre', 'mood', 'instrument']
        discrimination_dfs = []
        
        for category in tag_categories:
            print(f"\n{'-' * 80}")
            print(f"Analyzing {category.upper()} tags")
            print(f"{'-' * 80}\n")
            
            # Get top tags and filter data
            filtered_df, top_tags = self.get_top_tags(df, category)
            
            if len(filtered_df) < 10:
                print(f"Warning: Not enough data for {category}, skipping...")
                continue
            
            # Get corresponding feature data
            filtered_feature_df = feature_df.loc[filtered_df.index]
            tag_labels = filtered_df[f'{category}_single']
            
            # Calculate discrimination
            discrimination_df = self.calculate_feature_discrimination(filtered_feature_df, tag_labels)
            discrimination_dfs.append(discrimination_df)
            
            # Plot violin plots (1D)
            self.plot_violin_1d(filtered_df, filtered_feature_df, category, tag_labels, top_tags)
            
            # Plot scatter plots (2D) - auto-selected pairs
            self.plot_scatter_2d(filtered_df, filtered_feature_df, category, tag_labels, top_tags)
            
            # Plot specific feature pairs: spec_contrast_mean_i vs spec_valley_mean_i for all i
            contrast_pairs = []
            for i in range(6):  # 0 to 5
                feat1 = f'spec_contrast_mean_{i}'
                feat2 = f'spec_valley_mean_{i}'
                if feat1 in filtered_feature_df.columns and feat2 in filtered_feature_df.columns:
                    contrast_pairs.append((feat1, feat2))
            
            if contrast_pairs:
                print(f"Plotting {len(contrast_pairs)} contrast-valley pairs: {contrast_pairs}")
                self.plot_scatter_2d_specific(
                    filtered_df, filtered_feature_df, category, tag_labels, 
                    top_tags, contrast_pairs, suffix='_contrast'
                )
            
            # Plot 3D tonal vector
            self.plot_tonal_vector_3d(filtered_df, filtered_feature_df, category, tag_labels, top_tags)
            
            # Plot interactive UMAP 3D
            self.plot_umap_interactive_3d(filtered_df, filtered_feature_df, category, tag_labels, top_tags)
        
        # Save results
        self.save_discrimination_results(discrimination_dfs, tag_categories)
        
        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)
        print(f"\nAll results saved to: {self.output_dir}")


def main():
    """Main execution function with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Feature Discrimination Analysis for Tag Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_corr.py                        # Use cached CSV (features_all.csv or features_2tags.csv)
  python analyze_corr.py --f                    # Extract ALL features, save as features_all.csv
  python analyze_corr.py --comprehensive        # Extract COMPREHENSIVE features (ALL AcousticBrainz), save as features_comprehensive.csv
  python analyze_corr.py --f --name custom      # Extract ALL features, save as features_custom.csv
  python analyze_corr.py --comprehensive --name my_analysis  # Extract comprehensive, save as my_analysis.csv
  python analyze_corr.py --top-n 15             # Analyze top 15 tags per category
  python analyze_corr.py --no-normalize         # Disable feature normalization
        """
    )
    
    parser.add_argument('--f', action='store_true',
                        help='Extract ALL features (MFCC+MEL, Centroid+Energy, all optional features)')
    parser.add_argument('--comprehensive', action='store_true',
                        help='Extract COMPREHENSIVE features (ALL AcousticBrainz features - Priority Low まで)')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom name for features CSV (default: features_all.csv)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top tags to analyze per category (default: 10)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable feature normalization')
    
    args = parser.parse_args()
    
    # Determine CSV name and extraction method
    extract_comprehensive = args.comprehensive
    
    if extract_comprehensive:
        # Extract comprehensive features
        csv_name = args.name if args.name else 'features_comprehensive.csv'
        extract_all = False
    elif args.f:
        # Extract all features
        csv_name = args.name if args.name else 'features_all.csv'
        extract_all = True
    else:
        # Try to use cached CSV
        # Priority: custom name > features_all.csv > features_2tags.csv
        if args.name:
            csv_name = args.name if args.name.endswith('.csv') else f'{args.name}.csv'
        else:
            # Check which cached CSV exists
            data_dir = Path(__file__).parent / 'outputs' / 'data'
            if (data_dir / 'features_all.csv').exists():
                csv_name = 'features_all.csv'
                print(f"Found cached features_all.csv, using it...")
            elif (data_dir / 'features_2tags.csv').exists():
                csv_name = 'features_2tags.csv'
                print(f"Found cached features_2tags.csv (from run.py), using it...")
            else:
                csv_name = 'features_all.csv'
                print(f"No cached CSV found, will extract default features...")
        extract_all = False
    
    normalize = not args.no_normalize
    
    print(f"\nConfiguration:")
    print(f"  CSV name: {csv_name}")
    print(f"  Extract all features: {extract_all}")
    print(f"  Extract comprehensive features: {extract_comprehensive}")
    print(f"  Top N tags: {args.top_n}")
    print(f"  Normalize: {normalize}")
    print()
    
    # Create analyzer
    analyzer = FeatureDiscriminationAnalyzer(top_n_tags=args.top_n)
    
    # Run analysis
    analyzer.run_analysis(
        csv_name=csv_name,
        extract_all=extract_all,
        extract_comprehensive=extract_comprehensive,
        normalize=normalize
    )


if __name__ == "__main__":
    main()
