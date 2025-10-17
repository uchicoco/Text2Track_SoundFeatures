import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataset_processor import DatasetProcessor

class PCAProcessor:
    def __init__(self, n_components=36, r_explained_var=0.95):
        self.exclude_cols = ["json_full", "track_id", "path", "genre", "mood", "instrument"]
        self.n_components = n_components
        self.r_explained_var = r_explained_var
        self.scaler = StandardScaler()
        self.dp = DatasetProcessor()

    def build_feature_matrix(self, df):
        """
        Args: df (pd.DataFrame): dataframe with features and tags
        Returns: np.ndarray: standardized feature matrix (rows=samples, cols=features)
        """
        feat_cols = [c for c in df.columns if c not in self.exclude_cols]
        feat_matrix = df[feat_cols].to_numpy(dtype=float)
        # standardize
        feat_matrix = self.scaler.fit_transform(feat_matrix)
        print(f"feature matrix shape: {feat_matrix.shape}")
        return feat_matrix
    
    def run_pca_components(self, feat_matrix, n_components=None):
        """
        Args: feat_matrix (np.ndarray): standardized feature matrix
                n_components (int or None): number of PCA components to keep; if None, use instance variable
        Returns: feat_pca (np.ndarray): PCA-transformed feature matrix
                cum_var (np.ndarray): cumulative explained variance ratio
                expl_var (np.ndarray): explained variance ratio per component
                pca (PCA object): fitted PCA model
        """
        if n_components is not None:
            self.n_components = n_components

        pca = PCA(n_components=self.n_components)
        feat_pca = pca.fit_transform(feat_matrix)

        expl_var = pca.explained_variance_ratio_
        cum_var = np.cumsum(expl_var)

        print(f"PCA: {feat_matrix.shape[1]} -> {feat_pca.shape[1]} dims")
        print(f"Total variance retained: {cum_var[-1]:.3f}")
        print(f"Explained variance (first 5 comps): {[round(v, 4) for v in expl_var[:5]]}")
        return feat_pca, cum_var, expl_var, pca
    
    def run_pca_ratio(self, feat_matrix, r_explained_var=None):
        """
        Args: feat_matrix (np.ndarray): standardized feature matrix
                r_explained_var (float or None): target explained variance ratio; if None, use instance variable
        Returns: feat_pca (np.ndarray): PCA-transformed feature matrix
                cum_var (np.ndarray): cumulative explained variance ratio
                expl_var (np.ndarray): explained variance ratio per component
                pca (PCA object): fitted PCA model
        """
        if r_explained_var is not None:
            self.r_explained_var = r_explained_var

        pca = PCA(n_components=self.r_explained_var)
        feat_pca = pca.fit_transform(feat_matrix)

        expl_var = pca.explained_variance_ratio_
        cum_var = np.cumsum(expl_var)

        print(f"PCA: {feat_matrix.shape[1]} -> {feat_pca.shape[1]} dims")
        print(f"Total variance retained: {cum_var[-1]:.3f}")
        print(f"Explained variance (first 5 comps): {[round(v, 4) for v in expl_var[:5]]}")
        return feat_pca, cum_var, expl_var, pca

    def save_feat_pca(self, feat_pca, expl_var):
        data_dir = Path(self.dp.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / f"features_pca.npy", feat_pca)
        pd.Series(expl_var).to_csv(data_dir / "pca_explained_variance_ratio.csv", index=False)
        return None
    
    def plot_pca_with_tags(self, feat_pca, df, save_html=True, 
                          sample_size=1000, figsize=(12, 8), n_tags=5):
        """
        Plot PCA results with color-coded tags using interactive plotly visualization
        Creates 3D plots for genre, mood, and instrument
        
        Args: feat_pca (np.ndarray): PCA-transformed feature matrix
               df (pd.DataFrame): original dataframe with tag information
               save_html (bool): whether to save interactive plot as HTML
               sample_size (int): number of samples to plot (for performance)
               figsize (tuple): figure size for static plots
        """
        # create 3D plots
        for tag_column in ['genre', 'mood', 'instrument']:
            if tag_column not in df.columns:
                continue
                
            self._create_single_tag_plot(feat_pca, df, tag_column, save_html, sample_size, n_tags)

        return None

    def _create_single_tag_plot(self, feat_pca, df, tag_column, save_html, sample_size, n_tags):
        """
        Create a single 3D plot for the specified tag column
        """
        df_plot = df.copy()
        feat_pca_plot = feat_pca.copy()
        
        # clean tag data
        df_plot[tag_column] = df_plot[tag_column].fillna('') # fill NaN with empty string
        valid_mask = df_plot[tag_column] != ''
        df_plot = df_plot[valid_mask].reset_index(drop=True)  # remove empty tags
        feat_pca_plot = feat_pca_plot[valid_mask]

        top_tags = df_plot[tag_column].value_counts().head(n_tags).index.tolist() # get top tags
        top_mask = df_plot[tag_column].isin(top_tags)
        df_plot = df_plot[top_mask].reset_index(drop=True) # keep only top tags
        feat_pca_plot = feat_pca_plot[top_mask]

        # sample data if too large
        if len(feat_pca_plot) > sample_size:
            idx = np.random.choice(len(feat_pca_plot), sample_size, replace=False)
            feat_pca_plot = feat_pca_plot[idx]
            df_plot = df_plot.iloc[idx].copy().reset_index(drop=True)
        else:
            feat_pca_plot = feat_pca_plot.copy()
            df_plot = df_plot.copy().reset_index(drop=True)

        # create interactive 3D plot with plotly
        if feat_pca_plot.shape[1] >= 3:
            fig = px.scatter_3d(
                x=feat_pca_plot[:, 0], 
                y=feat_pca_plot[:, 1], 
                z=feat_pca_plot[:, 2],
                color=df_plot[tag_column],
                hover_data={'track_id': df_plot['track_id'].values if 'track_id' in df_plot.columns else None},
                title=f'PCA Visualization colored by {tag_column.title()}',
                labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                opacity=0.7
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2', 
                    zaxis_title='PC3'
                ),
                width=800,
                height=600
            )
        else:
            # 2D plot if less than 3 components
            fig = px.scatter(
                x=feat_pca_plot[:, 0], 
                y=feat_pca_plot[:, 1],
                color=df_plot[tag_column],
                hover_data={'track_id': df_plot['track_id'].values if 'track_id' in df_plot.columns else None},
                title=f'PCA Visualization colored by {tag_column.title()}',
                labels={'x': 'PC1', 'y': 'PC2'},
                opacity=0.7
            )

        fig.show()

        # save interactive plot
        if save_html:
            figures_dir = Path(self.dp.output_dir) / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            output_path = figures_dir / f"pca_plot_{tag_column}.html"
            fig.write_html(output_path)
            print(f"Interactive plot saved: {output_path}")

    def plot_explained_variance(self, expl_var, cum_var):
        """
        Plot explained variance ratio and cumulative variance
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # individual explained variance
        ax1.bar(range(1, len(expl_var) + 1), expl_var)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.grid(True, alpha=0.3)

        # cumulative explained variance
        ax2.plot(range(1, len(cum_var) + 1), cum_var, 'bo-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95%')
        ax2.axhline(y=0.90, color='orange', linestyle='--', label='90%')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()

        # save variance plot
        figures_dir = Path(self.dp.output_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        output_path = figures_dir / "pca_explained_variance.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Variance plot saved: {output_path}")

def main():
    from feature_extractor import FeatureExtractor

    fe = FeatureExtractor()
    dp = DatasetProcessor()

    csv_path = Path(dp.output_dir) / "data/features_2tags.csv"
    if csv_path.exists():
        print("csv found")
        df_two = pd.read_csv(csv_path)
    else:
        print("csv not found, building from tsv")
        mood_df  = dp.load_tag_tsv("autotagging_moodtheme.tsv", "mood")
        genre_df = dp.load_tag_tsv("autotagging_genre.tsv", "genre")
        inst_df  = dp.load_tag_tsv("autotagging_instrument.tsv", "instrument")
        tags_merged = (mood_df
                    .merge(genre_df, on=["track_id", "path"], how="outer")
                    .merge(inst_df, on=["track_id", "path"], how="outer"))
        tags_merged[["genre", "mood", "instrument"]] = tags_merged[["genre", "mood", "instrument"]].fillna("")

        df = fe.build_features_dataframe(tags_merged)
        df_two = fe.filter_min2tags(df)

    pp = PCAProcessor()
    feat_matrix = pp.build_feature_matrix(df_two)
    feat_pca_n, cum_var, expl_var, pca = pp.run_pca_components(feat_matrix)
    feat_pca_r, cum_var, expl_var, pca = pp.run_pca_ratio(feat_matrix)

    pp.plot_pca_with_tags(feat_pca_n, df_two)
    pp.plot_explained_variance(expl_var, cum_var)
    pp.save_feat_pca(feat_pca_n, expl_var)

if __name__ == "__main__":
    main()