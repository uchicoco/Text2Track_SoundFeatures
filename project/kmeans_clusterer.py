import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path

class KMeansProcessor:
    def __init__(self, n_tokens=2, n_clusters=128, max_iter=200, n_init=16,
                 n_queries=100, k_eval=10):
        self.n_tokens = n_tokens
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.n_queries = n_queries
        self.k_eval = k_eval

    def run_rvq(self, feat_pca,
                n_tokens, n_clusters, max_iter, n_init):
        """
        Args: feat_pca (np.ndarray): PCA-reduced feature matrix
                n_tokens (int): number of tokens to create
                n_clusters (int): number of clusters for k-means
                max_iter (int): maximum iterations for k-means
                n_init (int): number of initializations for k-means
        Returns: codebook (np.ndarray): cluster centers
                codes (np.ndarray): cluster labels for each sample
                n_tokens (int): number of tokens used
                n_clusters (int): number of clusters used
        """
        # set parameters
        if n_tokens is None: n_tokens = self.n_tokens
        if n_clusters is None: n_clusters = self.n_clusters 
        if max_iter is None: max_iter = self.max_iter
        if n_init is None: n_init = self.n_init

        codebooks = [] # save k-means models for each token
        codes = np.zeros((feat_pca.shape[0], n_tokens), dtype=np.int32) # [n_samples, n_tokens]

        residual = feat_pca.copy()

        for l in range(n_tokens):
            print(f"Running KMeans for token {l+1}/{n_tokens}")
            kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, random_state=42)
            labels = kmeans.fit_predict(residual)
            assigned_centers = kmeans.cluster_centers_[labels]
            codebooks.append(kmeans)
            codes[:, l] = labels.astype(np.int32)

            # update residuals
            residual -= assigned_centers
        return codebooks, codes, n_tokens, n_clusters
    
def main():
    from pathlib import Path

    from dataset_processor import DatasetProcessor
    from pca_processor import PCAProcessor
    from feature_extractor import FeatureExtractor
    
    dp = DatasetProcessor()
    
    # load PCA features if available
    pca_path = Path(dp.output_dir) / "data/features_pca.npy"
    if pca_path.exists():
        print("Loading PCA features from file")
        feat_pca = np.load(pca_path)
    else:
        print("PCA features not found, generating from scratch")
        
        # load dataset
        csv_path = Path(dp.output_dir) / "data/features_2tags.csv"
        if csv_path.exists():
            df_two = pd.read_csv(csv_path)
        else:
            # build dataset from scratch
            fe = FeatureExtractor()
            mood_df = dp.load_tag_tsv("autotagging_moodtheme.tsv", "mood")
            genre_df = dp.load_tag_tsv("autotagging_genre.tsv", "genre")
            inst_df = dp.load_tag_tsv("autotagging_instrument.tsv", "instrument")
            tags_merged = (mood_df
                        .merge(genre_df, on=["track_id", "path"], how="outer")
                        .merge(inst_df, on=["track_id", "path"], how="outer"))
            tags_merged[["genre", "mood", "instrument"]] = tags_merged[["genre", "mood", "instrument"]].fillna("")
            
            df = fe.build_features_dataframe(tags_merged)
            df_two = fe.filter_min2tags(df)
        
        # run PCA
        pp = PCAProcessor()
        feat_matrix = pp.build_feature_matrix(df_two)
        feat_pca, cum_var, expl_var, pca = pp.run_pca_components(feat_matrix)
        pp.save_feat_pca(feat_pca, expl_var)

    # run RVQ clustering
    kmp = KMeansProcessor()
    codebooks, codes, n_tokens, n_clusters = kmp.run_rvq(
        feat_pca, n_tokens=2, n_clusters=32, max_iter=200, n_init=16
    )
    
    # display results
    print(f"\nRVQ Results:")
    print(f"Feature matrix shape: {feat_pca.shape}")
    print(f"Generated codes shape: {codes.shape}")
    print(f"Number of codebooks: {len(codebooks)}")
    
    # check code distribution
    for i in range(n_tokens):
        unique_codes = np.unique(codes[:, i])
        print(f"Token {i}: {len(unique_codes)} unique codes (expected: {n_clusters})")
    
    # generate semantic IDs for example
    semantic_ids = []
    for row in codes:
        semantic_id = "".join([f"<{int(c):03d}>" for c in row])
        semantic_ids.append(semantic_id)
    
    # check semantic ID distribution
    unique_semantic_ids = len(set(semantic_ids))
    print(f"\nSemantic ID Statistics:")
    print(f"Total tracks: {len(semantic_ids)}")
    print(f"Unique semantic IDs: {unique_semantic_ids}")
    print(f"ID diversity: {unique_semantic_ids / len(semantic_ids):.3f}")
    
    # show sample IDs
    print(f"\nSample semantic IDs:")
    for i in range(min(10, len(semantic_ids))):
        print(f"  {semantic_ids[i]}")

if __name__ == "__main__":
    main()