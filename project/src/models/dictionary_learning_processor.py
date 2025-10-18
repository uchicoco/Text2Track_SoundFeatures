from pathlib import Path
import sys
import numpy as np
from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning, sparse_encode

# Add project root to sys.path for standalone execution
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class DictionaryLearningProcessor:
    def __init__(self, n_nonzero_coefs=2, n_dict_components=128, max_iter=100, batch_size=256):
        """
        Args:
            n_nonzero_coefs (int): equivalent to n_tokens
            n_dict_components (int): equivalent to n_clusters
            max_iter (int): maximum iterations for dictionary learning
            batch_size (int): size of mini-batches for fitting
        """
        self.n_nonzero_coefs = n_nonzero_coefs
        self.n_dict_components = n_dict_components
        self.max_iter = max_iter
        self.batch_size = batch_size
    
    @staticmethod
    def _sparse_codes_to_tokens(sparse_codes, n_nonzero_coefs, n_dict_components):
        """Convert sparse codes to token indices
        
        Args:
            sparse_codes (np.ndarray): Sparse coefficient matrix
            n_nonzero_coefs (int): Number of non-zero coefficients to keep
            n_dict_components (int): Number of dictionary components
            
        Returns:
            np.ndarray: Token indices of shape (n_samples, n_nonzero_coefs)
        """
        n_samples = sparse_codes.shape[0]
        codes = np.zeros((n_samples, n_nonzero_coefs), dtype=np.int32)
        
        for i in range(n_samples):
            coef = sparse_codes[i, :]
            nonzero_indices = np.nonzero(coef)[0]
            
            # Sort by absolute value and take top-k
            if len(nonzero_indices) >= n_nonzero_coefs:
                abs_coef = np.abs(coef[nonzero_indices])
                top_k_positions = np.argpartition(abs_coef, -n_nonzero_coefs)[-n_nonzero_coefs:]
                sorted_positions = top_k_positions[np.argsort(-abs_coef[top_k_positions])]
                top_indices = nonzero_indices[sorted_positions]
            else:
                top_indices = nonzero_indices[np.argsort(-np.abs(coef[nonzero_indices]))]
            
            # Assign tokens: positive -> idx, negative -> idx + n_dict_components
            for k, idx in enumerate(top_indices):
                codes[i, k] = idx if coef[idx] >= 0 else idx + n_dict_components
        
        return codes

    def run_sparse_coding_minibatch(self, feat_pca,
                          n_nonzero_coefs=None, n_dict_components=None, max_iter=None,
                          batch_size=256):
        """
        Generate semantic IDs using Dictionary Learning and Sparse Coding.

        Args:
            feat_pca (np.ndarray): PCA-reduced feature matrix
            n_nonzero_coefs (int, optional): number of tokens to create
            n_dict_components (int, optional): number of dictionary components to use
            max_iter (int, optional): maximum iterations for dictionary learning
            batch_size (int): size of mini-batches for fitting

        Returns:
            dictionary (np.ndarray): learned dictionary (n_dict_components, n_features).
            codes (np.ndarray): cluster labels for each sample
            n_tokens (int): number of tokens used
            n_dict_components (int): number of dictionary components used
        """
        # Set parameters
        if n_nonzero_coefs is None: n_nonzero_coefs = self.n_nonzero_coefs
        if n_dict_components is None: n_dict_components = self.n_dict_components
        if max_iter is None: max_iter = self.max_iter

        # Dictionary learning
        print(f"Running DictionaryLearning with {n_dict_components} components...")
        dict_learner = MiniBatchDictionaryLearning(
            n_components=n_dict_components,
            transform_algorithm='omp',  # Orthogonal Matching Pursuit
            transform_n_nonzero_coefs=n_nonzero_coefs,
            max_iter=max_iter,
            batch_size=batch_size,
            random_state=42,
            n_jobs=-1 # For parallel processing
        )
        # y = D*x
        # Learn dictionary D and sparse codes x
        print("Fitting dictionary...")
        dict_learner.fit(feat_pca)
        dictionary = dict_learner.components_
        sparse_codes = dict_learner.transform(feat_pca)

        print("Generating semantic ID tokens from sparse codes...")
        codes = self._sparse_codes_to_tokens(sparse_codes, n_nonzero_coefs, n_dict_components)

        return dictionary, codes, n_nonzero_coefs, n_dict_components

    # Warning: takes a long time to run
    def run_sparse_coding(self, feat_pca,
                          n_nonzero_coefs=None, n_dict_components=None, max_iter=None):
        """
        Generate semantic IDs using Dictionary Learning and Sparse Coding.

        Args:
            feat_pca (np.ndarray): PCA-reduced feature matrix
            n_nonzero_coefs (int, optional): number of tokens to create
            n_dict_components (int, optional): number of dictionary components to use
            max_iter (int, optional): maximum iterations for dictionary learning

        Returns:
            dictionary (np.ndarray): learned dictionary (n_dict_components, n_features).
            codes (np.ndarray): cluster labels for each sample
            n_tokens (int): number of tokens used
            n_dict_components (int): number of dictionary components used
        """
        # Set parameters
        if n_nonzero_coefs is None: n_nonzero_coefs = self.n_nonzero_coefs
        if n_dict_components is None: n_dict_components = self.n_dict_components
        if max_iter is None: max_iter = self.max_iter

        # Dictionary learning
        print(f"Running DictionaryLearning with {n_dict_components} components...")
        dict_learner = DictionaryLearning(
            n_components=n_dict_components,
            transform_algorithm='omp',  # Orthogonal Matching Pursuit
            transform_n_nonzero_coefs=n_nonzero_coefs,
            max_iter=max_iter,
            random_state=42,
            n_jobs=-1 # For parallel processing
        )
        # y = D*x
        # Learn dictionary D and sparse codes x
        print("Fitting dictionary...")
        sparse_codes = dict_learner.fit_transform(feat_pca)
        dictionary = dict_learner.components_

        print("Generating semantic ID tokens from sparse codes...")
        codes = self._sparse_codes_to_tokens(sparse_codes, n_nonzero_coefs, n_dict_components)

        return dictionary, codes, n_nonzero_coefs, n_dict_components
    
def main():
    from pathlib import Path
    import pandas as pd

    from src.data.dataset_processor import DatasetProcessor
    from src.models.pca_processor import PCAProcessor
    from src.data.feature_extractor import FeatureExtractor

    dp = DatasetProcessor()

    # Load PCA features if available
    pca_path = Path(dp.output_dir) / "data/features_pca.npy"
    if pca_path.exists():
        print("Loading PCA features from file")
        feat_pca = np.load(pca_path)
    else:
        print("PCA features not found, generating from scratch")

        # Load dataset
        csv_path = Path(dp.output_dir) / "data/features_2tags.csv"
        if csv_path.exists():
            df_two = pd.read_csv(csv_path)
        else:
            # Build dataset from scratch
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

        # Run PCA
        pp = PCAProcessor()
        feat_matrix = pp.build_feature_matrix(df_two)
        feat_pca, cum_var, expl_var, pca = pp.run_pca_components(feat_matrix)
        pp.save_feat_pca(feat_pca, expl_var)

    # Run dictionary learning
    dlp = DictionaryLearningProcessor()
    dictionary, codes, n_nonzero_coefs, n_dict_components = dlp.run_sparse_coding_minibatch(
        feat_pca, 
        n_nonzero_coefs=2, 
        n_dict_components=16, 
        max_iter=200
    )

    # Display results
    print(f"\nSparse Coding Results:")
    print(f"Feature matrix shape: {feat_pca.shape}")
    print(f"Generated codes shape: {codes.shape}")
    print(f"Learned dictionary shape: {dictionary.shape}")

    # Check code distribution
    vocab_size = 2 * n_dict_components
    for i in range(n_nonzero_coefs):
        unique_codes = np.unique(codes[:, i])
        print(f"Token {i+1}: {len(unique_codes)} unique codes (total vocab size: {vocab_size})")
    
    # generate semantic IDs
    semantic_ids = []
    for row in codes:
        semantic_id = "".join([f"<{int(c):03d}>" for c in row])
        semantic_ids.append(semantic_id)

    # Check semantic ID distribution
    unique_semantic_ids = len(set(semantic_ids))
    print(f"\nSemantic ID Statistics:")
    print(f"Total tracks: {len(semantic_ids)}")
    print(f"Unique semantic IDs: {unique_semantic_ids}")
    print(f"ID diversity: {unique_semantic_ids / len(semantic_ids):.3f}")

    # Show sample IDs
    print(f"\nSample semantic IDs:")
    for i in range(min(10, len(semantic_ids))):
        print(f"  {semantic_ids[i]}")

if __name__ == "__main__":
    main()
