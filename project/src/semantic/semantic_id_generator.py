from collections import Counter
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import sparse_encode
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Handle imports for both module usage and standalone execution
try:
    from ..data.dataset_processor import DatasetProcessor
    from ..models.dictionary_learning_processor import DictionaryLearningProcessor
    from ..models.kmeans_clusterer import KMeansProcessor
except ImportError:
    # When run as standalone script, add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.data.dataset_processor import DatasetProcessor
    from src.models.dictionary_learning_processor import DictionaryLearningProcessor
    from src.models.kmeans_clusterer import KMeansProcessor

class SemanticIDGenerator:
    def __init__(self):
        self.dp = DatasetProcessor()
        self.kmp = KMeansProcessor()
        self.dlp = DictionaryLearningProcessor()
    
    @staticmethod
    def _codes_to_semantic_ids(codes):
        """Convert code matrix to semantic ID strings
        
        Args:
            codes (np.ndarray): Code matrix of shape (n_samples, n_tokens)
        
        Returns:
            list: List of semantic ID strings in format '<000><001>...'
        """
        return ["".join(f"<{int(c):03d}>" for c in row) for row in codes]
    
    def assign_sem_ids_kmeans(self, feat_pca, n_tokens, n_clusters, max_iter, n_init):
        """K-means RVQ semantic ID assignment"""
        kmp = KMeansProcessor()
        codebooks, codes, _, _ = kmp.run_rvq(feat_pca, n_tokens, n_clusters, max_iter, n_init)
        semantic_ids = self._codes_to_semantic_ids(codes)
        return semantic_ids, codebooks, codes

    def assign_sem_ids_kmean_manual(self, feat_pca, n_tokens=None, n_clusters=None, max_iter=None, n_init=None):
        """Save semantic IDs with manually specified config"""
        if n_tokens is None: n_tokens = self.kmp.n_tokens
        if n_clusters is None: n_clusters = self.kmp.n_clusters
        if max_iter is None: max_iter = self.kmp.max_iter
        if n_init is None: n_init = self.kmp.n_init

        print(f"Generating semantic IDs with manual config: L={n_tokens}, K={n_clusters}, n_init={n_init}")
        
        # Generate semantic IDs
        semantic_ids, codebooks, codes = self.assign_sem_ids_kmeans(
            feat_pca, n_tokens, n_clusters, max_iter, n_init
        )

        # Evaluate quality
        vc = pd.Series(semantic_ids).value_counts()
        print(f"Manual config results:")
        print(f"  Unique IDs: {vc.size}")
        print(f"  Mean per ID: {vc.mean():.2f}")
        print(f"  Singleton %: {(vc == 1).mean():.1%}")

        # Save config
        data_dir = Path(self.dp.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for l, cb in enumerate(codebooks):
            joblib.dump(cb, data_dir / f"best_kmeans_level{l}.joblib")
        np.save(data_dir / f"best_kmeans_codes.npy", codes)

        print(f"Saved manual config to {data_dir}")
        return semantic_ids
    
    def assign_sem_ids_kmean_load(self, feat_pca):
        """Load and apply saved config"""
        prefix = "best_kmeans"
        
        # Load saved codebooks
        codebooks = []
        level = 0
        while True:
            path = Path(self.dp.output_dir) / f"data/{prefix}_level{level}.joblib"
            if path.exists():
                codebooks.append(joblib.load(path))
                level += 1
            else:
                break
        
        if not codebooks:
            raise FileNotFoundError(f"No saved config found")
        
        print(f"Loaded saved config: L={len(codebooks)}")

        # Apply
        codes = np.zeros((feat_pca.shape[0], len(codebooks)), dtype=np.int32)
        residual = feat_pca.copy()
        
        for l, km in enumerate(codebooks):
            labels = km.predict(residual)
            centers = km.cluster_centers_[labels]
            residual = residual - centers
            codes[:, l] = labels.astype(np.int32)

        # Generate semantic IDs
        semantic_ids = self._codes_to_semantic_ids(codes)
        return semantic_ids
    
    def search_best_kmean(self, feat_pca,
                                  n_tokens_list=None,
                                  n_clusters_list=None,
                                  max_iter=200,
                                  n_init_values=None,
                                  ideal_num_per_id=3):
        """Grid search with evaluation and save best config
            Args: feat_pca - PCA reduced feature matrix
                    n_tokens_list - List of token counts to evaluate
                    n_clusters_list - List of cluster counts to evaluate
                    max_iter - Maximum iterations for KMeans
                    n_init_values - List of n_init values to evaluate
                    ideal_num_per_id - (assumed) Ideal average number of tracks per semantic ID
            Returns: best_semantic_ids - Semantic IDs from best config"""
        
        if n_tokens_list is None:
            n_tokens_list = [2, 3]
        if n_clusters_list is None:
            n_clusters_list = [32, 64, 128]
        if n_init_values is None:
            n_init_values = [16]
        
        results = []
        all_configs = []

        for K in n_clusters_list:
            for L in n_tokens_list:
                for n_init in n_init_values:
                    print(f"Testing L={L}, K={K}, n_init={n_init}")

                    # Generate semantic IDs
                    semantic_ids, codebooks, codes = self.assign_sem_ids_kmeans(
                        feat_pca, n_tokens=L, n_clusters=K, max_iter=max_iter, n_init=n_init
                    )

                    # Evaluate quality
                    vc = pd.Series(semantic_ids).value_counts()
                    unique_ids = int(vc.size)
                    mean_per_id = float(vc.mean())
                    singleton_pct = float((vc == 1).mean())
                    
                    # KNN recall evaluation
                    Xn = feat_pca / (np.linalg.norm(feat_pca, axis=1, keepdims=True) + 1e-12)
                    recall = self._evaluate_recall(codes, Xn, L)
                    
                    result = {
                        'L': L, 'K': K, 'n_init': n_init,
                        'UniqueIDs': unique_ids, 'MeanPerID': mean_per_id,
                        'SingletonPct': singleton_pct, 'Recall': recall
                    }
                    results.append(result)
                    all_configs.append((semantic_ids, codebooks, codes, result))

        # Print results
        df_results = pd.DataFrame(results)
        print("\nGrid Search Results:")
        print(df_results.to_string(index=False))

        # Select best (high recall, low singleton%, balanced mean_per_id)
        best_idx = df_results.assign(
            score=df_results['Recall'] - df_results['SingletonPct'] - abs(df_results['MeanPerID'] - ideal_num_per_id) * 0.1
        )['score'].idxmax()
        
        best_config = all_configs[best_idx]
        best_semantic_ids, best_codebooks, best_codes, best_stats = best_config
        
        print(f"\nBest config: L={best_stats['L']}, K={best_stats['K']}")

        # Save best config
        data_dir = Path(self.dp.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for l, cb in enumerate(best_codebooks):
            joblib.dump(cb, data_dir / f"best_kmeans_level{l}.joblib")
        np.save(data_dir / f"best_kmeans_codes.npy", best_codes)
        
        return best_semantic_ids
    
    def assign_sem_ids_dl(self, feat_pca, n_nonzero_coefs, n_dict_components, max_iter, batch_size):
        """Dictionary learning semantic ID assignment"""
        dictionary, codes, _, _ = self.dlp.run_sparse_coding_minibatch(
            feat_pca, n_nonzero_coefs, n_dict_components, max_iter, batch_size
        )
        semantic_ids = self._codes_to_semantic_ids(codes)
        return semantic_ids, dictionary, codes
    
    def assign_sem_ids_dl_manual(self, feat_pca, n_nonzero_coefs=None, n_dict_components=None, max_iter=None, batch_size=None):
        """Save semantic IDs with manually specified config"""
        if n_nonzero_coefs is None: n_nonzero_coefs = self.dlp.n_nonzero_coefs
        if n_dict_components is None: n_dict_components = self.dlp.n_dict_components
        if max_iter is None: max_iter = self.dlp.max_iter
        if batch_size is None: batch_size = self.dlp.batch_size
        
        print(f"Generating semantic IDs with manual DL config: C={n_nonzero_coefs}, D={n_dict_components}, batch_size={batch_size}")

        # Generate semantic IDs
        semantic_ids, dictionary, codes = self.assign_sem_ids_dl(
            feat_pca, n_nonzero_coefs, n_dict_components, max_iter, batch_size
        )

        # Evaluate quality
        vc = pd.Series(semantic_ids).value_counts()
        print(f"Manual DL config results: Unique IDs: {vc.size}, Mean per ID: {vc.mean():.2f}, Singleton %: {(vc == 1).mean():.1%}")

        # Save config
        data_dir = Path(self.dp.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(dictionary, data_dir / "best_dl_dictionary.joblib")
        np.save(data_dir / "best_dl_codes.npy", codes)
        config = {'n_nonzero_coefs': n_nonzero_coefs, 'n_dict_components': n_dict_components}
        joblib.dump(config, data_dir / "best_dl_config.joblib")
        
        print(f"Saved manual DL config to {data_dir}")
        return semantic_ids
    
    def assign_sem_ids_dl_manual_nomood(self, feat_pca, n_nonzero_coefs=None, n_dict_components=None, max_iter=None, batch_size=None):
        """Save semantic IDs with manually specified config"""
        if n_nonzero_coefs is None: n_nonzero_coefs = self.dlp.n_nonzero_coefs
        if n_dict_components is None: n_dict_components = self.dlp.n_dict_components
        if max_iter is None: max_iter = self.dlp.max_iter
        if batch_size is None: batch_size = self.dlp.batch_size
        
        print(f"Generating semantic IDs with manual DL config: C={n_nonzero_coefs}, D={n_dict_components}, batch_size={batch_size}")

        # Generate semantic IDs
        semantic_ids, dictionary, codes = self.assign_sem_ids_dl(
            feat_pca, n_nonzero_coefs, n_dict_components, max_iter, batch_size
        )

        # Evaluate quality
        vc = pd.Series(semantic_ids).value_counts()
        print(f"Manual DL config results: Unique IDs: {vc.size}, Mean per ID: {vc.mean():.2f}, Singleton %: {(vc == 1).mean():.1%}")

        # Save config
        data_dir = Path(self.dp.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(dictionary, data_dir / "best_dl_dictionary_nomood.joblib")
        np.save(data_dir / "best_dl_codes_nomood.npy", codes)
        config = {'n_nonzero_coefs': n_nonzero_coefs, 'n_dict_components': n_dict_components}
        joblib.dump(config, data_dir / "best_dl_config_nomood.joblib")
        
        print(f"Saved manual DL config to {data_dir}")
        return semantic_ids
    
    def assign_sem_ids_dl_load(self, feat_pca):
        """Load saved DL dictionary and apply to new data"""

        # Load saved config
        dict_path = Path(self.dp.output_dir) / "data/best_dl_dictionary.joblib"
        config_path = Path(self.dp.output_dir) / "data/best_dl_config.joblib"
        if not dict_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"No saved DL config found in {self.dp.output_dir}")
        dictionary = joblib.load(dict_path)
        config = joblib.load(config_path)
        n_nonzero_coefs, n_dict_components = config['n_nonzero_coefs'], config['n_dict_components']
        print(f"Loaded saved DL config: C={n_nonzero_coefs}, D={n_dict_components}")
        
        print("Applying loaded dictionary")
        sparse_codes = sparse_encode(feat_pca, dictionary, algorithm='omp', n_nonzero_coefs=n_nonzero_coefs, n_jobs=-1)

        # Apply
        codes = np.zeros((feat_pca.shape[0], n_nonzero_coefs), dtype=np.int32)
        for i in tqdm(range(feat_pca.shape[0]), desc="Generating IDs from loaded dict"):
            coef = sparse_codes[i, :]
            nonzero_indices = np.nonzero(coef)[0]
            sorted_indices = sorted(nonzero_indices, key=lambda j: abs(coef[j]), reverse=True)
            top_indices = sorted_indices[:n_nonzero_coefs]
            for k, idx in enumerate(top_indices):
                sign = np.sign(coef[idx])
                codes[i, k] = idx if sign >= 0 else idx + n_dict_components

        # Generate semantic IDs
        semantic_ids = self._codes_to_semantic_ids(codes)
        return semantic_ids
    
    def search_best_dl(self, feat_pca,
                       n_nonzero_coefs_list=None, n_dict_components_list=None,
                       batch_size_list=None, max_iter=200, ideal_num_per_id=3):
        """Grid search with evaluation and save best config
            Args: feat_pca - PCA reduced feature matrix
                    n_nonzero_coefs_list - List of token counts to evaluate
                    n_dict_components_list - List of cluster counts to evaluate
                    max_iter - Maximum iterations for KMeans
                    batch_size_list - List of batch sizes to evaluate
                    ideal_num_per_id - (assumed) Ideal average number of tracks per semantic ID
            Returns: best_semantic_ids - Semantic IDs from best config"""
        
        if n_nonzero_coefs_list is None:
            n_nonzero_coefs_list = [2, 3]
        if n_dict_components_list is None:
            n_dict_components_list = [32, 64, 128]
        if batch_size_list is None:
            batch_size_list = [256]
        
        results = []
        all_configs = []

        for D in n_dict_components_list:
            for C in n_nonzero_coefs_list:
                for B in batch_size_list:
                    print(f"Testing DL: C={C}, D={D}, batch_size={B}")

                    # Generate semantic IDs
                    semantic_ids, dictionary, codes = self.assign_sem_ids_dl(
                        feat_pca, n_nonzero_coefs=C, n_dict_components=D, max_iter=max_iter, batch_size=B
                    )

                    # Evaluate quality
                    vc = pd.Series(semantic_ids).value_counts()
                    unique_ids = int(vc.size)
                    mean_per_id = float(vc.mean())
                    singleton_pct = float((vc == 1).mean())

                    # KNN recall evaluation
                    Xn = feat_pca / (np.linalg.norm(feat_pca, axis=1, keepdims=True) + 1e-12)
                    recall = self._evaluate_recall(codes, Xn, C)

                    result = {'C': C, 'D': D, 'batch_size': B, 'UniqueIDs': unique_ids,
                              'MeanPerID': mean_per_id, 'SingletonPct': singleton_pct, 'Recall': recall}
                    results.append(result)
                    all_configs.append((semantic_ids, dictionary, codes, result))

        # Print results
        df_results = pd.DataFrame(results)
        print("\nGrid Search Results (Dictionary Learning):")
        print(df_results.to_string(index=False))

        # Select best (high recall, low singleton%, balanced mean_per_id)
        best_idx = df_results.assign(
            score=df_results['Recall'] - df_results['SingletonPct'] - abs(df_results['MeanPerID'] - ideal_num_per_id) * 0.1
        )['score'].idxmax()
        
        best_semantic_ids, best_dictionary, best_codes, best_stats = all_configs[best_idx]

        print(f"\nBest DL config: C={best_stats['C']}, D={best_stats['D']}")
        
        data_dir = Path(self.dp.output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_dictionary, data_dir / "best_dl_dictionary.joblib")
        np.save(data_dir / f"best_dl_codes.npy", best_codes)
        config = {'n_nonzero_coefs': best_stats['C'], 'n_dict_components': best_stats['D']}
        joblib.dump(config, data_dir / "best_dl_config.joblib")
        
        return best_semantic_ids
    
    def _evaluate_recall(self, codes, Xn, L, n_queries=50, k_eval=10):
        """Recall evaluation"""
        # Build inverted index
        inv = [{t: np.where(codes[:, l] == t)[0] for t in np.unique(codes[:, l])} for l in range(L)]
        
        rng = np.random.default_rng(42)
        qs = rng.choice(len(Xn), size=min(n_queries, len(Xn)), replace=False)
        
        recalls = []
        for q in qs:
            # Ground truth KNN
            sims = Xn @ Xn[q]; sims[q] = -np.inf
            gt_idx = np.argpartition(-sims, k_eval)[:k_eval]
            gt = set(gt_idx)
            
            # RVQ candidates
            cands = np.unique(np.concatenate([inv[l].get(codes[q, l], []) for l in range(L)]))
            cands = cands[cands != q]
            
            if len(cands) > 0:
                cand_sims = Xn[cands] @ Xn[q]
                top_cands = set(cands[np.argsort(-cand_sims)[:k_eval]])
                recalls.append(len(gt & top_cands) / k_eval)
            else:
                recalls.append(0.0)
        
        return float(np.mean(recalls))
    
    def integrate_with_dataset(self, df, semantic_ids):
        """integrate semantic IDs with original dataset"""
        df_integrated = df.copy()
        df_integrated['semantic_id'] = semantic_ids[:len(df)]
        return df_integrated
    
    def print_stats(self, semantic_ids):
        """Print statistics of generated semantic IDs
            Args
                semantic_ids (list): List of generated semantic IDs
        """
        vc = pd.Series(semantic_ids).value_counts()
        print("Unique IDs:", vc.size)
        print("Mean tracks per ID:", vc.mean())
        print("Median tracks per ID:", vc.median())
        print("IDs with only 1 track:", (vc==1).sum(), f"({(vc==1).mean():.1%})")

        print("\nTop 10 most common IDs:")
        print(vc.head(10).to_frame("count"))

    def _cluster_purity(self, labels, clusters):
        dfc = pd.DataFrame({"label": labels, "cluster": clusters})
        dfc = dfc[dfc["label"].astype(str).str.len() > 0].copy()
        if dfc.empty:
            return dict(micro=0.0, macro=0.0)
        
        groups = dfc.groupby("cluster")["label"]
        per_cluster_purity = groups.apply(lambda s: s.value_counts(normalize=True).iloc[0])
        cluster_counts = groups.size()
        
        micro_purity = float((per_cluster_purity * cluster_counts).sum() / cluster_counts.sum())
        macro_purity = float(per_cluster_purity.mean())
        
        return dict(micro=micro_purity, macro=macro_purity)

    def _neighbor_purity_at_k(self, X_features, labels, k):
        labels = np.array(labels)
        has_label_mask = np.array([isinstance(s, str) and s != "" for s in labels])
        X_filtered, labels_filtered = X_features[has_label_mask], labels[has_label_mask]
        
        if len(labels_filtered) <= k:
            return dict(micro=0.0, macro=0.0)
            
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_filtered)
        _, indices = nbrs.kneighbors(X_filtered)
        indices = indices[:, 1:]
        
        micro_scores, per_class_scores = [], {}
        for i, neighbor_indices in enumerate(indices):
            neighbor_labels = labels_filtered[neighbor_indices]
            score = Counter(neighbor_labels).most_common(1)[0][1] / k
            micro_scores.append(score)
            per_class_scores.setdefault(labels_filtered[i], []).append(score)
            
        micro_purity_at_k = float(np.mean(micro_scores))
        macro_purity_at_k = float(np.mean([np.mean(v) for v in per_class_scores.values()])) if per_class_scores else 0.0
        
        return dict(micro=micro_purity_at_k, macro=macro_purity_at_k)

    def evaluate_id_quality(self):
        """
        Evaluate semantic ID uality using purity
        """
        print("Evaluating semantic ID purity")
        
        data_csv_path = Path(self.dp.output_dir) / "data/dataset_with_semantic_ids.csv"
        features_pca_path = Path(self.dp.output_dir) / "data/features_pca.npy"
        
        try:
            df_eval = pd.read_csv(data_csv_path)
            X_emb = np.load(features_pca_path)
            print(f"Loaded dataset from: {data_csv_path}")
            print(f"Loaded PCA features from: {features_pca_path}")
        except FileNotFoundError as e:
            print(f"Error: Could not find required file. {e}")
            print("Please run the semantic ID generation process first.")
            return

        tag_cols = ["genre", "mood", "instrument"]
        df_eval[tag_cols] = df_eval[tag_cols].fillna("")
        
        original_rows = len(X_emb)
        
        if len(df_eval) != original_rows:
            min_rows = min(len(X_emb), len(df_eval))
            X_emb = X_emb[:min_rows]
            df_eval = df_eval.iloc[:min_rows]

        for tag_col in tag_cols:
            if tag_col not in df_eval.columns:
                continue
            
            labels = df_eval[tag_col].values
            clusters = df_eval["semantic_id"].values

            cp = self._cluster_purity(labels, clusters)
            print(f"\n--- {tag_col.upper()} ---")
            print(f"Cluster Purity (micro/macro): {cp['micro']:.3f} | {cp['macro']:.3f}")

            p20 = self._neighbor_purity_at_k(X_emb, labels, 20)
            print(f"Purity@20 (micro/macro):    {p20['micro']:.3f} | {p20['macro']:.3f}")

    def evaluate_id_quality_nomood(self):
        """
        Evaluate semantic ID uality using purity
        """
        print("Evaluating semantic ID purity")
        
        data_csv_path = Path(self.dp.output_dir) / "data/dataset_with_semantic_ids_nomood.csv"
        features_pca_path = Path(self.dp.output_dir) / "data/features_pca_nomood.npy"
        
        try:
            df_eval = pd.read_csv(data_csv_path)
            X_emb = np.load(features_pca_path)
            print(f"Loaded dataset from: {data_csv_path}")
            print(f"Loaded PCA features from: {features_pca_path}")
        except FileNotFoundError as e:
            print(f"Error: Could not find required file. {e}")
            print("Please run the semantic ID generation process first.")
            return

        tag_cols = ["genre", "instrument"]
        df_eval[tag_cols] = df_eval[tag_cols].fillna("")
        
        original_rows = len(X_emb)
        
        if len(df_eval) != original_rows:
            min_rows = min(len(X_emb), len(df_eval))
            X_emb = X_emb[:min_rows]
            df_eval = df_eval.iloc[:min_rows]

        for tag_col in tag_cols:
            if tag_col not in df_eval.columns:
                continue
            
            labels = df_eval[tag_col].values
            clusters = df_eval["semantic_id"].values

            cp = self._cluster_purity(labels, clusters)
            print(f"\n--- {tag_col.upper()} ---")
            print(f"Cluster Purity (micro/macro): {cp['micro']:.3f} | {cp['macro']:.3f}")

            p20 = self._neighbor_purity_at_k(X_emb, labels, 20)
            print(f"Purity@20 (micro/macro):    {p20['micro']:.3f} | {p20['macro']:.3f}")

def main():
    sig = SemanticIDGenerator()
    
    # Load PCA features
    pca_path = Path(sig.dp.output_dir) / "data/features_pca.npy"
    feat_pca = np.load(pca_path)

    # Load dataset
    csv_path = Path(sig.dp.output_dir) / "data/features_2tags.csv"
    df = pd.read_csv(csv_path)
    
    # k-means
    # Grid search and get best semantic IDs
    print("===== K-means =====")
    ### We can change the parameters of grid search
    ### And we have to keep them
    best_semantic_ids = sig.search_best_kmean(feat_pca)
    
    # Integrate with dataset
    df_final = sig.integrate_with_dataset(df, best_semantic_ids)

    # Save final result
    df_final.to_csv(Path(sig.dp.output_dir) / "data/dataset_with_semantic_ids.csv", index=False)
    print(f"\nSaved final dataset with semantic IDs")
    print(df_final.head())



    # Dictionary learning
    print("\n===== Dictionary Learning =====")
    best_semantic_ids = sig.search_best_dl(feat_pca)

    # Integrate with dataset
    df_final = sig.integrate_with_dataset(df, best_semantic_ids)

    # Save final result
    df_final.to_csv(Path(sig.dp.output_dir) / "data/dataset_with_semantic_ids.csv", index=False)
    print(f"\nSaved final dataset with semantic IDs")
    print(df_final.head())

    # Print statistics of semantic IDs
    sig.print_stats(best_semantic_ids)

    sig.evaluate_id_quality()

def main_nomood():
    sig = SemanticIDGenerator()
    
    # Load PCA features
    pca_path = Path(sig.dp.output_dir) / "data/features_pca_nomood.npy"
    feat_pca = np.load(pca_path)

    # Load dataset
    csv_path = Path(sig.dp.output_dir) / "data/features_2tags_nomood.csv"
    df = pd.read_csv(csv_path)
    
    # # k-means
    # # Grid search and get best semantic IDs
    # print("===== K-means =====")
    # ### We can change the parameters of grid search
    # ### And we have to keep them
    # best_semantic_ids = sig.search_best_kmean(feat_pca)
    
    # # Integrate with dataset
    # df_final = sig.integrate_with_dataset(df, best_semantic_ids)

    # # Save final result
    # df_final.to_csv(Path(sig.dp.output_dir) / "data/dataset_with_semantic_ids.csv", index=False)
    # print(f"\nSaved final dataset with semantic IDs")
    # print(df_final.head())



    # Dictionary learning
    print("\n===== Dictionary Learning =====")
    # best_semantic_ids = sig.search_best_dl(feat_pca)
    semantic_ids = sig.assign_sem_ids_dl_manual_nomood(
        feat_pca,
        n_nonzero_coefs=2,
        n_dict_components=16,
        max_iter=200,
        batch_size=256
    )

    # Integrate with dataset
    df_final = sig.integrate_with_dataset(df, semantic_ids)

    # Save final result
    df_final.to_csv(Path(sig.dp.output_dir) / "data/dataset_with_semantic_ids_nomood.csv", index=False)
    print(f"\nSaved final dataset with semantic IDs")
    print(df_final.head())

    # Print statistics of semantic IDs
    sig.print_stats(semantic_ids)

    sig.evaluate_id_quality_nomood()

if __name__ == "__main__":
    # main()
    main_nomood()