from networkx import display
import pandas as pd
from pathlib import Path

from dataset_processor import DatasetProcessor
from feature_extractor import FeatureExtractor
from pca_processor import PCAProcessor
from semantic_id_generator import SemanticIDGenerator
from t5_processor import T5Processor

def main():
    # load and prepare data
    print("Loading dataset")
    dp = DatasetProcessor()
    mood_df  = dp.load_tag_tsv("autotagging_moodtheme.tsv", "mood")
    genre_df = dp.load_tag_tsv("autotagging_genre.tsv", "genre")
    inst_df  = dp.load_tag_tsv("autotagging_instrument.tsv", "instrument")
    print("Finish loading")
    print("\nHead of mood_df:", mood_df.head())

    # extract features and save df to csv
    fe = FeatureExtractor()

    tags_merged = (mood_df
               .merge(genre_df, on=["track_id","path"], how="outer")
               .merge(inst_df,  on=["track_id","path"], how="outer"))
    
    tags_merged[["genre","mood","instrument"]] = tags_merged[["genre","mood","instrument"]].fillna("")
    
    #####
    # we can change features extracted here
    # all False is the default setting
    df = fe.build_features_dataframe(tags_merged,
                                     use_mel_alt = False,
                                     use_energy_alt = False,
                                     add_valley = False,
                                     add_timbre_dist = False,
                                     add_tonality = False,
                                     add_rhythm_struct = False)
    #####

    df_two = fe.filter_min2tags(df)
    fe.check_tag_distribution(df_two)

    out_csv_two = Path(fe.dp.output_dir) / "data/features_2tags.csv"
    out_csv_two.parent.mkdir(parents=True, exist_ok=True)
    df_two.to_csv(out_csv_two, index=False)
    print("Saved CSV in", out_csv_two)
    print("Head of df_2tags:", df_two.head())

    # run PCA
    print("\nRunning PCA")
    pp = PCAProcessor()
    feat_matrix = pp.build_feature_matrix(df_two)

    #####
    # we can change either n_components or r_explained_var here
    # feat_pca, cum_var, expl_var, pca = pp.run_pca_components(feat_matrix, n_components=32) # dedicated number of components
    feat_pca, cum_var, expl_var, pca = pp.run_pca_ratio(feat_matrix, r_explained_var=0.95) # dedicated explained variance ratio
    #####
    
    pp.save_feat_pca(feat_pca, expl_var)

    # if you want, you can plot the result of pca
    # pp.plot_pca_with_tags(feat_pca, df_two)
    # pp.plot_explained_variance(expl_var, cum_var)

    sig = SemanticIDGenerator()

    # we can chose the method for generating semantic IDs here
    ##### kmeans #####
    # we have to change the parameters of RVQ here
    # semantic_ids = sig.assign_sem_ids_kmean_manual(feat_pca,
    #                                                n_tokens=2,
    #                                                n_clusters=32,
    #                                                max_iter=200,
    #                                                n_init=16)
    
    # we can search the best parameters here
    # semantic_ids = sig.search_best_kmean(feat_pca,
    #                                     n_tokens_list=[2,3],
    #                                     n_clusters_list=[32,64,128],
    #                                     max_iter=200,
    #                                     n_init_values=[16],
    #                                     ideal_num_per_id=3)

    # we can use the saved best parameters here
    # semantic_ids = sig.assign_kmeans_ids_kmean_load(feat_pca)
    #####
    ##### dictionary learning #####
    # we have to change the parameters of dictionary learning here
    # semantic_ids = sig.assign_sem_ids_dl_manual(feat_pca,
    #                                            n_nonzero_coefs=2,
    #                                            n_dict_components=32,
    #                                            max_iter=200,
    #                                            batch_size=256)
    # we can search the best parameters here
    semantic_ids = sig.search_best_dl(feat_pca,
                                    n_nonzero_coefs_list=[2,3],
                                    n_dict_components_list=[32,64,128],
                                    batch_size_list=[256],
                                    max_iter=200,
                                    ideal_num_per_id=3)
    # we can use the saved best parameters here
    # semantic_ids = sig.assign_dl_ids_dl_load(feat_pca)
    #####

    # show the stats of semantic IDs
    # sig.print_stats(semantic_ids)

    # integrate semantic IDs into the dataframe and save to csv
    df_ids = sig.integrate_with_dataset(df_two, semantic_ids)
    print("\nGenerated semantic IDs:", df_ids.head())

    # train T5 model
    t5p = T5Processor()
    train_df, val_df, test_df = t5p.prepare_data(df_ids=df_ids)
    print("\nPrompt-target pairs for T5 training/validation/testing")
    print("Train:", train_df.head())
    print("Validation:", val_df.head())
    print("Test:", test_df.head())

    tokenizer, model = t5p.train_t5(train_df, val_df)
    hits_at_10 = t5p.evaluate_t5(tokenizer=tokenizer, model=model, test_df=test_df)
    print(f"Hits@10: {hits_at_10:.4f}")
    t5p.visualize_umap(tokenizer=tokenizer, model=model, test_df=test_df)




if __name__ == "__main__":
    main()