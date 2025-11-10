from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .dataset_processor import DatasetProcessor
except ImportError:
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.data.dataset_processor import DatasetProcessor

# For parallel processing
def _extract_row(args):
    # Args: (row_dict, kwargs)
    row, kwargs = args
    fe = FeatureExtractorNoMood()
    feat = fe.generate_dictionary(
        row["json_full"],
        **kwargs
    )
    if feat:
        feat["track_id"] = row["track_id"]
        feat["path"] = row["path"]
        return feat
    return None

class FeatureExtractorNoMood:
    def __init__(self):
        self.dp = DatasetProcessor()
        self.chromatic_scale = ('c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b')
        self.fifths_map = ('c', 'g', 'd', 'a', 'e', 'b', 'f#', 'c#', 'g#', 'd#', 'a#', 'f')
    # Absolutely use
    def record_json_path(self, json_path):
        """
        Args: json_path (str): path to acousticbrainz json file
        Returns: Dict["json_full", str]: dictionary with key "json_full" and value json_path
        """
        rec = {'json_full': json_path}
        return rec

    # GFCC mean [0, 1, 3, 5, 7, 11]
    def extract_gfcc(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]*6 : dictionary of extracted features appended to rec
        """
        idx = [0, 1, 3, 5, 7, 11]
        gfcc = self.dp._get(j, "lowlevel.gfcc")
        if isinstance(gfcc, dict):
            if "mean" in gfcc:
                gm = np.asarray(gfcc["mean"], float)
                for i in idx:
                    rec[f"gfcc_mean_{i}"] = float(gm[i])
        return rec

    # Spectral valley mean [0, 3, 4, 5]
    def extract_spectral_valley(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]*4: dictionary of extracted features appended to rec
        """
        idx = [0, 3, 4, 5]
        sc_valleys = self.dp._get(j, "lowlevel.spectral_contrast_valleys.mean")
        if sc_valleys is not None:
            sc_valleys = np.asarray(sc_valleys, float).ravel()
            for i in idx:
                rec[f"spec_valley_mean_{i}"] = float(sc_valleys[i])
        return rec
    
    # Spectral flux mean
    def extract_spectral_flux(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        sf = self.dp._get(j, "lowlevel.spectral_flux.mean")
        if sf is not None:
            rec["spec_flux_mean"] = float(sf)
        return rec
    
    # Spectral spread mean
    def extract_spectral_spread(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        ssp = self.dp._get(j, "lowlevel.spectral_spread.mean")
        if ssp is not None:
            rec["spec_spread_mean"] = float(ssp)
        return rec
    
    # Spectral skewness mean
    def extract_spectral_skewness(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        ssk = self.dp._get(j, "lowlevel.spectral_skewness.mean")
        if ssk is not None:
            rec["spec_skewness_mean"] = float(ssk)
        return rec
    
    # Spectral kurtosis mean
    def extract_spectral_kurtosis(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        sk = self.dp._get(j, "lowlevel.spectral_kurtosis.mean")
        if sk is not None:
            rec["spec_kurtosis_mean"] = float(sk)
        return rec
    
    # Spectral RMS mean
    def extract_spectral_rms(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        srms = self.dp._get(j, "lowlevel.spectral_rms.mean")
        if srms is not None:
            rec["spec_rms_mean"] = float(srms)
        return rec
    
    # Erabbands spread mean
    def extract_erb_spread(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        es = self.dp._get(j, "lowlevel.erbbands_spread.mean")
        if es is not None:
            rec["erb_spread_mean"] = float(es)
        return rec
    
    # Erabbands flatness mean
    def extract_erb_flatness(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        ef = self.dp._get(j, "lowlevel.erbbands_flatness_db.mean")
        if ef is not None:
            rec["erb_flatness_mean"] = float(ef)
        return rec

    # Erabbands crest mean
    def extract_erb_crest(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        ec = self.dp._get(j, "lowlevel.erbbands_crest.mean")
        if ec is not None:
            rec["erb_crest_mean"] = float(ec)
        return rec
    
    # Dissonance mean
    def extract_dissonance(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        dis = self.dp._get(j, "lowlevel.dissonance.mean")
        if dis is not None:
            rec["dissonance_mean"] = float(dis)
        return rec
    
    # HPCP entropy mean
    def extract_hpcp_entropy(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        he = self.dp._get(j, "tonal.hpcp_entropy.mean")
        if he is not None:
            rec["hpcp_entropy_mean"] = float(he)
        return rec
    
    # Average loudness
    def extract_avg_loudness(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        avgl = self.dp._get(j, "lowlevel.average_loudness")
        if avgl is not None:
            rec["avg_loudness"] = float(avgl)
        return rec

    # Danceability
    def extract_danceability(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        danc = self.dp._get(j, "rhythm.danceability")
        if danc is not None:
            rec["danceability"] = float(danc)
        return rec

    # Spectral centroid mean
    def extract_spectral_centroid(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        sc = self.dp._get(j, "lowlevel.spectral_centroid.mean")
        if sc is not None:
            rec["spec_centroid_mean"] = float(sc)
        return rec
    
    # Specral complexity var
    def extract_spectral_complexity(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        scomp = self.dp._get(j, "lowlevel.spectral_complexity.var")
        if scomp is not None:
            rec["spec_comp_var"] = float(scomp)
        return rec
    
    # Specral entropy mean
    def extract_spectral_entropy(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        se = self.dp._get(j, "lowlevel.spectral_entropy.mean")
        if se is not None:
            rec["spec_entropy_mean"] = float(se)
        return rec
    
    # Tuning
    def extract_tuning(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        tn = self.dp._get(j, "tonal.tuning_equal_tempered_deviation")
        if tn is not None:
            rec["tuning"] = float(tn)
        return rec
    
    def generate_dictionary(
        self,
        json_path: str,
    ):
        """
        Args:
            json_path (str): Path to the AcousticBrainz JSON file.
        Returns:
            A dictionary containing the extracted features for one track, or None if an error occurs.
        """
        try:
            with open(json_path, "r") as f:
                j = json.load(f)
        except Exception:
            return None
        
        rec = self.record_json_path(json_path)
        rec = self.extract_gfcc(j, rec)
        rec = self.extract_spectral_valley(j, rec)
        rec = self.extract_spectral_flux(j, rec)
        rec = self.extract_spectral_spread(j, rec)
        rec = self.extract_spectral_skewness(j, rec)
        rec = self.extract_spectral_kurtosis(j, rec)
        rec = self.extract_spectral_rms(j, rec)
        rec = self.extract_erb_spread(j, rec)
        rec = self.extract_erb_flatness(j, rec)
        rec = self.extract_erb_crest(j, rec)
        rec = self.extract_hpcp_entropy(j, rec)
        rec = self.extract_dissonance(j, rec)
        rec = self.extract_avg_loudness(j, rec)
        rec = self.extract_danceability(j, rec)
        rec = self.extract_spectral_centroid(j, rec)
        rec = self.extract_spectral_complexity(j, rec)
        rec = self.extract_spectral_entropy(j, rec)
        rec = self.extract_tuning(j, rec)

        return rec if len(rec) > 1 else None
    
    def build_features_dataframe(self, tags_merged):
        """
        Extract features from all tracks and merge with tags.
        
        Args: tags_merged (pd.DataFrame): merged tags dataframe with columns ['track_id', 'path', 'genre', 'instrument']
               use_mel_alt (bool): use mel bands instead of mfcc
               use_energy_alt (bool): use spectral energy instead of spectral centroid
               add_valley (bool): add spectral valley features
               add_timbre_dist (bool): add barkbands skewness/kurtosis
               add_tonality (bool): add key/scale/strength
               add_rhythm_struct (bool): add rhythm structure features
        Returns: pd.DataFrame: features + tags (genre, instrument)
        """
        
        dp = self.dp
        
        tags_merged["json_full"] = tags_merged["path"].apply(dp.resolve_json_full)
        
        print("JSON coverage for tagged tracks:",
          f"{tags_merged['json_full'].notna().mean()*100:.1f}%")
        
        rows = []
        # Parallel processing
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(_extract_row, (r, {}))
                for _, r in tags_merged.iterrows()
            ]
            for f in tqdm(as_completed(futures), total=len(futures)):
                res = f.result()
                if res:
                    rows.append(res)


        features_df = pd.DataFrame(rows)
        print("features_df shape:", features_df.shape)
        
        df = features_df.merge(
            tags_merged[["track_id","path","genre","instrument"]],
            on=["track_id","path"],
            how="left",
            validate="one_to_one"
        )
        
        def pct(n, d): return f"{(100*n/d):.1f}%" if d else "n/a"
        n = len(df)
        has_g = df["genre"].astype(str).str.len().gt(0).sum()
        has_i = df["instrument"].astype(str).str.len().gt(0).sum()
        print(f"Rows: {n} | genre: {has_g} ({pct(has_g,n)}) | instrument: {has_i} ({pct(has_i,n)})")
        
        return df
    
    def filter_min2tags(self, df):
        """
        Args: df (pd.DataFrame): DataFrame with columns ['genre', 'instrument']
        Returns: pd.DataFrame: Filtered DataFrame with at least 2 non-empty tag
        """
        cols = ["genre", "instrument"]
        mask = (df[cols].ne("").sum(axis=1) >= 2)
        df_two = df.loc[mask].reset_index(drop=True)

        N, K = len(df), len(df_two)
        print(f"Kept {K}/{N} rows ({(100*K/N):.1f}%) with >=2 tag categories.")
        print("Non-empty counts in kept set:",
              "genre =", (df_two["genre"] != "").sum(),
              "instrument =", (df_two["instrument"] != "").sum())
        return df_two

    def check_tag_distribution(self, df_two, top=20):
        """
        Args: df_two (pd.DataFrame): DataFrame with at least 2 non-empty tags
               top (int): number of top tags to display
        """
        # Check uniqueness
        assert df_two["path"].is_unique, "Duplicate paths detected!"

        # Count rows with 2 non-empty tags
        cat_nonempty = (
            (df_two["genre"].astype(str).str.len() > 0).astype(int) +
            (df_two["instrument"].astype(str).str.len() > 0).astype(int)
        )
        print("Rows with exactly 2 categories:", (cat_nonempty == 2).sum())

        # Top tags
        def top_counts(series, top=top):
            vc = series[series.astype(str).str.len() > 0].value_counts()
            return vc.head(top)
        print("\nTop genres:")
        print(top_counts(df_two["genre"]))
        print("\nTop instruments:")
        print(top_counts(df_two["instrument"]))
    
def main():
    from pathlib import Path
    
    fe = FeatureExtractorNoMood()
    dp = fe.dp

    genre_df = dp.load_tag_tsv("autotagging_genre.tsv", "genre")
    inst_df  = dp.load_tag_tsv("autotagging_instrument.tsv", "instrument")

    tags_merged = (genre_df
               .merge(inst_df,  on=["track_id","path"], how="outer"))
    
    tags_merged[["genre","instrument"]] = tags_merged[["genre","instrument"]].fillna("")
    
    df = fe.build_features_dataframe(tags_merged)
        
    out_csv = Path(fe.dp.output_dir) / "data/features_tags_nomood.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Saved CSV in", out_csv)

    df_two = fe.filter_min2tags(df)
    fe.check_tag_distribution(df_two)

    out_csv_two = Path(fe.dp.output_dir) / "data/features_2tags_nomood.csv"
    out_csv_two.parent.mkdir(parents=True, exist_ok=True)
    df_two.to_csv(out_csv_two, index=False)
    print("Saved CSV in", out_csv_two)

    with pd.option_context("display.max_columns", None):
        print(df_two.head())

if __name__ == "__main__":
    main()