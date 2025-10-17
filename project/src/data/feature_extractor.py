from concurrent.futures import ProcessPoolExecutor, as_completed
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from .dataset_processor import DatasetProcessor

# For parallel processing
def _extract_row(args):
    # Args: (row_dict, kwargs)
    row, kwargs = args
    fe = FeatureExtractor()
    feat = fe.generate_dictionary(
        row["json_full"],
        **kwargs
    )
    if feat:
        feat["track_id"] = row["track_id"]
        feat["path"] = row["path"]
        return feat
    return None

### Referenced to Teodor's code!! Thank you very much!!###
class FeatureExtractor:
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

    # Default use (13 mfcc mean & std)
    def extract_mfcc(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]*26 : dictionary of extracted features appended to rec
        """
        mfcc = self.dp._get(j, "lowlevel.mfcc")
        if isinstance(mfcc, dict):
            if "mean" in mfcc:
                mm = np.asarray(mfcc["mean"], float)
                for i, v in enumerate(mm):
                    rec[f"mfcc_mean_{i}"] = float(v)
            if "cov" in mfcc:
                sd = np.sqrt(np.diag(np.asarray(mfcc["cov"], float)))
                for i, v in enumerate(sd):
                    rec[f"mfcc_std_{i}"] = float(v)
        return rec

    # Alternatively use (40 melbands mean & std)
    def extract_mel(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]*80 : dictionary of extracted features appended to rec
        """
        mel = self.dp._get(j, "lowlevel.melbands")
        if isinstance(mel, dict):
            if "mean" in mel:
                mm = np.asarray(mel["mean"], float)
                for i, v in enumerate(mm):
                    rec[f"mel_mean_{i}"] = float(v)
            if "var" in mel:
                sd = np.sqrt(np.asarray(mel["var"], float))
                for i, v in enumerate(sd):
                    rec[f"mel_std_{i}"] = float(v)
        return rec

    # Default use (spectral centroid)
    def extract_spectral_centroid(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        sc = self.dp._get(j, "lowlevel.spectral_centroid.mean")
        if sc is not None:
            rec["spec_centroid_mean_0"] = float(sc)
        return rec

    # Alternatively use (spectral energy 4 bands)
    def extract_spectral_energy(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]*4: dictionary of extracted features appended to rec
        """
        high = self.dp._get(j, "lowlevel.spectral_energyband_high.mean")
        if high is not None:
            rec["spec_energy_0"] = float(high)
        low = self.dp._get(j, "lowlevel.spectral_energyband_low.mean")
        if low is not None:
            rec["spec_energy_1"] = float(low)
        mid_h = self.dp._get(j, "lowlevel.spectral_energyband_middle_high.mean")
        if mid_h is not None:
            rec["spec_energy_2"] = float(mid_h)
        mid_l = self.dp._get(j, "lowlevel.spectral_energyband_middle_low.mean")
        if mid_l is not None:
            rec["spec_energy_3"] = float(mid_l)
        return rec
 
    # Default use (spectral contrast 6 bands)
    def extract_spectral_contrast(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]*6: dictionary of extracted features appended to rec
        """
        sc_coeffs = self.dp._get(j, "lowlevel.spectral_contrast_coeffs.mean")
        if sc_coeffs is not None:
            sc_coeffs = np.asarray(sc_coeffs, float).ravel()
            for i, v in enumerate(sc_coeffs):
                rec[f"spec_contrast_mean_{i}"] = float(v)
        return rec

    # Optional use (spectral valley 6 bands)
    def extract_spectral_valley(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]*6: dictionary of extracted features appended to rec
        """
        sc_valleys = self.dp._get(j, "lowlevel.spectral_contrast_valleys.mean")
        if sc_valleys is not None:
            sc_valleys = np.asarray(sc_valleys, float).ravel()
            for i, v in enumerate(sc_valleys):
                rec[f"spec_valley_mean_{i}"] = float(v)
        return rec

    # Default use (HPCP 36 bands)
    def extract_hpcp(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]*36: dictionary of extracted features appended to rec
        """
        hpcp = self.dp._get(j, "tonal.hpcp.mean")
        if hpcp is not None:
            hpcp = np.asarray(hpcp, float).ravel()
            for i, v in enumerate(hpcp):
                rec[f"hpcp_mean_{i}"] = float(v)
        return rec

    # Default use (tempo in bpm)
    def extract_tempo_bpm(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        bpm = self.dp._get(j, "rhythm.bpm")
        if bpm is not None:
            rec["tempo_bpm"] = float(bpm)
        return rec

    # Optional use (barkbands skewness and kurtosis)<-- distribution of timbre
    def extract_barkbands(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]*2: dictionary of extracted features appended to rec
        """
        skewness = self.dp._get(j, "lowlevel.barkbands_skewness.mean")
        if skewness is not None:
            rec["barkbands_skewness"] = float(skewness)
        kurtosis = self.dp._get(j, "lowlevel.barkbands_kurtosis.mean")
        if kurtosis is not None:
            rec["barkbands_kurtosis"] = float(kurtosis)
        return rec

    # Optional use (tonality)<-- simplified tonnetz (more distinctive major or minor)
    # We should investigate Tonal Interval Space??
    def extract_tonality(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: Dict[str, (np.ndarray(3))]: dictionary of extracted features appended to rec
        """
        key = self.dp._get(j, "tonal.key_key")
        scale = self.dp._get(j, "tonal.key_scale")
        strength = self.dp._get(j, "tonal.key_strength")
        if all(x is not None for x in (key, scale, strength)):
            key = key.lower()
            if scale == 'minor': # Find relative major
                minor_index = self.chromatic_scale.index(key)
                major_index = (minor_index + 3) % 12
                key = self.chromatic_scale[major_index]
            # Calculate angle
            fifths_index = self.fifths_map.index(key)
            angle = (fifths_index * 2 * np.pi) / 12

            # Calculate coordinates
            x = np.cos(angle)
            y = np.sin(angle)
            z = 0.5 if scale == 'major' else -0.5

            # Calculate final vector
            v = strength * np.array([x, y, z])
        rec["key_vector_x"] = v[0]
        rec["key_vector_y"] = v[1]
        rec["key_vector_z"] = v[2]
        return rec
    
    # Optional use (onset rate)<-- density of notes
    def extract_onset_rate(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]: dictionary of extracted features appended to rec
        """
        onset = self.dp._get(j, "rhythm.onset_rate")
        if onset is not None:
            rec["onset_rate"] = float(onset)
        return rec
    
    # Optional use (beat loudness)<-- dynamics of the piece
    def extract_beat_loudness(self, j, rec):
        """
        Args: j (Dict[str, Any]): loaded json data
               rec (Dict[str, Any]): existing record to append features to
        Returns: +Dict[str, float]*2: dictionary of extracted features appended to rec
        """
        beat = self.dp._get(j, "rhythm.beats_loudness")
        if isinstance(beat, dict):
            if "mean" in beat:
                bm = beat["mean"]
                rec["beat_loudness_mean"] = float(bm)
            if "var" in beat:
                sd = np.sqrt(beat["var"])
                rec["beat_loudness_std"] = float(sd)
        return rec
    
    def generate_dictionary(
        self,
        json_path: str,
        use_mel_alt: bool = False,
        use_energy_alt: bool = False,
        add_valley: bool = False,
        add_timbre_dist: bool = False,
        add_tonality: bool = False,
        add_rhythm_struct: bool = False
    ):
        """
        Args:
            json_path (str): Path to the AcousticBrainz JSON file.
            feature_extractor (FeatureExtractor): An instance of the FeatureExtractor class.
            use_mel_alt (bool): If True, use melband stats instead of MFCCs.
            use_energy_alt (bool): If True, use 4-band energy instead of spectral centroid.
            add_valley (bool): If True, add spectral valley information.
            add_timbre_dist (bool): If True, add timbre distribution stats (barkbands).
            add_tonality (bool): If True, add high-level tonality features.
            add_rhythm_struct (bool): If True, add rhythm structure features.

        Returns:
            A dictionary containing the extracted features for one track, or None if an error occurs.
        """
        try:
            with open(json_path, "r") as f:
                j = json.load(f)
        except Exception:
            return None
        
        rec = self.record_json_path(json_path)

        if use_mel_alt:
            rec = self.extract_mel(j, rec)
        else:
            rec = self.extract_mfcc(j, rec)

        if use_energy_alt:
            rec = self.extract_spectral_energy(j, rec)
        else:
            rec = self.extract_spectral_centroid(j, rec)

        rec = self.extract_spectral_contrast(j, rec)
        if add_valley:
            rec = self.extract_spectral_valley(j, rec)
        
        rec = self.extract_hpcp(j, rec)
        rec = self.extract_tempo_bpm(j, rec)
        
        if add_timbre_dist:
            rec = self.extract_barkbands(j, rec)
            
        if add_tonality:
            rec = self.extract_tonality(j, rec)
        
        if add_rhythm_struct:
            rec = self.extract_onset_rate(j, rec)
            rec = self.extract_beat_loudness(j, rec)
        
        return rec if len(rec) > 1 else None
    
    def build_features_dataframe(self, tags_merged, use_mel_alt=False, use_energy_alt=False, 
                                add_valley=False, add_timbre_dist=False, 
                                add_tonality=False, add_rhythm_struct=False):
        """
        Extract features from all tracks and merge with tags.
        
        Args: tags_merged (pd.DataFrame): merged tags dataframe with columns ['track_id', 'path', 'genre', 'mood', 'instrument']
               use_mel_alt (bool): use mel bands instead of mfcc
               use_energy_alt (bool): use spectral energy instead of spectral centroid
               add_valley (bool): add spectral valley features
               add_timbre_dist (bool): add barkbands skewness/kurtosis
               add_tonality (bool): add key/scale/strength
               add_rhythm_struct (bool): add rhythm structure features
        Returns: pd.DataFrame: features + tags (genre, mood, instrument)
        """
        
        dp = self.dp
        
        tags_merged["json_full"] = tags_merged["path"].apply(dp.resolve_json_full)
        
        print("JSON coverage for tagged tracks:",
          f"{tags_merged['json_full'].notna().mean()*100:.1f}%")
        
        kwargs = dict(
            use_mel_alt=use_mel_alt,
            use_energy_alt=use_energy_alt,
            add_valley=add_valley,
            add_timbre_dist=add_timbre_dist,
            add_tonality=add_tonality,
            add_rhythm_struct=add_rhythm_struct
        )
        
        rows = []
        # Parallel processing
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(_extract_row, (r, kwargs))
                for _, r in tags_merged.iterrows()
            ]
            for f in tqdm(as_completed(futures), total=len(futures)):
                res = f.result()
                if res:
                    rows.append(res)


        features_df = pd.DataFrame(rows)
        print("features_df shape:", features_df.shape)
        
        df = features_df.merge(
            tags_merged[["track_id","path","genre","mood","instrument"]],
            on=["track_id","path"],
            how="left",
            validate="one_to_one"
        )
        
        def pct(n, d): return f"{(100*n/d):.1f}%" if d else "n/a"
        n = len(df)
        has_g = df["genre"].astype(str).str.len().gt(0).sum()
        has_m = df["mood"].astype(str).str.len().gt(0).sum()
        has_i = df["instrument"].astype(str).str.len().gt(0).sum()
        print(f"Rows: {n} | genre: {has_g} ({pct(has_g,n)}) | mood: {has_m} ({pct(has_m,n)}) | instrument: {has_i} ({pct(has_i,n)})")
        
        return df
    
    def filter_min2tags(self, df):
        """
        Args: df (pd.DataFrame): DataFrame with columns ['genre', 'mood', 'instrument']
        Returns: pd.DataFrame: Filtered DataFrame with at least 2 non-empty tag
        """
        cols = ["genre", "mood", "instrument"]
        mask = (df[cols].ne("").sum(axis=1) >= 2)
        df_two = df.loc[mask].reset_index(drop=True)

        N, K = len(df), len(df_two)
        print(f"Kept {K}/{N} rows ({(100*K/N):.1f}%) with >=2 tag categories.")
        print("Non-empty counts in kept set:",
              "genre =", (df_two["genre"] != "").sum(),
              "mood =", (df_two["mood"] != "").sum(),
              "instrument =", (df_two["instrument"] != "").sum())
        return df_two

    def check_tag_distribution(self, df_two, top=20):
        """
        Args: df_two (pd.DataFrame): DataFrame with at least 2 non-empty tags
               top (int): number of top tags to display
        """
        # Check uniqueness
        assert df_two["path"].is_unique, "Duplicate paths detected!"

        # Count rows with 2 or 3 non-empty tags
        cat_nonempty = (
            (df_two["genre"].astype(str).str.len() > 0).astype(int) +
            (df_two["mood"].astype(str).str.len() > 0).astype(int) +
            (df_two["instrument"].astype(str).str.len() > 0).astype(int)
        )
        print("Rows with exactly 2 categories:", (cat_nonempty == 2).sum())
        print("Rows with all 3 categories:", (cat_nonempty == 3).sum())

        # Top tags
        def top_counts(series, top=top):
            vc = series[series.astype(str).str.len() > 0].value_counts()
            return vc.head(top)
        print("\nTop genres:")
        print(top_counts(df_two["genre"]))
        print("\nTop moods:")
        print(top_counts(df_two["mood"]))
        print("\nTop instruments:")
        print(top_counts(df_two["instrument"]))
    
def main():
    from pathlib import Path
    
    fe = FeatureExtractor()
    dp = fe.dp
    
    mood_df  = dp.load_tag_tsv("autotagging_moodtheme.tsv", "mood")
    genre_df = dp.load_tag_tsv("autotagging_genre.tsv", "genre")
    inst_df  = dp.load_tag_tsv("autotagging_instrument.tsv", "instrument")

    tags_merged = (mood_df
               .merge(genre_df, on=["track_id","path"], how="outer")
               .merge(inst_df,  on=["track_id","path"], how="outer"))
    
    tags_merged[["genre","mood","instrument"]] = tags_merged[["genre","mood","instrument"]].fillna("")
    
    df = fe.build_features_dataframe(tags_merged, add_tonality=True)
        
    out_csv = Path(fe.dp.output_dir) / "data/features_tags.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Saved CSV in", out_csv)

    df_two = fe.filter_min2tags(df)
    fe.check_tag_distribution(df_two)

    out_csv_two = Path(fe.dp.output_dir) / "data/features_2tags.csv"
    out_csv_two.parent.mkdir(parents=True, exist_ok=True)
    df_two.to_csv(out_csv_two, index=False)
    print("Saved CSV in", out_csv_two)

    with pd.option_context("display.max_columns", None):
        print(df_two.head())

if __name__ == "__main__":
    main()