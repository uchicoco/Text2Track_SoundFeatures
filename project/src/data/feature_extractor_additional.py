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
    fe = FeatureExtractorAdditional()
    feat = fe.generate_dictionary(
        row["json_full"],
        **kwargs
    )
    if feat:
        feat["track_id"] = row["track_id"]
        feat["path"] = row["path"]
        return feat
    return None

### Extended Feature Extractor with ALL AcousticBrainz features ###
class FeatureExtractorAdditional:
    def __init__(self):
        self.dp = DatasetProcessor()
        self.chromatic_scale = ('c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b')
        self.fifths_map = ('c', 'g', 'd', 'a', 'e', 'b', 'f#', 'c#', 'g#', 'd#', 'a#', 'f')
    
    def record_json_path(self, json_path):
        rec = {'json_full': json_path}
        return rec

    # ========== EXISTING FEATURES ==========
    def extract_mfcc(self, j, rec):
        """MFCC 13 coefficients (mean + std) = 26 features"""
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

    def extract_mel(self, j, rec):
        """Mel bands 40 bands (mean + std) = 80 features"""
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

    def extract_spectral_centroid(self, j, rec):
        """Spectral centroid (mean) = 1 feature"""
        sc = self.dp._get(j, "lowlevel.spectral_centroid.mean")
        if sc is not None:
            rec["spec_centroid_mean_0"] = float(sc)
        return rec

    def extract_spectral_energy(self, j, rec):
        """Spectral energy 4 bands = 4 features"""
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
 
    def extract_spectral_contrast(self, j, rec):
        """Spectral contrast 6 bands = 6 features"""
        sc_coeffs = self.dp._get(j, "lowlevel.spectral_contrast_coeffs.mean")
        if sc_coeffs is not None:
            sc_coeffs = np.asarray(sc_coeffs, float).ravel()
            for i, v in enumerate(sc_coeffs):
                rec[f"spec_contrast_mean_{i}"] = float(v)
        return rec

    def extract_spectral_valley(self, j, rec):
        """Spectral valley 6 bands = 6 features"""
        sc_valleys = self.dp._get(j, "lowlevel.spectral_contrast_valleys.mean")
        if sc_valleys is not None:
            sc_valleys = np.asarray(sc_valleys, float).ravel()
            for i, v in enumerate(sc_valleys):
                rec[f"spec_valley_mean_{i}"] = float(v)
        return rec

    def extract_hpcp(self, j, rec):
        """HPCP 36 bands = 36 features"""
        hpcp = self.dp._get(j, "tonal.hpcp.mean")
        if hpcp is not None:
            hpcp = np.asarray(hpcp, float).ravel()
            for i, v in enumerate(hpcp):
                rec[f"hpcp_mean_{i}"] = float(v)
        return rec

    def extract_tonality(self, j, rec):
        """Tonality vector (x, y, z) = 3 features"""
        key = self.dp._get(j, "tonal.key_key")
        scale = self.dp._get(j, "tonal.key_scale")
        strength = self.dp._get(j, "tonal.key_strength")
        if all(x is not None for x in (key, scale, strength)):
            key = key.lower()
            if scale == 'minor':
                minor_index = self.chromatic_scale.index(key)
                major_index = (minor_index + 3) % 12
                key = self.chromatic_scale[major_index]
            fifths_index = self.fifths_map.index(key)
            angle = (fifths_index * 2 * np.pi) / 12
            x = np.cos(angle)
            y = np.sin(angle)
            z = 0.5 if scale == 'major' else -0.5
            v = strength * np.array([x, y, z])
            rec["key_vector_x"] = v[0]
            rec["key_vector_y"] = v[1]
            rec["key_vector_z"] = v[2]
        return rec
    
    # ========== PRIORITY HIGH: NEW FEATURES ==========
    
    def extract_bpm_features(self, j, rec):
        """BPM and related rhythm features = 7 features"""
        bpm = self.dp._get(j, "rhythm.bpm")
        if bpm is not None:
            if isinstance(bpm, (list, tuple)) and len(bpm) > 0:
                rec["bpm"] = float(bpm[0])
            elif not isinstance(bpm, dict):
                rec["bpm"] = float(bpm)
        
        danceability = self.dp._get(j, "rhythm.danceability")
        if danceability is not None:
            if isinstance(danceability, (list, tuple)) and len(danceability) > 0:
                rec["danceability"] = float(danceability[0])
            elif not isinstance(danceability, dict):
                rec["danceability"] = float(danceability)
        
        beats_count = self.dp._get(j, "rhythm.beats_count")
        if beats_count is not None:
            if isinstance(beats_count, (list, tuple)) and len(beats_count) > 0:
                rec["beats_count"] = float(beats_count[0])
            elif not isinstance(beats_count, dict):
                rec["beats_count"] = float(beats_count)
        
        # BPM histogram peaks (can be dict with statistics or scalar)
        # If dict: use median (more robust than mean for peaks)
        first_peak_bpm = self.dp._get(j, "rhythm.bpm_histogram_first_peak_bpm")
        if first_peak_bpm is not None:
            if isinstance(first_peak_bpm, dict):
                # Prefer median for peak values (more robust)
                if "median" in first_peak_bpm:
                    rec["bpm_hist_first_peak_bpm"] = float(first_peak_bpm["median"])
                elif "mean" in first_peak_bpm:
                    rec["bpm_hist_first_peak_bpm"] = float(first_peak_bpm["mean"])
            elif isinstance(first_peak_bpm, (list, tuple)):
                # If array, take first element (main peak)
                if len(first_peak_bpm) > 0:
                    rec["bpm_hist_first_peak_bpm"] = float(first_peak_bpm[0])
            else:
                rec["bpm_hist_first_peak_bpm"] = float(first_peak_bpm)
        
        first_peak_weight = self.dp._get(j, "rhythm.bpm_histogram_first_peak_weight")
        if first_peak_weight is not None:
            if isinstance(first_peak_weight, dict):
                if "median" in first_peak_weight:
                    rec["bpm_hist_first_peak_weight"] = float(first_peak_weight["median"])
                elif "mean" in first_peak_weight:
                    rec["bpm_hist_first_peak_weight"] = float(first_peak_weight["mean"])
            elif isinstance(first_peak_weight, (list, tuple)):
                if len(first_peak_weight) > 0:
                    rec["bpm_hist_first_peak_weight"] = float(first_peak_weight[0])
            else:
                rec["bpm_hist_first_peak_weight"] = float(first_peak_weight)
        
        second_peak_bpm = self.dp._get(j, "rhythm.bpm_histogram_second_peak_bpm")
        if second_peak_bpm is not None:
            if isinstance(second_peak_bpm, dict):
                if "median" in second_peak_bpm:
                    rec["bpm_hist_second_peak_bpm"] = float(second_peak_bpm["median"])
                elif "mean" in second_peak_bpm:
                    rec["bpm_hist_second_peak_bpm"] = float(second_peak_bpm["mean"])
            elif isinstance(second_peak_bpm, (list, tuple)):
                if len(second_peak_bpm) > 0:
                    rec["bpm_hist_second_peak_bpm"] = float(second_peak_bpm[0])
            else:
                rec["bpm_hist_second_peak_bpm"] = float(second_peak_bpm)
        
        second_peak_weight = self.dp._get(j, "rhythm.bpm_histogram_second_peak_weight")
        if second_peak_weight is not None:
            if isinstance(second_peak_weight, dict):
                if "median" in second_peak_weight:
                    rec["bpm_hist_second_peak_weight"] = float(second_peak_weight["median"])
                elif "mean" in second_peak_weight:
                    rec["bpm_hist_second_peak_weight"] = float(second_peak_weight["mean"])
            elif isinstance(second_peak_weight, (list, tuple)):
                if len(second_peak_weight) > 0:
                    rec["bpm_hist_second_peak_weight"] = float(second_peak_weight[0])
            else:
                rec["bpm_hist_second_peak_weight"] = float(second_peak_weight)
        
        return rec
    
    def extract_spectral_shape(self, j, rec):
        """Spectral shape features (mean + var) = 14 features"""
        # Rolloff
        rolloff = self.dp._get(j, "lowlevel.spectral_rolloff")
        if isinstance(rolloff, dict):
            if "mean" in rolloff:
                rec["spec_rolloff_mean"] = float(rolloff["mean"])
            if "var" in rolloff:
                rec["spec_rolloff_var"] = float(rolloff["var"])
        
        # Flux
        flux = self.dp._get(j, "lowlevel.spectral_flux")
        if isinstance(flux, dict):
            if "mean" in flux:
                rec["spec_flux_mean"] = float(flux["mean"])
            if "var" in flux:
                rec["spec_flux_var"] = float(flux["var"])
        
        # Spread
        spread = self.dp._get(j, "lowlevel.spectral_spread")
        if isinstance(spread, dict):
            if "mean" in spread:
                rec["spec_spread_mean"] = float(spread["mean"])
            if "var" in spread:
                rec["spec_spread_var"] = float(spread["var"])
        
        # Skewness
        skewness = self.dp._get(j, "lowlevel.spectral_skewness")
        if isinstance(skewness, dict):
            if "mean" in skewness:
                rec["spec_skewness_mean"] = float(skewness["mean"])
            if "var" in skewness:
                rec["spec_skewness_var"] = float(skewness["var"])
        
        # Kurtosis
        kurtosis = self.dp._get(j, "lowlevel.spectral_kurtosis")
        if isinstance(kurtosis, dict):
            if "mean" in kurtosis:
                rec["spec_kurtosis_mean"] = float(kurtosis["mean"])
            if "var" in kurtosis:
                rec["spec_kurtosis_var"] = float(kurtosis["var"])
        
        # Decrease
        decrease = self.dp._get(j, "lowlevel.spectral_decrease")
        if isinstance(decrease, dict):
            if "mean" in decrease:
                rec["spec_decrease_mean"] = float(decrease["mean"])
            if "var" in decrease:
                rec["spec_decrease_var"] = float(decrease["var"])
        
        # Entropy
        entropy = self.dp._get(j, "lowlevel.spectral_entropy")
        if isinstance(entropy, dict):
            if "mean" in entropy:
                rec["spec_entropy_mean"] = float(entropy["mean"])
            if "var" in entropy:
                rec["spec_entropy_var"] = float(entropy["var"])
        
        return rec
    
    def extract_loudness_dynamics(self, j, rec):
        """Loudness and dynamics features = 5 features"""
        avg_loudness = self.dp._get(j, "lowlevel.average_loudness")
        if avg_loudness is not None:
            if isinstance(avg_loudness, (list, tuple)) and len(avg_loudness) > 0:
                rec["avg_loudness"] = float(avg_loudness[0])
            elif not isinstance(avg_loudness, dict):
                rec["avg_loudness"] = float(avg_loudness)
        
        dyn_complexity = self.dp._get(j, "lowlevel.dynamic_complexity")
        if dyn_complexity is not None:
            if isinstance(dyn_complexity, (list, tuple)) and len(dyn_complexity) > 0:
                rec["dynamic_complexity"] = float(dyn_complexity[0])
            elif not isinstance(dyn_complexity, dict):
                rec["dynamic_complexity"] = float(dyn_complexity)
        
        rms = self.dp._get(j, "lowlevel.spectral_rms")
        if isinstance(rms, dict):
            if "mean" in rms:
                rec["spec_rms_mean"] = float(rms["mean"])
            if "var" in rms:
                rec["spec_rms_var"] = float(rms["var"])
        
        zcr = self.dp._get(j, "lowlevel.zerocrossingrate")
        if isinstance(zcr, dict):
            if "mean" in zcr:
                rec["zerocrossingrate_mean"] = float(zcr["mean"])
        
        return rec
    
    def extract_pitch_harmonic(self, j, rec):
        """Pitch and harmonic features = 6 features"""
        pitch_sal = self.dp._get(j, "lowlevel.pitch_salience")
        if isinstance(pitch_sal, dict):
            if "mean" in pitch_sal:
                rec["pitch_salience_mean"] = float(pitch_sal["mean"])
            if "var" in pitch_sal:
                rec["pitch_salience_var"] = float(pitch_sal["var"])
        
        dissonance = self.dp._get(j, "lowlevel.dissonance")
        if isinstance(dissonance, dict):
            if "mean" in dissonance:
                rec["dissonance_mean"] = float(dissonance["mean"])
            if "var" in dissonance:
                rec["dissonance_var"] = float(dissonance["var"])
        
        hfc = self.dp._get(j, "lowlevel.hfc")
        if isinstance(hfc, dict):
            if "mean" in hfc:
                rec["hfc_mean"] = float(hfc["mean"])
            if "var" in hfc:
                rec["hfc_var"] = float(hfc["var"])
        
        return rec
    
    def extract_chords(self, j, rec):
        """Chord features = 3 features"""
        chords_changes = self.dp._get(j, "tonal.chords_changes_rate")
        if chords_changes is not None:
            if isinstance(chords_changes, (list, tuple)) and len(chords_changes) > 0:
                rec["chords_changes_rate"] = float(chords_changes[0])
            elif not isinstance(chords_changes, dict):
                rec["chords_changes_rate"] = float(chords_changes)
        
        chords_strength = self.dp._get(j, "tonal.chords_strength")
        if isinstance(chords_strength, dict):
            if "mean" in chords_strength:
                rec["chords_strength_mean"] = float(chords_strength["mean"])
            if "var" in chords_strength:
                rec["chords_strength_var"] = float(chords_strength["var"])
        
        return rec
    
    def extract_hpcp_entropy(self, j, rec):
        """HPCP entropy = 2 features"""
        hpcp_ent = self.dp._get(j, "tonal.hpcp_entropy")
        if isinstance(hpcp_ent, dict):
            if "mean" in hpcp_ent:
                rec["hpcp_entropy_mean"] = float(hpcp_ent["mean"])
            if "var" in hpcp_ent:
                rec["hpcp_entropy_var"] = float(hpcp_ent["var"])
        return rec
    
    # ========== PRIORITY MEDIUM: ADDITIONAL FEATURES ==========
    
    def extract_silence(self, j, rec):
        """Silence rate at different thresholds = 3 features"""
        sil_20 = self.dp._get(j, "lowlevel.silence_rate_20dB")
        if isinstance(sil_20, dict) and "mean" in sil_20:
            rec["silence_rate_20dB"] = float(sil_20["mean"])
        
        sil_30 = self.dp._get(j, "lowlevel.silence_rate_30dB")
        if isinstance(sil_30, dict) and "mean" in sil_30:
            rec["silence_rate_30dB"] = float(sil_30["mean"])
        
        sil_60 = self.dp._get(j, "lowlevel.silence_rate_60dB")
        if isinstance(sil_60, dict) and "mean" in sil_60:
            rec["silence_rate_60dB"] = float(sil_60["mean"])
        
        return rec
    
    def extract_spectral_complexity(self, j, rec):
        """Spectral complexity = 2 features"""
        spec_comp = self.dp._get(j, "lowlevel.spectral_complexity")
        if isinstance(spec_comp, dict):
            if "mean" in spec_comp:
                rec["spec_complexity_mean"] = float(spec_comp["mean"])
            if "var" in spec_comp:
                rec["spec_complexity_var"] = float(spec_comp["var"])
        return rec
    
    def extract_spectral_strongpeak(self, j, rec):
        """Spectral strong peak = 2 features"""
        strongpeak = self.dp._get(j, "lowlevel.spectral_strongpeak")
        if isinstance(strongpeak, dict):
            if "mean" in strongpeak:
                rec["spec_strongpeak_mean"] = float(strongpeak["mean"])
            if "var" in strongpeak:
                rec["spec_strongpeak_var"] = float(strongpeak["var"])
        return rec
    
    def extract_beats_loudness(self, j, rec):
        """Beat loudness statistics = 2 features"""
        beat = self.dp._get(j, "rhythm.beats_loudness")
        if isinstance(beat, dict):
            if "mean" in beat:
                rec["beat_loudness_mean"] = float(beat["mean"])
            if "var" in beat:
                rec["beat_loudness_var"] = float(beat["var"])
        return rec
    
    def extract_onset_rate(self, j, rec):
        """Onset rate = 1 feature"""
        onset = self.dp._get(j, "rhythm.onset_rate")
        if onset is not None:
            if isinstance(onset, (list, tuple)) and len(onset) > 0:
                rec["onset_rate"] = float(onset[0])
            elif not isinstance(onset, dict):
                rec["onset_rate"] = float(onset)
        return rec
    
    def extract_thpcp(self, j, rec):
        """Tonal HPCP 36 bands = 36 features"""
        thpcp = self.dp._get(j, "tonal.thpcp")
        if isinstance(thpcp, dict) and "mean" in thpcp:
            thpcp_mean = np.asarray(thpcp["mean"], float).ravel()
            for i, v in enumerate(thpcp_mean):
                rec[f"thpcp_mean_{i}"] = float(v)
        return rec
    
    def extract_chords_number_rate(self, j, rec):
        """Chords number rate = 1 feature"""
        chords_num = self.dp._get(j, "tonal.chords_number_rate")
        if chords_num is not None:
            if isinstance(chords_num, (list, tuple)) and len(chords_num) > 0:
                rec["chords_number_rate"] = float(chords_num[0])
            elif not isinstance(chords_num, dict):
                rec["chords_number_rate"] = float(chords_num)
        return rec
    
    # ========== PRIORITY LOW: FREQUENCY BAND STATISTICS ==========
    
    def extract_barkbands_stats(self, j, rec):
        """Bark bands statistics = 10 features"""
        features = ["crest", "flatness_db", "kurtosis", "skewness", "spread"]
        for feat in features:
            data = self.dp._get(j, f"lowlevel.barkbands_{feat}")
            if isinstance(data, dict):
                if "mean" in data:
                    rec[f"barkbands_{feat}_mean"] = float(data["mean"])
                if "var" in data:
                    rec[f"barkbands_{feat}_var"] = float(data["var"])
        return rec
    
    def extract_erbbands_stats(self, j, rec):
        """ERB bands statistics = 10 features"""
        features = ["crest", "flatness_db", "kurtosis", "skewness", "spread"]
        for feat in features:
            data = self.dp._get(j, f"lowlevel.erbbands_{feat}")
            if isinstance(data, dict):
                if "mean" in data:
                    rec[f"erbbands_{feat}_mean"] = float(data["mean"])
                if "var" in data:
                    rec[f"erbbands_{feat}_var"] = float(data["var"])
        return rec
    
    def extract_melbands_stats(self, j, rec):
        """Mel bands statistics = 10 features"""
        features = ["crest", "flatness_db", "kurtosis", "skewness", "spread"]
        for feat in features:
            data = self.dp._get(j, f"lowlevel.melbands_{feat}")
            if isinstance(data, dict):
                if "mean" in data:
                    rec[f"melbands_{feat}_mean"] = float(data["mean"])
                if "var" in data:
                    rec[f"melbands_{feat}_var"] = float(data["var"])
        return rec
    
    def extract_tuning(self, j, rec):
        """Tuning features = 3 features"""
        tuning_freq = self.dp._get(j, "tonal.tuning_frequency")
        if tuning_freq is not None:
            if isinstance(tuning_freq, (list, tuple)) and len(tuning_freq) > 0:
                rec["tuning_frequency"] = float(tuning_freq[0])
            elif not isinstance(tuning_freq, dict):
                rec["tuning_frequency"] = float(tuning_freq)
        
        tuning_diatonic = self.dp._get(j, "tonal.tuning_diatonic_strength")
        if tuning_diatonic is not None:
            if isinstance(tuning_diatonic, (list, tuple)) and len(tuning_diatonic) > 0:
                rec["tuning_diatonic_strength"] = float(tuning_diatonic[0])
            elif not isinstance(tuning_diatonic, dict):
                rec["tuning_diatonic_strength"] = float(tuning_diatonic)
        
        tuning_deviation = self.dp._get(j, "tonal.tuning_equal_tempered_deviation")
        if tuning_deviation is not None:
            if isinstance(tuning_deviation, (list, tuple)) and len(tuning_deviation) > 0:
                rec["tuning_equal_tempered_deviation"] = float(tuning_deviation[0])
            elif not isinstance(tuning_deviation, dict):
                rec["tuning_equal_tempered_deviation"] = float(tuning_deviation)
        
        return rec
    
    def extract_gfcc(self, j, rec):
        """GFCC (Gammatone Frequency Cepstral Coefficients) = 26 features"""
        gfcc = self.dp._get(j, "lowlevel.gfcc")
        if isinstance(gfcc, dict):
            if "mean" in gfcc:
                gfcc_mean = np.asarray(gfcc["mean"], float).ravel()
                for i, v in enumerate(gfcc_mean):
                    rec[f"gfcc_mean_{i}"] = float(v)
            if "cov" in gfcc:
                gfcc_std = np.sqrt(np.diag(np.asarray(gfcc["cov"], float)))
                for i, v in enumerate(gfcc_std):
                    rec[f"gfcc_std_{i}"] = float(v)
        return rec
    
    # ========== MAIN EXTRACTION FUNCTION ==========
    
    def generate_dictionary(
        self,
        json_path: str,
        use_mel_alt: bool = False,
        use_energy_alt: bool = False,
        extract_all: bool = True  # New parameter to extract ALL features
    ):
        """
        Extract features from AcousticBrainz JSON
        
        Args:
            json_path: Path to JSON file
            use_mel_alt: Use mel bands instead of MFCC
            use_energy_alt: Use spectral energy instead of centroid
            extract_all: If True, extract ALL available features
        
        Returns:
            Dictionary of features
        """
        try:
            with open(json_path, "r") as f:
                j = json.load(f)
        except Exception:
            return None
        
        rec = self.record_json_path(json_path)

        # Core features (mutually exclusive)
        if use_mel_alt:
            rec = self.extract_mel(j, rec)
        else:
            rec = self.extract_mfcc(j, rec)

        if use_energy_alt:
            rec = self.extract_spectral_energy(j, rec)
        else:
            rec = self.extract_spectral_centroid(j, rec)

        # Standard features
        rec = self.extract_spectral_contrast(j, rec)
        rec = self.extract_spectral_valley(j, rec)
        rec = self.extract_hpcp(j, rec)
        rec = self.extract_tonality(j, rec)
        
        if extract_all:
            # Priority HIGH
            rec = self.extract_bpm_features(j, rec)
            rec = self.extract_spectral_shape(j, rec)
            rec = self.extract_loudness_dynamics(j, rec)
            rec = self.extract_pitch_harmonic(j, rec)
            rec = self.extract_chords(j, rec)
            rec = self.extract_hpcp_entropy(j, rec)
            
            # Priority MEDIUM
            rec = self.extract_silence(j, rec)
            rec = self.extract_spectral_complexity(j, rec)
            rec = self.extract_spectral_strongpeak(j, rec)
            rec = self.extract_beats_loudness(j, rec)
            rec = self.extract_onset_rate(j, rec)
            rec = self.extract_thpcp(j, rec)
            rec = self.extract_chords_number_rate(j, rec)
            
            # Priority LOW
            rec = self.extract_barkbands_stats(j, rec)
            rec = self.extract_erbbands_stats(j, rec)
            rec = self.extract_melbands_stats(j, rec)
            rec = self.extract_tuning(j, rec)
            rec = self.extract_gfcc(j, rec)
        
        return rec if len(rec) > 1 else None
    
    def build_features_dataframe(self, tags_merged, use_mel_alt=False, use_energy_alt=False, 
                                extract_all=True):
        """
        Extract features from all tracks and merge with tags.
        
        Args: 
            tags_merged: merged tags dataframe
            use_mel_alt: use mel bands instead of mfcc
            use_energy_alt: use spectral energy instead of spectral centroid
            extract_all: extract ALL available features
        Returns: 
            pd.DataFrame: features + tags
        """
        dp = self.dp
        tags_merged["json_full"] = tags_merged["path"].apply(dp.resolve_json_full)
        
        print("JSON coverage for tagged tracks:",
          f"{tags_merged['json_full'].notna().mean()*100:.1f}%")
        
        kwargs = dict(
            use_mel_alt=use_mel_alt,
            use_energy_alt=use_energy_alt,
            extract_all=extract_all
        )
        
        rows = []
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
    
    def filter_min1tags(self, df):
        """Filter to tracks with at least 1 non-empty tag"""
        cols = ["genre", "mood", "instrument"]
        mask = (df[cols].ne("").sum(axis=1) >= 1)
        df_one = df.loc[mask].reset_index(drop=True)

        N, K = len(df), len(df_one)
        print(f"Kept {K}/{N} rows ({(100*K/N):.1f}%) with >=1 tag categories.")
        return df_one

def main():
    from pathlib import Path
    
    fe = FeatureExtractorAdditional()
    dp = fe.dp
    
    mood_df  = dp.load_tag_tsv("autotagging_moodtheme.tsv", "mood")
    genre_df = dp.load_tag_tsv("autotagging_genre.tsv", "genre")
    inst_df  = dp.load_tag_tsv("autotagging_instrument.tsv", "instrument")

    tags_merged = (mood_df
               .merge(genre_df, on=["track_id","path"], how="outer")
               .merge(inst_df,  on=["track_id","path"], how="outer"))
    
    tags_merged[["genre","mood","instrument"]] = tags_merged[["genre","mood","instrument"]].fillna("")
    
    df = fe.build_features_dataframe(tags_merged, extract_all=True)
        
    out_csv = Path(dp.output_dir) / "data/features_all_comprehensive.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Saved CSV in", out_csv)

    df_one = fe.filter_min1tags(df)
    
    out_csv_one = Path(dp.output_dir) / "data/features_all_comprehensive_1tag.csv"
    df_one.to_csv(out_csv_one, index=False)
    print("Saved CSV in", out_csv_one)

    with pd.option_context("display.max_columns", None):
        print(df_one.head())
    
    print(f"\nTotal features extracted: {len(df_one.columns) - 5}")  # Exclude track_id, path, json_full, genre, mood, instrument

if __name__ == "__main__":
    main()
