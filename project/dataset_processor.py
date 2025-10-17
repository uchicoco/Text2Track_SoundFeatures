import numpy as np
import os
from pathlib import Path
import pandas as pd

### referred to Teodor's code!! Thank you very much!!###
class DatasetProcessor:
    def __init__(self):
        np.random.seed(42)

        # paths
        self.current_dir = Path(__file__).resolve().parent
        self.parent_dir = self.current_dir.parent

        self.output_dir = self.current_dir / 'outputs'
        self.acb_dir = self.current_dir / 'data/acousticbrainz_raw30s'
        self.jamendo_dir = self.parent_dir / 'mtg-jamendo-dataset' / 'data'

        # make outputs dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def get_paths(self):
        return self.current_dir, self.parent_dir, self.output_dir, self.acb_dir, self.jamendo_dir

    def load_tag_tsv(self, tsv_name: str, category: str):
        """
        Args: tsv_name (str): filename of tsv file
              category (str): 'mood', 'genre' or 'instrument'
        Returns: pd.DataFrame: with columns ['track_id', 'path', category]
        """
        # load tsv and make column names lowercase
        df = pd.read_csv(self.jamendo_dir / tsv_name, sep='\t', engine='python', on_bad_lines='skip')
        df.columns = [c.lower() for c in df.columns]

        # grab the single tag
        prefix = "mood/theme---" if category == "mood" else f"{category}---"
        def grab_tag(s):
            if not isinstance(s, str): return ""
            for tok in s.split():
                if tok.startswith(prefix):
                    return tok.split("---", 1)[1]
            return ""
        
        df[category] = df.get("tags", "").apply(grab_tag).fillna("")
        return df[["track_id", "path", category]].astype(str)
    
    def rel_mp3_to_json(self, rel):
        return rel.replace(".mp3", ".json")
    
    def resolve_json_full(self, rel):
        """
        Args: rel (str): relative path to mp3 file from jamendo_dir
        Returns: str or None: full path to corresponding json file in acb_dir, or None if not exists
        """
        p = self.acb_dir / self.rel_mp3_to_json(rel)
        if p.exists():
            return str(p)
        return None
    
    def _get(self, d, dotted):
        """
        Args: d (dict): nested dictionary
                dotted (str): key in dotted notation, e.g. "highlevel.mood_acoustic"
        Returns: value corresponding to the dotted key in the nested dictionary
        """
        cur = d
        for k in dotted.split("."):
            cur = cur[k]
        return cur
    
# for testing
def main():
    dp = DatasetProcessor()
    print("Current dir:", dp.current_dir)
    print("Parent dir:", dp.parent_dir)
    print("Output dir:", dp.output_dir)
    print("ACB dir:", dp.acb_dir)
    print("Jamendo dir:", dp.jamendo_dir)

    print("get_paths:", dp.get_paths())

    mood_df  = dp.load_tag_tsv("autotagging_moodtheme.tsv", "mood")
    genre_df = dp.load_tag_tsv("autotagging_genre.tsv", "genre")
    inst_df  = dp.load_tag_tsv("autotagging_instrument.tsv", "instrument")
    print(mood_df.head())
    print(genre_df.head())
    print(inst_df.head())

if __name__ == "__main__":
    main()