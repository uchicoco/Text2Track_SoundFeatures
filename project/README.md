# Music Informatics Project

Music feature extraction, semantic ID generation, and T5-based music recommendation system.

## 📁 Project Structure

```
project/
├── config/                  # Configuration files
│   ├── __init__.py
│   └── settings.py         # Project-wide settings
├── src/                    # Source code
│   ├── __init__.py
│   ├── data/              # Data processing modules
│   │   ├── __init__.py
│   │   ├── dataset_processor.py
│   │   └── feature_extractor.py
│   ├── models/            # Machine learning models
│   │   ├── __init__.py
│   │   ├── pca_processor.py
│   │   ├── kmeans_clusterer.py
│   │   └── dictionary_learning_processor.py
│   ├── semantic/          # Semantic ID generation
│   │   ├── __init__.py
│   │   └── semantic_id_generator.py
│   └── t5/               # T5 model
│       ├── __init__.py
│       ├── t5_processor.py
│       └── t5_utils.py
├── data/                  # Data files
│   └── acousticbrainz_raw30s/
├── outputs/               # Output files
│   ├── data/
│   └── figures/
├── models/                # Saved models
├── run_new.py            # Main execution script (new structure)
├── run.py                # Main execution script (old structure)
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place the MTG-Jamendo dataset and AcousticBrainz data:
- `../mtg-jamendo-dataset/data/` for Jamendo data
- `data/acousticbrainz_raw30s/` for AcousticBrainz JSON files

## 📊 Usage

### Basic Usage (New Structure)

```bash
python run_new.py
```

### Pipeline Overview

1. **Data Loading**: Retrieve tag information from MTG-Jamendo
2. **Feature Extraction**: Extract audio features from AcousticBrainz JSON
3. **PCA**: Dimensionality reduction (preserving 95% variance)
4. **Semantic ID Generation**: K-means or Dictionary Learning
5. **T5 Training**: Train on prompt-target pairs
6. **Evaluation**: Evaluate using Hits@10 metric

### Customization

#### Customize Feature Extraction

Edit the following section in `run_new.py`:

```python
df = fe.build_features_dataframe(
    tags_merged,
    use_mel_alt=False,        # Alternative mel spectrogram extraction
    use_energy_alt=False,     # Alternative energy feature extraction
    add_valley=False,         # Valley detection features
    add_timbre_dist=False,    # Timbre distribution features
    add_tonality=False,       # Tonality features
    add_rhythm_struct=False   # Rhythm structure features
)
```

#### Choose Semantic ID Generation Method

Toggle comments in `run_new.py`:

```python
# K-means with manual config
semantic_ids = sig.assign_sem_ids_kmean_manual(feat_pca, n_tokens=2, n_clusters=32)

# K-means with grid search
semantic_ids = sig.search_best_kmean(feat_pca)

# Dictionary Learning with manual config
semantic_ids = sig.assign_sem_ids_dl_manual(feat_pca, n_nonzero_coefs=2)

# Dictionary Learning with grid search (default)
semantic_ids = sig.search_best_dl(feat_pca)
```

## 📝 Configuration

Modify settings in `config/settings.py`:

- **Path Settings**: Data directories, output directories
- **PCA Settings**: Number of components, variance ratio
- **K-means Settings**: Number of tokens, clusters, iterations
- **Dictionary Learning Settings**: Non-zero coefficients, dictionary components
- **T5 Settings**: Model name, max length, upsampling threshold
- **Training Settings**: Batch size, learning rate, epochs

## 📦 Main Modules

### data module
- `DatasetProcessor`: Dataset loading and path management
- `FeatureExtractor`: Music feature extraction (MFCC, spectral, rhythm, etc.)

### models module
- `PCAProcessor`: Principal Component Analysis for dimensionality reduction
- `KMeansProcessor`: RVQ (Residual Vector Quantization)
- `DictionaryLearningProcessor`: Sparse coding

### semantic module
- `SemanticIDGenerator`: Semantic ID generation and evaluation

### t5 module
- `T5Processor`: T5 model training and evaluation
- `T5Dataset`: PyTorch dataset class

## 🔧 Troubleshooting

### Import Errors

For new structure, use `run_new.py`:

```bash
python run_new.py
```

For old structure (all files in root), use `run.py`:

```bash
python run.py
```

### Memory Issues

- Reduce number of PCA components
- Decrease batch size
- Filter dataset

## 📊 Output Files

### outputs/data/
- `features_2tags.csv`: Extracted features (tracks with 2+ tags)
- `features_pca.npy`: PCA-transformed features
- `dataset_with_semantic_ids.csv`: Dataset with semantic IDs
- `train.csv`, `val.csv`, `test.csv`: Split data for T5
- `best_kmeans_*.joblib`: Best K-means model
- `best_dl_*.joblib`: Best Dictionary Learning model

### outputs/figures/
- `pca_plot_*.html`: PCA visualizations (by genre/mood/instrument)
- `umap_interactive.html`: UMAP visualization

### models/
- `t5_semantic_id_model/`: Trained T5 model
  - `checkpoint-*/`: Model checkpoints

## 📚 References

- MTG-Jamendo Dataset
- AcousticBrainz
- T5: Text-To-Text Transfer Transformer

## 🤝 Contributing

Please report bugs or feature requests via Issues.

## 📄 License

[Add license information]
