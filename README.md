# Sleep Stage Classification Using Machine Learning

## Project overview

I built this project to study how classical machine-learning models can classify sleep stages from electroencephalography (EEG) signals, and whether giving a model information from neighbouring epochs can improve its predictions.

Sleep staging divides a sleep recording into five stages: Wake, N1, N2, N3, and rapid eye movement (REM). These stages are usually scored by trained specialists. Automating parts of this process could help researchers analyse long recordings more efficiently, but a model must be evaluated carefully to understand whether it works for people it has not seen before.

This project provides an end-to-end research pipeline for preprocessing EEG recordings, extracting interpretable signal features, training and comparing classical classifiers, and evaluating their performance with participant-level separation. It is a research prototype, not a medical device or a clinical diagnostic system.

## Research question and scope

The main question is:

**How does adding information from neighbouring labelled EEG epochs affect the performance of classical machine-learning models for five-stage sleep classification when each test participant is kept separate from training?**

The work focuses on three connected parts:

1. **A single-epoch baseline.** Each 30-second EEG epoch is represented by handcrafted time-domain, Hjorth-style, and spectral features.
2. **A context-aware representation.** The feature vector for an epoch is combined with feature vectors from neighbouring retained labelled epochs in the same recording. This lets the classifier use surrounding signal patterns as well as the central epoch.
3. **A controlled model comparison.** Random Forest, Extra Trees, RBF Support Vector Machine, k-Nearest Neighbours, and XGBoost are evaluated using leave-one-subject-out cross-validation. An additional XGBoost experiment investigates different context-window sizes.

The project is designed to make the pipeline and its evaluation steps inspectable. The scripts, processed feature matrix, epoch metadata, saved aggregate results, and manuscript are included in this repository.

## Dataset

The experiments use a subset of the public dataset *A Continuous Multi-night Sleep Database for Exploring Sleep Structure Similarity* (Scientific Data, 2025, DOI: [10.1038/s41597-025-05202-6](https://doi.org/10.1038/s41597-025-05202-6)). The repository's processed data covers participants SLP001 to SLP005 and includes 13,628 labelled 30-second epochs from 13 recordings.

The five target labels are:

| Label code | Sleep stage |
|---:|---|
| 0 | Wake |
| 1 | N1 |
| 2 | N2 |
| 3 | N3 |
| 4 | REM |

The reference label for each retained epoch is created by majority voting across the available expert annotation tracks. This creates one target label per epoch for supervised learning.

The original EDF recordings are not included because of their size and the dataset's access terms. See [`DATASET_NOTE.txt`](DATASET_NOTE.txt) for information on obtaining the data and rebuilding the processed features. Please consult the official dataset record and follow its terms before downloading or redistributing any data.

## Signal processing and features

The preprocessing pipeline selects C4-A1 when available, with defined alternatives when it is not. The chosen signal is resampled to 100 Hz and filtered between 0.5 and 40 Hz. It is divided into non-overlapping 30-second epochs.

Each epoch is represented by 22 handcrafted features. They describe signal amplitude and distribution, Hjorth-style activity and variation, power in five frequency bands, relative band power, and spectral entropy. The frequency bands are delta (0.5 to 4 Hz), theta (4 to 8 Hz), alpha (8 to 12 Hz), sigma (12 to 16 Hz), and beta (16 to 30 Hz).

For the context-aware experiment, selected absolute band-power features are log-transformed. The central epoch's feature vector is concatenated with vectors from offsets up to four retained labelled epochs before and after it, producing a feature vector of up to 198 values.

There are two details worth understanding when interpreting this representation:

- The context offsets refer to neighbouring **retained labelled epochs** within a recording. Since unlabelled epochs are removed earlier in preprocessing, adjacent retained rows are not guaranteed to be consecutive 30-second intervals in the original recording.
- The context includes epochs from both before and after the target epoch. The current context-aware model is therefore intended for offline sleep-recording analysis, not causal real-time classification.

## Evaluation approach

The main evaluation uses **leave-one-subject-out (LOSO) cross-validation**. In each fold, all available epochs from one participant are held out for testing, while the remaining participants form the training set. This tests performance on a participant who was not included in that fold's training data.

The project reports:

- **Accuracy:** the proportion of correctly classified epochs.
- **Macro-F1:** the mean F1-score across the five sleep stages, giving each stage equal weight.
- **Weighted-F1:** the F1-score averaged according to each stage's number of epochs.
- **Cohen's kappa:** agreement between predictions and reference labels adjusted for agreement expected from the label distributions.

Macro-F1 is important because the dataset has an uneven distribution of stages. A high overall accuracy can hide weaker performance on less common stages such as N1.

## Current context-aware results

The aggregate results currently saved in [`results/model_comparison_context.csv`](results/model_comparison_context.csv) are shown below. They are included as recorded experiment outputs, and can be regenerated by running the context-aware training script.

| Model | Accuracy | Macro-F1 | Weighted-F1 | Cohen's kappa |
|---|---:|---:|---:|---:|
| Random Forest | 85.07% | 76.76% | 83.76% | 0.7941 |
| Extra Trees | **85.11%** | 76.67% | 83.69% | 0.7947 |
| RBF SVM | 82.07% | 77.36% | 82.56% | 0.7614 |
| k-NN | 80.06% | 70.44% | 78.37% | 0.7246 |
| XGBoost | 84.89% | **80.34%** | **85.24%** | **0.7975** |

In this run, Extra Trees has the highest accuracy. XGBoost has the highest macro-F1, weighted-F1, and Cohen's kappa. This illustrates why it is useful to compare several metrics instead of choosing a model by accuracy alone.

The project also includes a context-window ablation script for XGBoost. It evaluates the single-epoch representation and multiple context-window sizes. The generated ablation output should be checked against the final manuscript before quoting those values elsewhere.

These results are from a five-participant pilot subset. They should not be interpreted as a clinical performance estimate or as evidence that the same scores will be obtained on a larger or independent dataset. Runtime values can also vary by machine and software environment.

## Repository structure

```text
SleepStage_GitHub/
├── src/
│   ├── build_dataset.py          # Build processed features from raw EDF and annotations
│   ├── train_classical.py        # Single-epoch classical model comparison
│   ├── train_xgboost.py          # Standalone single-epoch XGBoost evaluation
│   ├── train_context.py          # Context-aware comparison of five classifiers
│   ├── train_context_ablation.py # XGBoost context-window ablation
│   └── train_deep.py             # Experimental deep-learning code, not part of reported results
├── results/
│   ├── features.npz               # Processed features, labels, and participant groups
│   ├── epoch_metadata.csv         # Epoch labels and recording metadata
│   ├── model_comparison_context.csv
│   ├── model_comparison_single_epoch.csv
│   └── saved metrics and predictions from prior runs
├── figures/
│   ├── model_comparison_context.png
│   └── other generated or baseline figures
├── tests/
│   └── test_features.py
├── docs/
│   └── final_manuscript.pdf
├── DATASET_NOTE.txt
├── requirements.txt
└── README.md
```

The deep-learning script is exploratory and its results are not included in the reported five-model comparison. The confusion matrices in the single-epoch baseline directory, where present, belong to the baseline experiment. To obtain confusion matrices for the current context-aware run, execute `src/train_context.py` and use the newly generated files with the `_context` suffix.

## Getting started

Python 3.10 or newer is recommended. Run these commands from the repository's root directory.

### 1. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use the Python executable inside `.venv` directly or follow your local Python environment policy.

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run the context-aware model comparison

The processed feature matrix and epoch metadata are already included, so this benchmark does not require the raw EDF files.

```bash
python src/train_context.py
```

The script writes the pooled model comparison, per-fold metrics, classification reports, held-out predictions, confusion-matrix CSV files, confusion-matrix PNG files, and model-comparison chart to the `results/` and `figures/` directories.

### 4. Run the context-window ablation

```bash
python src/train_context_ablation.py
```

This script evaluates the XGBoost baseline under several feature-context configurations. Its generated CSV and fold metrics are written to `results/`.

### 5. Run the single-epoch baselines

```bash
python src/train_classical.py
python src/train_xgboost.py
```

These scripts belong to the single-epoch baseline experiments. Their results should be kept separate from the context-aware results when preparing tables or figures.

### 6. Run the tests

```bash
python -m pytest -q
```

## Rebuilding the processed dataset

The processed feature matrix is provided for convenience. To rebuild it from the original recordings:

1. Obtain the dataset from its official source and check its access and use conditions.
2. Follow the expected directory layout described in [`DATASET_NOTE.txt`](DATASET_NOTE.txt).
3. Place the extracted participant data under `data/raw/` as described in that note.
4. Run:

```bash
python src/build_dataset.py
```

The script writes `results/features.npz` and `results/epoch_metadata.csv`. After rebuilding the features, run the relevant training script to regenerate model results. Rebuilding from raw data may require substantial disk space and processing time.

## Reproducibility notes

I have kept the baseline and context-aware experiments separate so that results from different feature representations are not accidentally mixed. The saved aggregate context-aware metrics are available in the repository, but a reproducibility check should still rerun the relevant scripts and compare their outputs before relying on exact values in a publication or external report.

The current study has several boundaries: it uses five participants, one selected EEG channel per recording, a majority-vote label for each epoch, and handcrafted features. The context-aware model also uses future epochs. These design choices define the current research scope and provide clear directions for follow-up work, including evaluation on more participants, examination of additional physiological channels, analysis of expert disagreement, and comparisons with sequence-based models under the same participant-level splits.

## Research paper

The manuscript PDF is available at [`docs/final_manuscript.pdf`](docs/final_manuscript.pdf). It describes the dataset subset, preprocessing pipeline, feature representation, validation protocol, model comparison, and the study's main findings.

## Dataset reference

S. Ying et al., "A Continuous Multi-night Sleep Database for Exploring Sleep Structure Similarity," *Scientific Data*, 2025. DOI: [10.1038/s41597-025-05202-6](https://doi.org/10.1038/s41597-025-05202-6).

## Author's note

I developed this project as a study of practical, interpretable approaches to automated sleep-stage classification. The goal is to provide a clear baseline and a reproducible starting point for exploring how temporal context changes model performance, rather than to claim a clinical-ready system.
