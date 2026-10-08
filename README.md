# 🎯 Grammar Scoring Engine

A multimodal machine learning system for **automated grammar and spoken-language scoring** from candidate speech transcripts and audio-derived features.

This project was developed as part of a **Research Engineer hiring assessment** and focuses on building a robust regression system that predicts a grammar/proficiency score from approximately **0–5**.

The system combines:

* 📝 Transcript / NLP features
* 🧠 Whisper speech embeddings
* 🎙️ Audio features
* 🔤 Grammar and fluency features
* 📊 TF-IDF word and character representations
* 🤗 Pretrained sentence embeddings
* 🤖 Ridge regression
* 🌳 HistGradientBoosting
* 🌲 ExtraTrees
* 🔄 Residual modeling
* 🔀 Cross-modal feature fusion
* 🧩 Out-of-fold stacking
* 🎯 Calibration and conservative model promotion

---

## 🚀 Project Objective

The goal is to predict a continuous grammar score from spoken responses.

### Input

Each training example contains information such as:

```text
filename
transcript
audio-derived features
Whisper embeddings
```

### Output

A continuous score:

```text
0.0 → 5.0
```

The primary evaluation metric is:

```text
RMSE (Root Mean Squared Error)
```

The project also monitors:

```text
Pearson Correlation
```

to understand how well model predictions follow the ordering and variation of human-provided scores.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     Audio Input     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Whisper Embeddings  │
                    └──────────┬──────────┘
                               │
                               │
                               ▼
┌──────────────┐       ┌─────────────────────┐
│  Transcript  │──────►│   Text Processing   │
└──────────────┘       └──────────┬──────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
               TF-IDF        Sentence       Grammar &
               Word/Char     Embeddings     Fluency
                    │             │             │
                    └─────────────┼─────────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │ Cross-Modal Model │
                        └────────┬─────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ OOF Model Predictions│
                      └──────────┬──────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Meta Learner  │
                         └───────┬───────┘
                                 │
                                 ▼
                          ┌────────────┐
                          │ Calibration│
                          └─────┬──────┘
                                │
                                ▼
                      ┌────────────────────┐
                      │ Final Grammar Score│
                      │       0 – 5        │
                      └────────────────────┘
```

---

# 🧠 Feature Engineering

## 1. Transcript Features

The transcript is treated as the primary linguistic source.

The pipeline extracts:

* Word count
* Character count
* Sentence count
* Average word length
* Word-length variance
* Vocabulary size
* Lexical diversity
* Repeated words
* Repetition rate
* Long-word frequency
* Very-long-word frequency
* Filler words
* Hesitation patterns
* Punctuation density
* Question frequency
* Exclamation frequency
* Comma usage
* Article usage
* Pronoun usage
* Auxiliary verbs
* Conjunctions
* Prepositions
* Sentence-length statistics
* Potential sentence fragments
* Very long sentences
* Capitalization patterns
* Digit usage

These features are designed to capture aspects of:

```text
Fluency
Vocabulary
Sentence construction
Repetition
Hesitation
Complexity
Grammar
```

---

# 🔤 TF-IDF Text Representation

Two complementary TF-IDF representations are used.

### Word TF-IDF

```text
1-gram
2-gram
3-gram
```

This captures lexical and phrase-level patterns.

### Character TF-IDF

```text
3-gram
4-gram
5-gram
6-gram
```

Character representations can capture:

* Morphological patterns
* Spelling variations
* Partial word patterns
* Common grammatical structures
* Word endings

The representations are compressed using:

```text
TruncatedSVD
```

before being combined with other features.

---

# 🤗 Semantic Text Embeddings

Where the Kaggle environment permits, the system uses pretrained SentenceTransformer models.

The primary candidate is:

```text
all-MiniLM-L6-v2
```

These embeddings provide a dense semantic representation of each transcript.

Instead of relying exclusively on manually engineered features, the model can learn higher-level linguistic relationships from pretrained language representations.

The system automatically falls back to the TF-IDF representation if pretrained sentence embeddings are unavailable.

---

# 🎙️ Whisper Speech Embeddings

Precomputed Whisper embeddings are incorporated into the model.

Whisper representations provide information derived from the original speech signal and can capture information that may not be completely represented by the transcript alone.

The pipeline applies:

```text
StandardScaler
      ↓
PCA
      ↓
Whisper representation
```

The Whisper representation is then combined with text and audio features.

---

# 🎧 Audio Features

The system also incorporates numerical audio features.

The processing pipeline includes:

```text
Missing-value imputation
        ↓
Median replacement
        ↓
StandardScaler
```

Audio features are used independently as well as in combination with:

```text
Whisper + Audio
Audio + Text
Whisper + Audio + Text
```

---

# 🔀 Cross-Modal Learning

A major focus of the project is combining information from different modalities.

Examples include:

### Whisper + Text

```text
Whisper embeddings
+
Sentence embeddings
+
Grammar features
```

### Whisper + Audio

```text
Whisper embeddings
+
Audio features
```

### Audio + Text

```text
Audio features
+
Semantic embeddings
+
Grammar features
```

### Full multimodal representation

```text
Whisper
+
Audio
+
Semantic Text
+
TF-IDF
+
Grammar Features
```

The goal is to capture complementary information rather than relying on a single representation.

---

# 🤖 Models

Several regression models are evaluated.

## Ridge Regression

Ridge is heavily used because high-dimensional NLP representations can contain many correlated features.

Different regularization strengths are evaluated:

```text
α = 10
α = 30
α = 50
α = 100
α = 150
α = 300
```

---

## HistGradientBoosting

Gradient boosting is used for nonlinear relationships.

It is particularly useful for:

```text
Grammar features
Audio features
Cross-modal representations
```

---

## ExtraTrees

ExtraTrees is used as a nonlinear challenger model for engineered grammar/audio representations.

---

## Residual Modeling

Instead of directly predicting the target, residual models attempt to learn what the baseline model misses.

Conceptually:

```text
Base Prediction
       ↓
Residual = Actual Score - Base Prediction
       ↓
Residual Model
       ↓
Improved Prediction
```

This allows additional modalities to focus on information that is not already captured by the primary model.

---

# 🔄 Out-of-Fold Training

A major part of the system is **Out-of-Fold (OOF) prediction**.

The training data is divided using:

```text
5-Fold K-Fold Cross Validation
```

For each fold:

```text
80% → train
20% → validation
```

The validation predictions are stored as OOF predictions.

This provides predictions for the entire training dataset without directly training a model on the corresponding validation sample.

OOF predictions are then used for:

```text
Model comparison
Error analysis
Prediction correlation
Stacking
Calibration
```

---

# 🧩 Stacking

Instead of simply averaging every model, the system creates a second-level model.

Example:

```text
Phase 16 prediction
        +
Semantic model
        +
Grammar model
        +
Whisper/Text model
        +
Audio/Text model
        ↓
     Meta Model
        ↓
 Final prediction
```

The meta learner learns how much each model contributes to the final prediction.

---

# 🎯 Model Diversity

Model diversity is explicitly measured.

For every challenger model, the project evaluates:

```text
RMSE
Pearson correlation
Prediction dispersion
Prediction correlation with Phase 16
```

Highly correlated models are not blindly added to the ensemble.

The objective is:

> Add models that provide genuinely different information.

---

# 📊 Error Analysis

The system evaluates performance across different target ranges:

```text
0–1
1–2
2–3
3–4
4–5
```

This helps identify whether the model performs differently for:

* Low-scoring responses
* Average responses
* High-scoring responses

This is especially important because a low overall RMSE can hide significant errors in particular score ranges.

---

# 📈 Calibration

After model stacking, a conservative calibration step searches for:

```text
prediction × scale + offset
```

The calibration parameters are selected using OOF predictions.

Example:

```text
raw prediction
       ↓
mean-centered
       ↓
scale
       ↓
offset
       ↓
clip to 0–5
```

Calibration is accepted only when it provides a meaningful improvement over the uncalibrated prediction.

---

# 🏆 Champion Model Strategy

The project uses a conservative promotion strategy.

The previous best model is treated as an immutable baseline.

For example:

```text
Phase 16 Champion
        │
        ├── Phase 17
        ├── Phase 18
        └── Phase 19
```

A new phase is promoted only when it beats the existing champion by a predefined margin.

This prevents accidental leaderboard degradation caused by overfitting or unstable ensembles.

---

# 🧪 Development Phases

The system has been developed iteratively.

### Phase 14

Established the initial multimodal scoring pipeline.

### Phase 16

Produced a stronger baseline/champion using Whisper and audio information.

### Phase 17

Experimented with additional modeling strategies but did not provide a meaningful improvement.

### Phase 18

Introduced:

* Cross-modal OOF stacking
* Hard-example modeling
* Whisper + Audio
* Whisper + Text
* Audio + Text
* Residual modeling
* Meta learning
* Prediction diversity analysis

### Phase 19

The focus shifted from adding more traditional ensemble models to improving the **linguistic representation**.

New components include:

* Pretrained sentence embeddings
* Expanded word TF-IDF
* Expanded character TF-IDF
* Grammar features
* Fluency features
* Semantic + grammar models
* Whisper + semantic models
* Whisper + audio + semantic models
* OOF meta learning
* Conservative calibration

---

# 📏 Evaluation

The primary metric is:

## RMSE

```text
RMSE = sqrt(mean((y_true - y_pred)²))
```

Lower is better.

The secondary metric is:

## Pearson Correlation

Higher is better.

The system tracks both because a model can have a strong correlation while still having poor absolute calibration.

---

# 📁 Project Structure

A typical project structure is:

```text
grammar-scoring-engine/
│
├── README.md
│
├── notebooks/
│   ├── phase14.ipynb
│   ├── phase16.ipynb
│   ├── phase17.ipynb
│   ├── phase18.ipynb
│   └── phase19.ipynb
│
├── data/
│   └── README.md
│
├── outputs/
│   ├── phase19_model_results.csv
│   ├── phase19_oof_predictions.csv
│   ├── phase19_test_predictions.csv
│   ├── phase19_summary.json
│   └── submission_phase19.csv
│
└── requirements.txt
```

Dataset files are intentionally **not committed** when they are private or provided through the assessment platform.

---

# 🛠️ Technology Stack

### Programming

* Python
* NumPy
* Pandas
* SciPy

### Machine Learning

* Scikit-learn
* Ridge Regression
* HistGradientBoosting
* ExtraTrees
* PCA
* TruncatedSVD
* K-Fold Cross Validation

### NLP

* TF-IDF
* Sentence Transformers
* Whisper embeddings
* Grammar/fluency feature engineering

### Experimentation

* Kaggle Notebooks
* Out-of-Fold validation
* Error analysis
* Ensemble modeling
* Model calibration

---

# ⚙️ Installation

```bash
pip install numpy pandas scipy scikit-learn
```

For semantic embeddings:

```bash
pip install sentence-transformers
```

---

# ▶️ Running the Project

The project is primarily designed for Kaggle.

After placing the required dataset files in the expected Kaggle input directory, run the notebook phases sequentially.

For Phase 19:

```text
Load dataset
      ↓
Load transcripts
      ↓
Load audio features
      ↓
Load Whisper embeddings
      ↓
Build grammar features
      ↓
Build TF-IDF
      ↓
Build semantic embeddings
      ↓
Train OOF models
      ↓
Evaluate models
      ↓
Stack predictions
      ↓
Calibrate
      ↓
Promotion check
      ↓
Generate submission
```

---

# 🔐 Data Privacy

The original assessment dataset may contain private hiring-assessment data.

Therefore:

* Raw assessment data should not be committed.
* Candidate audio should not be uploaded publicly.
* Private Kaggle challenge data should remain private.
* Generated embeddings derived from private assessment data should also be treated carefully.

Only code, methodology, documentation, and non-sensitive experiment results should be published.

---

# 📌 Current Research Direction

The key research question is:

> **How much can grammar-score prediction improve by combining speech representations, semantic language representations, and explicit grammar/fluency features?**

The current approach investigates three complementary sources of information:

```text
                Grammar
                   ▲
                   │
                   │
Speech ───────► Prediction ◄────── Language
                   │
                   ▼
                 Audio
```

The hypothesis is that:

* Whisper captures speech-level information.
* Audio features capture acoustic characteristics.
* Sentence embeddings capture semantic/language information.
* TF-IDF captures lexical and phrase-level patterns.
* Grammar features capture explicit linguistic behavior.

Combining these representations may provide a more robust estimate of spoken grammar proficiency than any single modality.

---

# 📊 Experiment Tracking

Each phase records:

```text
Model
RMSE
Pearson correlation
Prediction mean
Prediction standard deviation
Prediction diversity
Hard-example performance
```

This makes it possible to compare experiments systematically rather than selecting models only from intuition.

---

# 🚧 Limitations

This project is still experimental.

Important limitations include:

1. The dataset is relatively small compared with typical deep-learning NLP datasets.
2. The target labels may contain human scoring variability.
3. Transcript quality directly affects linguistic features.
4. Whisper embeddings may contain information unrelated to grammar.
5. Sentence embeddings are not specifically trained for this scoring task.
6. OOF validation reduces but does not completely eliminate experimentation bias.
7. A very low RMSE target may require understanding how the original labels were generated.

---

# 🔮 Future Improvements

Potential future research directions include:

### Grammar-specific pretrained models

Fine-tune a language model specifically for grammaticality or language proficiency scoring.

### Ordinal regression

The score range has an ordered structure:

```text
0 < 1 < 2 < 3 < 4 < 5
```

Ordinal approaches could potentially exploit this structure better than ordinary regression.

### Linguistic parsing

Add:

* POS tagging
* Dependency parsing
* Subject–verb agreement
* Verb tense consistency
* Article/determiner errors
* Sentence completeness
* Clause complexity
* Dependency-tree statistics

### Speech features

Investigate:

* Speaking rate
* Pause duration
* Filler frequency
* Pronunciation features
* Voice activity
* Prosody
* Pitch variation

### Multi-task learning

Predict multiple intermediate linguistic attributes alongside the final grammar score.

---

# 👨‍💻 Author

**Soumen Maity**

Software Engineer | Full-Stack Developer | Machine Learning Enthusiast

Interests:

```text
Machine Learning
NLP
Speech AI
Generative AI
Backend Engineering
Full-Stack Development
AI-powered Applications
```

---

# ⭐ Key Takeaway

This project is not simply an ensemble of regression models.

The main approach is to progressively investigate the problem from multiple perspectives:

```text
Baseline
   ↓
Multimodal Features
   ↓
Whisper
   ↓
Audio
   ↓
TF-IDF
   ↓
Cross-Modal Learning
   ↓
Residual Modeling
   ↓
OOF Stacking
   ↓
Semantic Embeddings
   ↓
Grammar Features
   ↓
Calibration
   ↓
Robust Grammar Scoring Engine
```

The long-term goal is to build a **reliable, interpretable, and generalizable automated grammar scoring system** rather than optimizing for a single leaderboard result.
