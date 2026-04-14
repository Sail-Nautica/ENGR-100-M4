# Question Classification with Sentence Embeddings — Codex-Style Embedding Model Comparison

## Overview
- Use the provided train.csv and test.csv files as-is
- Embed each question into a feature vector
- Learn the same kind of linear multiclass PyTorch model on top of each embedding space
- Compare the two embedding models head-to-head
- Identify where each embedding succeeds, where it fails, and which classes each model struggles with
- Use confusion matrices, disagreement examples, and coarse-category analysis to diagnose model deficiencies

## Inputs
- Notebook: `/Users/nautica/Documents/michigan/Classes/engr100/M4/newCodeBase.ipynb`
- Dataset: `question_classificatrion/question_classification_dataset/test.csv`
- Dataset: `question_classificatrion/question_classification_dataset/train.csv`

## Dependencies
- matplotlib
- numpy
- pandas
- scikit-learn
- seaborn
- sentence-transformers
- torch

## Embedding Models
- all-MiniLM-L6-v2
- paraphrase-MiniLM-L6-v2

## Notebook Sections
- Step 1: Load and inspect the data
- Step 2: Create a holdout split from the provided training set
- Step 3: Choose two embedding models
- Helper functions
- Step 4: Embed the data and train the linear multiclass PyTorch model
- Step 5: Baseline head-to-head comparison
- Step 9: Add targeted holdout samples back into training
- Step 10: Retrain after holdout augmentation
- Step 11: Confusion matrices after holdout augmentation

## Data Splits
- Holdout test_size: 0.1

## Training Details
- linear layer (nn.Linear)
- MSE loss
- SGD optimizer
- Epochs used: 1000, 2000, 10000, 20000, 30000
- Learning rates used: 0.01, 0.02, 4, 5, 6, 7, 8

## Key Functions
- build_linear_multiclass_model
- compute_accuracy_grid_for_model
- embed_texts
- labels_to_index
- make_multiclass_tensors
- model_fit_pytorch
- one_hot_encode
- plot_accuracy_grid_table
- plot_conf_mat
- pytorch_model_multiclass_inference
- train_one_embedding_pipeline

## Outputs
- Output directory: `matrices`
- Output directory: `new_matrices`
- Output file pattern: `*.png`
- Output file pattern: `augmented_{model_name}_coarse.png`
- Output file pattern: `augmented_{model_name}_fine.png`
- Output file pattern: `augmented_{model_name}_fine_subset5.png`
- Output file pattern: `augmented_{model_name}_loss.png`
- Output file pattern: `baseline_*_*.png`
- Output file pattern: `baseline_*_coarse.png`
- Output file pattern: `baseline_*_fine.png`
- Output file pattern: `baseline_{model_name}_coarse.png`
- Output file pattern: `baseline_{model_name}_fine.png`
- Output file pattern: `baseline_{model_name}_loss.png`
- Output file pattern: `combined_matrices.pdf`
- Output file pattern: `lr_epoch_accuracy_*.png`
- Output file pattern: `lr_epoch_accuracy_{model_name}.png`
- Output file pattern: `{subset_name}_{model_name}.png`

## Holdout Augmentation Sweep
Assumes the notebook variables and helper functions are already defined (splits, class mappings, and training helpers).

```python
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

augmentation_percentages = [0.05, 0.1, 0.2, 0.3, 0.4]
target_coarse_labels = sorted(train_split_df["label-coarse"].unique().tolist())
sweep_rows = []

for pct in augmentation_percentages:
    train_aug = train_split_df.copy()
    holdout_aug = holdout_split_df.copy()

    train_counts = train_aug["label-coarse"].value_counts()
    added_rows = []
    for coarse_label in target_coarse_labels:
        desired_increase = int(np.ceil(train_counts.get(coarse_label, 0) * pct))
        candidates = holdout_aug[holdout_aug["label-coarse"] == coarse_label]
        take_n = min(desired_increase, len(candidates))
        if take_n == 0:
            continue
        sampled = candidates.sample(n=take_n, random_state=RANDOM_STATE)
        added_rows.append(sampled)
        holdout_aug = holdout_aug.drop(index=sampled.index)

    if added_rows:
        added_df = pd.concat(added_rows, ignore_index=True)
        train_aug = pd.concat([train_aug, added_df], ignore_index=True)

    X_train_text = train_aug["text"].tolist()
    y_train_labels = train_aug["label-fine"].to_numpy()

    model_name = "all-MiniLM-L6-v2"
    artifacts = train_one_embedding_pipeline(
        model_name=model_name,
        X_train_text=X_train_text,
        X_test_text=X_test_text,
        y_train_labels=y_train_labels,
        y_test_labels=y_test_labels,
        class_to_idx=class_to_idx,
        class_names=class_names,
        epochs=2000,
        lr=0.02,
    )

    X_holdout_text = holdout_aug["text"].tolist()
    y_holdout_labels = holdout_aug["label-fine"].to_numpy()
    X_holdout_emb = embed_texts(model_name, X_holdout_text)
    X_holdout_tensor, _, y_holdout_idx = make_multiclass_tensors(
        X_holdout_emb, y_holdout_labels, class_to_idx
    )
    _, holdout_pred_idx = pytorch_model_multiclass_inference(
        artifacts["trained_model"], X_holdout_tensor
    )

    sweep_rows.append(
        {
            "pct_added": pct,
            "train_size": len(train_aug),
            "holdout_size": len(holdout_aug),
            "test_accuracy": accuracy_score(
                artifacts["y_test_idx"], artifacts["test_pred_idx"]
            ),
            "holdout_accuracy": accuracy_score(y_holdout_idx, holdout_pred_idx),
        }
    )

pd.DataFrame(sweep_rows)
```
