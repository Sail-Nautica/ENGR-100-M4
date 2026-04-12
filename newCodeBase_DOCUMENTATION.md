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
