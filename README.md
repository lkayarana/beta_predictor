# Beta Sheet Prediction with Supervised Hidden Markov Model

This project predicts beta-sheet regions from protein/domain sequences using a supervised Hidden Markov Model (HMM).

## Project idea
- Collect all-beta protein/domain examples
- Download structure files
- Use DSSP to generate residue-level secondary structure labels
- Convert DSSP labels into binary classes:
  - B = beta residue
  - N = non-beta residue
- Train a 2-state HMM on 1000 sequences
- Predict beta regions on 50 unseen sequences
- Report model performance

## Hidden states
- B = beta
- N = non-beta

## Observations
- 20 standard amino acids

## Evaluation metrics
- Accuracy
- Balanced accuracy
- MCC
- Precision
- Recall
- F1-score
- Confusion matrix

## Run order
1. Build train/test manifests
2. Download structures
3. Label train candidates with DSSP
4. Label test candidates with DSSP
5. Finalize clean train/test splits
6. Train HMM
7. Predict and evaluate


## Result

BETA PREDICTOR PROJECT SUMMARY
========================================
Test sequence count: 50
Residue-level accuracy: 0.6022
Balanced accuracy: 0.5898
MCC: 0.1959
Beta precision: 0.6183
Beta recall: 0.3941
Beta F1: 0.4814

Classification report:

              precision    recall  f1-score   support

           B       0.62      0.39      0.48      2964
           N       0.60      0.79      0.68      3363

    accuracy                           0.60      6327
   macro avg       0.61      0.59      0.58      6327
weighted avg       0.61      0.60      0.59      6327