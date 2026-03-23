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
