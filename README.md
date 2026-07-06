# Siamese Anomaly Detection

Spotting anomalies by learning a distance between sequences.

## What the notebook does

The notebook defines anomalies based on a quantile of the target: any value above the 95th percentile of the time-to-event variable is labeled anomalous, which marks roughly 10,000 of 200,000 rows (the top 5%). A memory-safe subset of 50,000 rows is turned into sequences using a window of 24 steps with a stride of 5, giving just under 10,000 sequences, which are split into train, validation, and test sets with stratification so each side keeps the same share of anomalies.

Balanced pairs are then built for metric learning: each similar pair joins two normal sequences, and each dissimilar pair joins a normal sequence with an anomalous one. A Siamese network processes both sides of a pair with the same shared encoder, a 1D CNN with two convolution blocks (64 and 128 filters) feeding a 128-dimensional embedding. The model is trained with contrastive loss, which pulls similar pairs close together in the embedding space and pushes dissimilar pairs apart. After training, distances on the validation pairs are used to pick a decision threshold, an ROC curve is drawn, and a helper function classifies a new sequence as anomalous by comparing its average distance to a reference set of normal sequences.

## Tech

Python, TensorFlow / Keras, scikit-learn, NumPy

## Data & trained models
Large data files and model weights are hosted on Google Drive: 
https://drive.google.com/drive/folders/1QHk6sbWbRPGKOpGKMo3FWEW9V5gGDTcC

## Notes

The separation looks near perfect because the sequences are built from the same variable the anomaly labels come from, so the network only learns a threshold on its own input. The biggest limitation is that flaw, plus an ROC cell that scores with raw distance and prints an inverted AUC. Training on independent sensor features and fixing the score direction would make the results meaningful. Metric learning like this can be used in fraud detection and equipment fault detection, where anomalies are too rare and varied to classify directly.
