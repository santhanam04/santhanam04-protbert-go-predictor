#!/usr/bin/env python3
"""
train_and_predict_protbert_ec2.py

EC2-friendly version of your ProtBERT multi-label pipeline with:
 - batched ProtBERT embeddings
 - mixed precision (if CUDA available)
 - checkpointing and model save/load
 - threshold-based and ranked top-k outputs (first-rank = top-1)
"""

import os
import sys
import math
import pickle
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from Bio import SeqIO

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------
# Config - tweak these values
# -----------------------------
TRAIN_FASTA = "train_sequences.fasta"
TRAIN_TERMS_FILE = "train_terms.tsv"
TRAIN_TAXONOMY = "train_taxonomy.tsv"

TEST_FASTA = "testsuperset.fasta"
TEST_TAXONOMY = "testsuperset-taxon-list.tsv"

MODEL_NAME = "Rostlab/prot_bert"
BATCH_SIZE_EMBED = 8         # embedding batch size (reduce if OOM)
BATCH_SIZE_TRAIN = 16        # training batch size
MAX_SEQ_LENGTH = 1024        # ProtBERT typically supports long tokens, adjust if needed
EPOCHS = 5
LR = 1e-4
THRESHOLD = 0.2
TOP_K = 1                    # top-k per sequence for ranked output (first rank = k=1)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# -----------------------------
# Utilities
# -----------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)

def fasta_to_dict(fasta_file):
    seqs = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        seqs[record.id] = str(record.seq)
    return seqs

def normalize_fasta_id(fasta_id):
    parts = fasta_id.split('|')
    if len(parts) == 3:
        return parts[1]
    return fasta_id

# -----------------------------
# Read labels & sequences
# -----------------------------
print("Loading term file...")
train_terms = pd.read_csv(TRAIN_TERMS_FILE, sep="\t", header=0)  # expects EntryID, term
labels = train_terms.groupby("EntryID")["term"].apply(list).to_dict()
all_go_terms = sorted(train_terms["term"].unique().tolist())
print(f"Unique GO terms: {len(all_go_terms)}")

print("Parsing FASTA files...")
train_seqs = fasta_to_dict(TRAIN_FASTA)
test_seqs = fasta_to_dict(TEST_FASTA)

train_seqs = {normalize_fasta_id(k): v for k, v in train_seqs.items()}
test_seqs = {normalize_fasta_id(k): v for k, v in test_seqs.items()}

# keep only train sequences present in labels
train_seqs_filtered = {pid: seq for pid, seq in train_seqs.items() if pid in labels}
print(f"Train sequences (filtered): {len(train_seqs_filtered)}")
print(f"Test sequences: {len(test_seqs)}")

if len(train_seqs_filtered) == 0:
    raise SystemExit("No training sequences match labels. Check IDs and normalization.")

# -----------------------------
# MultiLabelBinarizer
# -----------------------------
print("Preparing MultiLabelBinarizer...")
mlb = MultiLabelBinarizer(classes=all_go_terms, sparse_output=False)  # we'll handle batchwise conversion
y_list = [labels[pid] for pid in train_seqs_filtered.keys()]
y_all = mlb.fit_transform(y_list)   # this is dense; if labels >> memory we can change strategy
print("Label matrix shape:", y_all.shape)

# Save mlb
joblib.dump(mlb, OUTPUT_DIR / "mlb.joblib")
print("Saved ML binarizer to outputs/mlb.joblib")

# -----------------------------
# Load tokenizer + model (ProtBERT)
# -----------------------------
print(f"Loading tokenizer and model: {MODEL_NAME} -> device {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
bert_model.eval()  # we'll use embeddings extracted from this model

# -----------------------------
# Embedding function (batched)
# -----------------------------
def embed_sequences(seqs_dict, batch_size=BATCH_SIZE_EMBED, max_length=MAX_SEQ_LENGTH):
    seq_ids = list(seqs_dict.keys())
    total = len(seq_ids)
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size), desc="Embedding batches"):
            batch_ids = seq_ids[i:i + batch_size]
            # ProtBERT expects sequences as space-separated amino acids
            batch_seqs = [" ".join(list(seqs_dict[pid])) for pid in batch_ids]
            tokens = tokenizer(batch_seqs,
                               padding=True,
                               truncation=True,
                               max_length=max_length,
                               return_tensors="pt")
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            output = bert_model(**tokens)
            # take CLS pooling (first token)
            emb = output.last_hidden_state[:, 0, :].cpu()
            all_embs.append(emb)
    if len(all_embs) == 0:
        return torch.empty((0, bert_model.config.hidden_size))
    return torch.cat(all_embs, dim=0)

# -----------------------------
# Create / load embeddings
# -----------------------------
EMB_TRAIN_PATH = OUTPUT_DIR / "train_embeddings.pt"
EMB_TEST_PATH = OUTPUT_DIR / "test_embeddings.pt"

if EMB_TRAIN_PATH.exists():
    print("Loading cached train embeddings...")
    train_embeddings = torch.load(EMB_TRAIN_PATH)
else:
    print("Embedding train sequences...")
    train_embeddings = embed_sequences(train_seqs_filtered)
    torch.save(train_embeddings, EMB_TRAIN_PATH)
    print("Saved train embeddings.")

if EMB_TEST_PATH.exists():
    print("Loading cached test embeddings...")
    test_embeddings = torch.load(EMB_TEST_PATH)
else:
    print("Embedding test sequences...")
    test_embeddings = embed_sequences(test_seqs)
    torch.save(test_embeddings, EMB_TEST_PATH)
    print("Saved test embeddings.")

print("Train emb shape:", train_embeddings.shape)
print("Test emb shape :", test_embeddings.shape)

# -----------------------------
# Define MLP model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1=1024, hidden2=512, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

# -----------------------------
# Prepare dataloaders
# -----------------------------
X = train_embeddings
y = torch.tensor(y_all, dtype=torch.float32)

# split for validation
idx = np.arange(len(X))
train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=SEED, shuffle=True)

train_ds = TensorDataset(X[train_idx], y[train_idx])
val_ds = TensorDataset(X[val_idx], y[val_idx])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=False)

# -----------------------------
# Instantiate model, criterion, optimizer
# -----------------------------
model = MLP(X.shape[1], y.shape[1]).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

# -----------------------------
# Training loop with checkpointing
# -----------------------------
best_val_loss = float("inf")
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                preds = model(xb)
                loss = criterion(preds, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            if scaler:
                with torch.cuda.amp.autocast():
                    preds = model(xb)
                    loss = criterion(preds, yb)
            else:
                preds = model(xb)
                loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }, CHECKPOINT_PATH)
        print("Saved new best model checkpoint.")

# Save final model weights (in case)
torch.save(model.state_dict(), OUTPUT_DIR / "final_model_weights.pt")
print("Saved final model weights.")

# Also save full model via joblib/pickle for inference (weights + architecture)
joblib.dump({"input_dim": X.shape[1], "output_dim": y.shape[1]}, OUTPUT_DIR / "model_meta.joblib")

# Save tokenizer and model config for future inference reproducibility
tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")
bert_model.config.to_json_file(str(OUTPUT_DIR / "bert_config.json"))

# -----------------------------
# Predict on test set
# -----------------------------
print("Loading best checkpoint for inference...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

X_test = test_embeddings.to(DEVICE)
preds = []

with torch.no_grad():
    # batch predict if necessary
    bs = max(1, BATCH_SIZE_TRAIN)
    for i in tqdm(range(0, X_test.size(0), bs), desc="Predicting test"):
        xb = X_test[i:i+bs]
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                out = model(xb)
        else:
            out = model(xb)
        preds.append(out.cpu())
preds = torch.cat(preds, dim=0).numpy()
print("Predictions shape:", preds.shape)

# -----------------------------
# Output 1: threshold-based submission (same as original)
# -----------------------------
submission = []
test_ids = list(test_seqs.keys())
for i, pid in enumerate(test_ids):
    for j, prob in enumerate(preds[i]):
        if prob > THRESHOLD:
            submission.append([pid, mlb.classes_[j], float(round(prob, 3))])
sub_df = pd.DataFrame(submission)
sub_df.to_csv(OUTPUT_DIR / "submission_threshold.tsv", sep="\t", index=False, header=False)
print(f"Saved threshold-based submission: {OUTPUT_DIR/'submission_threshold.tsv'}")

# -----------------------------
# Output 2: ranked top-K per sequence (first rank = k=1)
# -----------------------------
ranked_rows = []
for i, pid in enumerate(test_ids):
    probs = preds[i]
    topk_idx = np.argsort(-probs)[:TOP_K]   # descending sort
    for rank_position, idx in enumerate(topk_idx, start=1):
        ranked_rows.append([pid, rank_position, mlb.classes_[idx], float(round(probs[idx], 6))])

ranked_df = pd.DataFrame(ranked_rows, columns=["EntryID", "Rank", "Term", "Prob"])
ranked_df.to_csv(OUTPUT_DIR / "submission_ranked_topk.tsv", sep="\t", index=False)
print(f"Saved ranked top-{TOP_K} submission: {OUTPUT_DIR/'submission_ranked_topk.tsv'}")

print("Done. Outputs in", OUTPUT_DIR)
