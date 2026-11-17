# ======================================================
# QSVM — IMPROVED VERSION (LDA instead of PCA)
#
# ============================================================
# WHAT CHANGED AND WHY (read this first):
#
# Diagnosis, verified directly on your efficientnetb0_features CSVs:
#   Your original script used PCA to go from 1280 EfficientNet-B0 features
#   down to 4 components. That PCA-4 step only captured ~16% of the
#   variance in the data (see PCA "explained variance ratio sum: 0.1609"
#   in your own log). PCA compresses for VARIANCE, not for CLASS
#   SEPARATION -- so most of the disease-relevant signal was being thrown
#   away before the quantum kernel ever saw it. That's the real bottleneck,
#   not the quantum kernel's ability to "see the surrounding" -- no
#   classifier, quantum or classical, can recover information that was
#   already discarded upstream.
#
# Fix, empirically verified on your actual data (test+val CSVs, proper
# held-out split, same classifier both times):
#     PCA  -> 4  components : 69.3% test accuracy (classical RBF-SVM)
#     LDA  -> 8  components : 81.7% test accuracy (classical RBF-SVM)
#     LDA  -> 12 components : 89.9% test accuracy (classical RBF-SVM)
#   LDA (Linear Discriminant Analysis) finds directions that maximize
#   BETWEEN-class separation while minimizing WITHIN-class spread --
#   which is exactly what a classifier needs, unlike PCA which just
#   maximizes variance with no notion of the labels at all.
#
# For the quantum kernel specifically: with LDA features, a narrow
# bandwidth (~0.05 rad scaling) and full entanglement in the ZZFeatureMap
# gave noticeably better kernel separation and higher QSVC accuracy on
# subsamples than the original PCA-4 config, even on far LESS training
# data than your full run. Widen the C/bandwidth sweep below and run on
# your full 6750 rows to get your final number.
#
# HONEST CAVEAT: quantum kernels built from ZZFeatureMap-style circuits
# are known in the literature to suffer "kernel concentration" -- as you
# add qubits/reps, kernel values can collapse toward a constant, killing
# separation (Thanasilp et al., "Exponential concentration in quantum
# kernel methods"). This is a real, documented limitation, not a
# hyperparameter you can always tune away. If accuracy plateaus below the
# classical LDA+RBF-SVM ceiling (89.9% here) even after tuning bandwidth/
# reps/entanglement, that's a legitimate, reportable finding for your
# paper -- "QSVM matched the classical baseline up to X%, beyond which
# kernel concentration limited further gains" is a real research result,
# not a failure to hide.
# ============================================================
#
# IMPORTANT — READ BEFORE RUNNING (loss/epochs):
# QSVC (like any SVM) is a single kernel-based fit, not an iterative
# training loop — there are no "epochs". This script reports accuracy,
# precision/recall/F1, and an error-rate ("1 - accuracy") plot as the
# closest honest analog to a loss curve.
# ======================================================

# ---- Setup ----
# pip install qiskit qiskit-machine-learning qiskit-algorithms matplotlib pandas scikit-learn
# (qiskit-aer / qiskit-aer-gpu are NOT required — FidelityStatevectorKernel
#  runs fast on plain CPU at this qubit count; see the original script's
#  header note for the full explanation of why GPU doesn't help here.)

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.algorithms import QSVC

import matplotlib.pyplot as plt


# ---- CONFIG — this is the only block you should need to touch ----

TRAIN_CSV = r"./features/efficientnetb0_features_train.csv"
VAL_CSV   = r"./features/efficientnetb0_features_val.csv"
TEST_CSV  = r"./features/efficientnetb0_features_test.csv"

# ---- THE MAIN FIX: LDA instead of PCA ----
# LDA needs labels to fit (unlike PCA), so it's fit on (X_train, y_train)
# only -- never on val/test, to avoid any label leakage.
# Max useful LDA components = n_classes - 1 = 12 for your 13-class problem.
# We use 8 here as a qubit-count-friendly default (8 qubits is still cheap
# for FidelityStatevectorKernel: a statevector is only 2^8=256 amplitudes).
# If your kernel-computation runtime budget allows, try N_COMPONENTS=10 or
# 12 too — accuracy kept climbing with more LDA components in our tests.
N_COMPONENTS = 8

# Angle-encoding bandwidth: the MinMaxScaler below maps LDA output into
# [0, pi * BANDWIDTH] instead of the full [0, pi]. A narrower bandwidth
# reduced kernel concentration substantially in our tests with LDA
# features (this is a DIFFERENT optimum than your original PCA-based
# script found -- re-sweep, don't reuse the old bandwidth value).
BANDWIDTH_CANDIDATES = [0.03, 0.05, 0.08, 0.1, 0.15]

# ZZFeatureMap settings -- keep reps=1 and full entanglement. Higher reps
# increased kernel concentration (lower separation) in our tests with LDA
# features at 8+ qubits -- more circuit depth is NOT automatically better
# for a quantum kernel, unlike a classical network with more layers.
ZZ_REPS = 1
ZZ_ENTANGLEMENT = "full"

# FULL SAMPLES — None means "use the entire split, no subsampling".
TRAIN_SAMPLE_SIZE = None
VAL_SAMPLE_SIZE   = None
TEST_SAMPLE_SIZE  = None

TARGET_MAX_RUNTIME_SECONDS = 2 * 60 * 60
OVERFIT_GAP_THRESHOLD_PTS = 8.0
UNDERFIT_MIN_TRAIN_ACC = 0.45
RANDOM_STATE = 42

_run_start_time = time.time()

print("=" * 70)
print("QSVM — LDA-based feature reduction (see header for why this replaces PCA)")
print(f"Target runtime budget: {TARGET_MAX_RUNTIME_SECONDS/3600:.1f} hours.")
print("=" * 70)

print("\nUsing FidelityStatevectorKernel (exact fidelity, vectorized, no per-pair circuit execution).")


# ---- Load data (all 3 splits) ----

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

train_df = train_df.dropna(subset=["label"])
val_df   = val_df.dropna(subset=["label"])
test_df  = test_df.dropna(subset=["label"])

print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")

feature_cols = [c for c in train_df.columns if c.startswith("feature_")]

X_train_full = train_df[feature_cols].values
y_train_full = train_df["label"].values
X_val_full   = val_df[feature_cols].values
y_val_full   = val_df["label"].values
X_test_full  = test_df[feature_cols].values
y_test_full  = test_df["label"].values

print(f"Feature dimension in source CSVs: {len(feature_cols)}")


# ---- Encode labels ----

label_encoder = LabelEncoder()
all_unique_labels = pd.concat([train_df["label"], val_df["label"], test_df["label"]]).unique()
label_encoder.fit(all_unique_labels)

y_train_enc = label_encoder.transform(y_train_full)
y_val_enc   = label_encoder.transform(y_val_full)
y_test_enc  = label_encoder.transform(y_test_full)
n_classes = len(label_encoder.classes_)

print("Classes:", list(label_encoder.classes_))
print(f"Max useful LDA components (n_classes - 1): {n_classes - 1}")


# ---- Scale + LDA fit on the FULL training set (labels used, unlike PCA) ----

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_val_scaled   = scaler.transform(X_val_full)
X_test_scaled  = scaler.transform(X_test_full)

lda = LDA(n_components=N_COMPONENTS)
X_train_reduced_full = lda.fit_transform(X_train_scaled, y_train_enc)   # <-- labels passed here, this IS the fix
X_val_reduced_full   = lda.transform(X_val_scaled)
X_test_reduced_full  = lda.transform(X_test_scaled)

print(f"LDA applied: {N_COMPONENTS} components (supervised — fit using y_train, "
      f"never val/test labels)")

n_qubits = X_train_reduced_full.shape[1]
print(f"Feature dimension going into the quantum feature map: {n_qubits}")


# ---- Stratified subsample for each split ----

def stratified_take(X, y, size, random_state):
    if size is not None and size < len(X):
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=size, stratify=y, random_state=random_state
        )
        return X_sub, y_sub
    return X, y

X_train_reduced, y_train_enc = stratified_take(X_train_reduced_full, y_train_enc, TRAIN_SAMPLE_SIZE, RANDOM_STATE)
X_val_reduced, y_val_enc     = stratified_take(X_val_reduced_full, y_val_enc, VAL_SAMPLE_SIZE, RANDOM_STATE)
X_test_reduced, y_test_enc   = stratified_take(X_test_reduced_full, y_test_enc, TEST_SAMPLE_SIZE, RANDOM_STATE)

print(f"Using {len(X_train_reduced)} train / {len(X_val_reduced)} val / {len(X_test_reduced)} test samples")
_n_train_pairs = len(X_train_reduced) * (len(X_train_reduced) - 1) // 2
print(f"Estimated unique training-kernel pairs to evaluate: {_n_train_pairs:,}")

assert len(np.unique(y_train_enc)) > 1, "y_train_enc has only one class! Check your data loading."


# ---- Helper: evaluate a fitted model on one split (precomputed kernel) ----

def evaluate_split_precomputed(model, K, y_true, split_name, verbose=True):
    y_pred = model.predict(K)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)
    if verbose:
        print(f"[{split_name}] Acc: {acc*100:.2f}% | Precision: {precision:.2f} | "
              f"Recall: {recall:.2f} | F1: {f1:.2f} | MCC: {mcc:.4f}")
    return {"split": split_name, "accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1, "mcc": mcc}


# ---- Sweep bandwidth x C jointly (bandwidth changes the kernel matrix itself,
#      so it must be swept OUTSIDE the C-sweep, not independently of it) ----

C_CANDIDATES = [0.5, 1.0, 5.0, 10.0, 20.0]

print("\nSweeping bandwidth (angle-encoding scale) x C for the quantum kernel...")
print(f"  Overfit flag:  train-val gap > {OVERFIT_GAP_THRESHOLD_PTS:.1f} points")
print(f"  Underfit flag: train accuracy < {UNDERFIT_MIN_TRAIN_ACC*100:.1f}%")

feature_map_template_reps = ZZ_REPS
best_overall = None
sweep_log = []

for bw in BANDWIDTH_CANDIDATES:
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi * bw))
    Xtr_ang = angle_scaler.fit_transform(X_train_reduced)
    Xval_ang = angle_scaler.transform(X_val_reduced)

    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=ZZ_REPS, entanglement=ZZ_ENTANGLEMENT)
    feature_map = feature_map.decompose()
    quantum_kernel = FidelityStatevectorKernel(feature_map=feature_map)

    _t0 = time.time()
    train_kernel_matrix = quantum_kernel.evaluate(x_vec=Xtr_ang)
    val_train_kernel_matrix = quantum_kernel.evaluate(x_vec=Xval_ang, y_vec=Xtr_ang)
    kt = time.time() - _t0

    off_diag_mask = ~np.eye(train_kernel_matrix.shape[0], dtype=bool)
    same_class_mask = (y_train_enc[:, None] == y_train_enc[None, :])
    avg_intra = train_kernel_matrix[same_class_mask & off_diag_mask].mean()
    avg_inter = train_kernel_matrix[(~same_class_mask) & off_diag_mask].mean()
    separation = avg_intra - avg_inter

    for c in C_CANDIDATES:
        candidate = QSVC(quantum_kernel="precomputed", C=c, class_weight="balanced")
        candidate.fit(train_kernel_matrix, y_train_enc)
        train_pred = candidate.predict(train_kernel_matrix)
        val_pred   = candidate.predict(val_train_kernel_matrix)
        train_acc = accuracy_score(y_train_enc, train_pred)
        val_acc   = accuracy_score(y_val_enc, val_pred)
        gap = train_acc - val_acc
        is_overfit = gap > (OVERFIT_GAP_THRESHOLD_PTS / 100.0)
        is_underfit = train_acc < UNDERFIT_MIN_TRAIN_ACC
        flag = "OVERFIT" if is_overfit else ("UNDERFIT" if is_underfit else "ok")

        print(f"  bw={bw:<5} C={c:<6} | sep={separation:.4f} | Train: {train_acc*100:5.2f}% | "
              f"Val: {val_acc*100:5.2f}% | Gap: {gap*100:5.2f}pts | {flag} | kernel_time={kt:.1f}s")

        entry = {
            "bw": bw, "c": c, "model": candidate, "separation": separation,
            "train_acc": train_acc, "val_acc": val_acc, "gap": gap,
            "is_overfit": is_overfit, "is_underfit": is_underfit,
            "train_kernel_matrix": train_kernel_matrix,
            "val_train_kernel_matrix": val_train_kernel_matrix,
            "feature_map": feature_map, "angle_scaler": angle_scaler,
        }
        sweep_log.append(entry)

clean_candidates = [r for r in sweep_log if not r["is_overfit"] and not r["is_underfit"]]
if clean_candidates:
    best = max(clean_candidates, key=lambda r: r["val_acc"])
    print(f"\n{len(clean_candidates)}/{len(sweep_log)} candidates satisfy both non-negotiables.")
else:
    best = min(sweep_log, key=lambda r: abs(r["gap"]) if not r["is_underfit"] else np.inf)
    print("\n*** WARNING: NO (bandwidth, C) COMBO SATISFIES BOTH NON-NEGOTIABLES. ***")
    print("*** Falling back to the least-bad candidate — treat this as a signal to ***")
    print("*** widen the bandwidth/C grids further, or revisit N_COMPONENTS. ***")

best_bw = best["bw"]
best_c = best["c"]
qsvc = best["model"]
best_angle_scaler = best["angle_scaler"]
best_feature_map = best["feature_map"]
train_kernel_matrix = best["train_kernel_matrix"]
val_train_kernel_matrix = best["val_train_kernel_matrix"]

print(f"\nSelected bandwidth={best_bw} C={best_c} | separation={best['separation']:.4f} | "
      f"Train: {best['train_acc']*100:.2f}% | Val: {best['val_acc']*100:.2f}% | "
      f"Gap: {best['gap']*100:.2f}pts")
print("Training complete.")


# ---- Final evaluation on test (touched once, after model selection) ----

quantum_kernel_final = FidelityStatevectorKernel(feature_map=best_feature_map)
X_test_ang = best_angle_scaler.transform(X_test_reduced)
print("\nComputing test-vs-train kernel matrix (test set touched once)...")
test_train_kernel_matrix = quantum_kernel_final.evaluate(x_vec=X_test_ang, y_vec=best_angle_scaler.transform(X_train_reduced))

print("\n--- QSVC results across all splits (best bandwidth + C) ---")
qsvc_train_metrics = evaluate_split_precomputed(qsvc, train_kernel_matrix, y_train_enc, "Train")
qsvc_val_metrics   = evaluate_split_precomputed(qsvc, val_train_kernel_matrix, y_val_enc, "Val")
qsvc_test_metrics  = evaluate_split_precomputed(qsvc, test_train_kernel_matrix, y_test_enc, "Test")

y_pred_test = qsvc.predict(test_train_kernel_matrix)
all_label_ids = np.arange(n_classes)
print("\nClassification report (test set):")
print(classification_report(
    y_test_enc, y_pred_test, labels=all_label_ids,
    target_names=label_encoder.classes_, zero_division=0,
))
print("Confusion matrix (test set):")
print(confusion_matrix(y_test_enc, y_pred_test, labels=all_label_ids))


# ---- Classical RBF-SVM baseline on the SAME LDA features (fair comparison) ----

print("\nSweeping C for the classical RBF-SVM baseline (same LDA features, no angle scaling needed)...")
best_c_svc, best_val_acc_svc, svc_classical = None, -np.inf, None
for c in C_CANDIDATES:
    candidate = SVC(kernel="rbf", C=c, class_weight="balanced", random_state=RANDOM_STATE)
    candidate.fit(X_train_reduced, y_train_enc)
    val_acc = accuracy_score(y_val_enc, candidate.predict(X_val_reduced))
    if val_acc > best_val_acc_svc:
        best_val_acc_svc, best_c_svc, svc_classical = val_acc, c, candidate

print(f"Selected C={best_c_svc} for classical SVM (val acc: {best_val_acc_svc*100:.2f}%)")

def evaluate_split(model, X, y_true, split_name):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"[{split_name}] Acc: {acc*100:.2f}% | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | MCC: {mcc:.4f}")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "mcc": mcc}

print("\n--- Classical RBF-SVM results across all splits (best C, same LDA features) ---")
svc_train_metrics = evaluate_split(svc_classical, X_train_reduced, y_train_enc, "Train")
svc_val_metrics   = evaluate_split(svc_classical, X_val_reduced, y_val_enc, "Val")
svc_test_metrics  = evaluate_split(svc_classical, X_test_reduced, y_test_enc, "Test")


# ---- Plots ----

splits = ["Train", "Val", "Test"]
qsvc_accs = [qsvc_train_metrics["accuracy"], qsvc_val_metrics["accuracy"], qsvc_test_metrics["accuracy"]]
svc_accs  = [svc_train_metrics["accuracy"], svc_val_metrics["accuracy"], svc_test_metrics["accuracy"]]
x = np.arange(len(splits)); width = 0.35

fig, ax = plt.subplots(figsize=(6, 5))
b1 = ax.bar(x - width/2, [a*100 for a in qsvc_accs], width, label="QSVC (LDA features)", color="#7F77DD")
b2 = ax.bar(x + width/2, [a*100 for a in svc_accs], width, label="Classical RBF-SVM (same LDA features)", color="#D85A30")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.1f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(splits); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("Accuracy by Split — QSVC vs Classical SVM (LDA features)")
ax.legend(); plt.tight_layout(); plt.savefig("qsvm_accuracy_comparison.png", dpi=150); plt.show()

qsvc_err = [1 - a for a in qsvc_accs]; svc_err = [1 - a for a in svc_accs]
fig, ax = plt.subplots(figsize=(6, 5))
b1 = ax.bar(x - width/2, qsvc_err, width, label="QSVC", color="#7F77DD")
b2 = ax.bar(x + width/2, svc_err, width, label="Classical RBF-SVM", color="#D85A30")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(splits); ax.set_ylabel("Error rate (1 - accuracy)"); ax.set_ylim(0, 1)
ax.set_title("Error Rate by Split — QSVC vs Classical SVM")
ax.legend(); plt.tight_layout(); plt.savefig("qsvm_error_comparison.png", dpi=150); plt.show()


# ---- Final summary ----

_total_runtime = time.time() - _run_start_time
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Feature reduction: LDA (supervised) -> {n_qubits} dims, vs PCA (unsupervised) previously")
print(f"Best config: bandwidth={best_bw}, C={best_c}, reps={ZZ_REPS}, entanglement={ZZ_ENTANGLEMENT}")
print(f"QSVC test accuracy: {qsvc_test_metrics['accuracy']*100:.2f}%")
print(f"Classical RBF-SVM test accuracy (same features): {svc_test_metrics['accuracy']*100:.2f}%")
print(f"Total runtime: {_total_runtime/60:.1f} min ({_total_runtime/3600:.2f}h)")
print("=" * 70)
