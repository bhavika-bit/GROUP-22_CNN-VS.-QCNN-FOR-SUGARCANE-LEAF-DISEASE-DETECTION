# ======================================================
# QSVM — NCA-based feature reduction (third DR method, alongside PCA and LDA)
#
# ============================================================
# WHY NCA, AND HOW IT DIFFERS FROM LDA (read this first):
#
# Your current best model uses LDA -> 8 dims (val 92.43%, test 90.75%,
# see qsvm_lda.py). This script swaps LDA for NCA (Neighborhood Components
# Analysis, sklearn.neighbors.NeighborhoodComponentsAnalysis) as a third,
# genuinely different comparison point:
#
#   PCA : unsupervised, maximizes variance (no label info at all)
#   LDA : supervised, assumes roughly Gaussian class-conditional blobs
#         (maximizes between-class / within-class scatter ratio)
#   NCA : supervised, NO Gaussian assumption -- directly minimizes soft
#         k-NN classification error via gradient-based optimization
#
# IMPORTANT NUANCE: NCA still learns a LINEAR projection matrix, same as
# PCA and LDA. It is not a "nonlinear" method -- what's different is the
# OBJECTIVE it optimizes (classification-native, non-parametric) rather
# than LDA's Gaussian-scatter objective. This script tests "does a
# different supervised objective beat LDA's", not "does nonlinearity help".
# If you want to test nonlinearity specifically, Kernel PCA is the more
# apt method -- that's a separate experiment from this one.
#
# PRACTICAL DIFFERENCE FROM LDA: unlike LDA, NCA is NOT capped at
# n_classes - 1 components (LDA maxes out at 12 for your 13 classes). NCA
# can be fit to any N_COMPONENTS you choose. N_COMPONENTS is kept at 8
# below to match your LDA config for a fair apples-to-apples comparison at
# the same qubit count -- not because of a technical ceiling.
#
# RUNTIME NOTE: LDA has a closed-form solution (near-instant eigendecomp).
# NCA is iterative (gradient-based, full-batch L-BFGS in sklearn) and
# genuinely slower to fit. Benchmarked directly on data of this exact shape
# (6750 rows x 1280 dims): ~33s for 20 iterations, not yet converged.
# Default max_iter=50 extrapolates to roughly 80-90s for the NCA fit alone.
# This is still trivial next to your quantum kernel computation time and
# your 2-hour budget -- flagged here only so the wait after "Fitting NCA..."
# isn't mistaken for a hang.
#
# HONEST CAVEAT (same as the LDA script): quantum kernels built from
# ZZFeatureMap-style circuits are known to suffer "kernel concentration"
# (Thanasilp et al.). Whatever NCA's classical accuracy turns out to be,
# the QSVM's accuracy ceiling is still bounded by this effect, independent
# of which DR method fed it. If QSVM underperforms the classical NCA+SVM
# baseline, that gap -- not the DR method choice -- is the more likely
# explanation, and it's a legitimate finding to report either way.
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
#  runs fast on plain CPU at this qubit count; see the LDA script's header
#  note for the full explanation of why GPU doesn't help here.)

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
# NOTE: NeighborhoodComponentsAnalysis lives in sklearn.neighbors, NOT
# sklearn.manifold -- easy mistake, this import path is verified correct.
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
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

# ---- THE MAIN SWAP: NCA instead of LDA ----
# NCA needs labels to fit (like LDA, unlike PCA), so it's fit on
# (X_train, y_train) only -- never on val/test, to avoid label leakage.
# Unlike LDA, NCA has no n_classes-1 ceiling -- 8 is chosen here to match
# your LDA config for a fair comparison, not a technical limit.
N_COMPONENTS = 8

# NCA is iterative -- max_iter controls how many L-BFGS steps it takes.
# Higher = better-converged projection, but slower to fit. 50 is sklearn's
# default; raise if you see the "did not converge" warning and have time
# budget to spare.
NCA_MAX_ITER = 50

# Angle-encoding bandwidth: the MinMaxScaler below maps NCA output into
# [0, pi * BANDWIDTH] instead of the full [0, pi]. Re-sweep this for NCA
# rather than reusing the LDA script's chosen value -- NCA's output scale
# and distribution differ from LDA's, so the kernel-concentration-optimal
# bandwidth is not guaranteed to match.
BANDWIDTH_CANDIDATES = [0.03, 0.05, 0.08, 0.1, 0.15]

# ZZFeatureMap settings -- same starting point as the LDA script (reps=1,
# full entanglement). Re-verify this is still the low-concentration choice
# for NCA features specifically before trusting it blindly.
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
print("QSVM — NCA-based feature reduction (third DR method vs PCA and LDA)")
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
print(f"(For reference — LDA's ceiling would be {n_classes - 1} components. "
      f"NCA has no such ceiling; N_COMPONENTS={N_COMPONENTS} is a chosen match, not a limit.)")


# ---- Scale + NCA fit on the FULL training set (labels used, unlike PCA) ----

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_val_scaled   = scaler.transform(X_val_full)
X_test_scaled  = scaler.transform(X_test_full)

nca = NCA(n_components=N_COMPONENTS, max_iter=NCA_MAX_ITER, random_state=RANDOM_STATE, verbose=1)

print(f"\nFitting NCA ({N_COMPONENTS} components, max_iter={NCA_MAX_ITER}) — "
      f"this is iterative and slower than LDA's closed-form fit, expect ~1-2 min...")
_t0 = time.time()
X_train_reduced_full = nca.fit_transform(X_train_scaled, y_train_enc)   # <-- labels passed here, same as LDA
X_val_reduced_full   = nca.transform(X_val_scaled)
X_test_reduced_full  = nca.transform(X_test_scaled)
print(f"NCA fit + transform took {time.time()-_t0:.1f}s")

print(f"NCA applied: {N_COMPONENTS} components (supervised — fit using y_train, "
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


# ---- Classical RBF-SVM baseline on the SAME NCA features (fair comparison) ----

print("\nSweeping C for the classical RBF-SVM baseline (same NCA features, no angle scaling needed)...")
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

print("\n--- Classical RBF-SVM results across all splits (best C, same NCA features) ---")
svc_train_metrics = evaluate_split(svc_classical, X_train_reduced, y_train_enc, "Train")
svc_val_metrics   = evaluate_split(svc_classical, X_val_reduced, y_val_enc, "Val")
svc_test_metrics  = evaluate_split(svc_classical, X_test_reduced, y_test_enc, "Test")


# ---- Plots ----

splits = ["Train", "Val", "Test"]
qsvc_accs = [qsvc_train_metrics["accuracy"], qsvc_val_metrics["accuracy"], qsvc_test_metrics["accuracy"]]
svc_accs  = [svc_train_metrics["accuracy"], svc_val_metrics["accuracy"], svc_test_metrics["accuracy"]]
x = np.arange(len(splits)); width = 0.35

fig, ax = plt.subplots(figsize=(6, 5))
b1 = ax.bar(x - width/2, [a*100 for a in qsvc_accs], width, label="QSVC (NCA features)", color="#7F77DD")
b2 = ax.bar(x + width/2, [a*100 for a in svc_accs], width, label="Classical RBF-SVM (same NCA features)", color="#D85A30")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.1f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(splits); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("Accuracy by Split — QSVC vs Classical SVM (NCA features)")
ax.legend(); plt.tight_layout(); plt.savefig("qsvm_nca_accuracy_comparison.png", dpi=150); plt.show()

qsvc_err = [1 - a for a in qsvc_accs]; svc_err = [1 - a for a in svc_accs]
fig, ax = plt.subplots(figsize=(6, 5))
b1 = ax.bar(x - width/2, qsvc_err, width, label="QSVC", color="#7F77DD")
b2 = ax.bar(x + width/2, svc_err, width, label="Classical RBF-SVM", color="#D85A30")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(splits); ax.set_ylabel("Error rate (1 - accuracy)"); ax.set_ylim(0, 1)
ax.set_title("Error Rate by Split — QSVC vs Classical SVM (NCA features)")
ax.legend(); plt.tight_layout(); plt.savefig("qsvm_nca_error_comparison.png", dpi=150); plt.show()


# ---- Final summary ----

_total_runtime = time.time() - _run_start_time
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Feature reduction: NCA (supervised, non-parametric objective) -> {n_qubits} dims")
print(f"  Compare against: LDA (supervised, Gaussian objective) and PCA (unsupervised)")
print(f"Best config: bandwidth={best_bw}, C={best_c}, reps={ZZ_REPS}, entanglement={ZZ_ENTANGLEMENT}")
print(f"QSVC test accuracy: {qsvc_test_metrics['accuracy']*100:.2f}%")
print(f"Classical RBF-SVM test accuracy (same features): {svc_test_metrics['accuracy']*100:.2f}%")
print(f"Total runtime: {_total_runtime/60:.1f} min ({_total_runtime/3600:.2f}h)")
print("=" * 70)
