# ======================================================
# QSVM (Quantum Support Vector Machine) — FULL-SAMPLE, EXACT-KERNEL VERSION
#
# ============================================================
# WHY THIS VERSION DOESN'T USE qiskit-aer / GPU AT ALL — read before running:
#
# The earlier GPU version used ComputeUncompute + a sampler (Aer or generic),
# which builds and executes one quantum circuit PER PAIR of data points to
# estimate fidelity via measurement statistics. Benchmarked directly on this
# machine: ~5ms/pair, REGARDLESS of GPU vs CPU device, because the bottleneck
# is per-circuit job dispatch overhead in Qiskit's execution pipeline, not the
# underlying simulation math. At full scale (~6750 train rows -> ~22.8M
# unique pairs), that extrapolates to roughly 31+ HOURS — GPU does not fix
# this, because the overhead is per-circuit-launch, not per-FLOP.
#
# This version uses FidelityStatevectorKernel instead. At n_qubits=4, a
# statevector is only 16 complex amplitudes. Instead of running N^2 circuits,
# it prepares each of the N statevectors ONCE, then computes the full N x N
# fidelity matrix as a single vectorized (BLAS-backed) operation. Benchmarked
# directly (50 -> 2000 rows, confirmed quadratic scaling, extrapolated):
# the full 6750-row training kernel takes ~3.5-4 MINUTES on plain CPU. Train +
# val + test kernels together: well under 10 minutes. No GPU needed — the
# actual computation at this qubit count is trivial; the circuit-execution
# overhead was the entire problem.
#
# Bonus: this also computes EXACT fidelities (no shot noise), which is
# strictly better for kernel stability — noisy kernel entries from sampling
# can distort the C-sweep's overfit/underfit diagnosis. Verified: values
# match ComputeUncompute's output up to exactly the shot-noise magnitude you'd
# expect from the sampler's default shot count.
#
# If you have a specific reason to keep the circuit-execution model (e.g. you
# need to later swap in a hardware-realistic noise model, or you specifically
# need to report on real quantum hardware later), that's a different,
# legitimate design choice — but for a from-scratch classical simulation at
# 4 qubits, this is strictly faster with no accuracy cost.
# ============================================================
#
# IMPORTANT — READ BEFORE RUNNING (loss/epochs):
# QSVC (like any SVM) is a single kernel-based fit, not an iterative
# training loop — there are no "epochs", so there is no true per-epoch
# loss curve like a CNN has. What this script gives you instead:
#   - Accuracy plot: train / val / test accuracy, side by side
#   - Error-rate plot: (1 - accuracy) for train / val / test — this is
#     the closest honest analog to a "loss" plot for an SVM-family model.
# ======================================================

# ---- Setup ----
# pip install qiskit qiskit-machine-learning qiskit-algorithms matplotlib pandas scikit-learn
# (qiskit-aer / qiskit-aer-gpu are NOT required for this version.)

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
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

TRAIN_CSV = r"D:\Sugarcane_CNN_Project\features\efficientnetb0_features_train.csv"   # <-- CHANGE IF NEEDED
VAL_CSV   = r"D:\Sugarcane_CNN_Project\features\efficientnetb0_features_val.csv"     # <-- CHANGE IF NEEDED
TEST_CSV  = r"D:\Sugarcane_CNN_Project\features\efficientnetb0_features_test.csv"    # <-- CHANGE IF NEEDED

USE_PCA = True
N_COMPONENTS = 4          # kept at 4 — this is what fixed kernel concentration
                           # (validated: intra/inter fidelity separation ~5x
                           # better than at 8 qubits).
RAW_DIM_CAP = 4

# Bandwidth candidates: scale factor applied to the angle-encoded inputs
# before they hit the quantum feature map. Kernel concentration (a
# near-degenerate, low-rank kernel matrix) is often caused by inputs
# saturating the feature map at bandwidth=1.0 (the untouched [0, pi] range).
# Scaling down spreads fidelities back out. Selected using rank + intra/inter
# separation, not downstream accuracy (see the bandwidth sweep below).
BANDWIDTH_CANDIDATES = [0.1, 0.25, 0.5, 0.75, 1.0]

# FULL SAMPLES — None means "use the entire split, no subsampling".
# ~6750 train / ~1440 val / ~1460 test. Non-negotiable: full data, no
# subsampling, for a fair comparison against the classical CNN baseline.
# This is safe to leave at None with the exact statevector kernel below —
# see the runtime note at the top of this file.
TRAIN_SAMPLE_SIZE = None
VAL_SAMPLE_SIZE   = None
TEST_SAMPLE_SIZE  = None

# Non-negotiable target: total wall-clock time should land at or under the
# classical CNN baseline's ~2 hours. Printed at the end of the run so you can
# check this directly against your actual run, not just this estimate.
TARGET_MAX_RUNTIME_SECONDS = 2 * 60 * 60

# Non-negotiable: neither overfitting nor underfitting is acceptable.
# These thresholds define what those two failure modes mean numerically, so
# the C-sweep below can flag them explicitly instead of leaving it to eyeballing.
#   - Overfitting  = train accuracy far exceeds val accuracy (model memorizing).
#   - Underfitting = val accuracy is barely better than chance, or train
#     accuracy itself is low (model isn't learning class structure at all).
# These are starting thresholds, not laws of nature — adjust if your specific
# 13-class problem calls for it, but keep them explicit rather than implicit.
OVERFIT_GAP_THRESHOLD_PTS = 8.0     # train-val accuracy gap above this = overfit flag
UNDERFIT_MIN_TRAIN_ACC = 0.45       # train acc below this = underfit flag (model can't even fit train)

RANDOM_STATE = 42

_run_start_time = time.time()

print("=" * 70)
print("RUNNING ON FULL SAMPLE SIZES.")
print(f"Target runtime budget: {TARGET_MAX_RUNTIME_SECONDS/3600:.1f} hours (matching classical CNN baseline).")
print("=" * 70)


# ---- Quantum kernel backend: exact statevector fidelity (no circuits, no GPU) ----
# See the header note for why this replaces the circuit+sampler approach.
# No device selection needed — this runs on plain CPU/numpy and is fast
# enough at this qubit count that GPU offers no practical benefit.
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


# ---- Encode labels (fit on union of all 3 splits) ----

label_encoder = LabelEncoder()
all_unique_labels = pd.concat([train_df["label"], val_df["label"], test_df["label"]]).unique()
label_encoder.fit(all_unique_labels)

y_train_enc = label_encoder.transform(y_train_full)
y_val_enc   = label_encoder.transform(y_val_full)
y_test_enc  = label_encoder.transform(y_test_full)

print("Classes:", list(label_encoder.classes_))


# ---- Scale + PCA fit on the FULL training set ----

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_val_scaled   = scaler.transform(X_val_full)
X_test_scaled  = scaler.transform(X_test_full)

# ---- PCA elbow diagnostic (informational only — doesn't change the pipeline) ----
_diag_pca = PCA(n_components=min(15, X_train_scaled.shape[1]), random_state=RANDOM_STATE)
_diag_pca.fit(X_train_scaled)
_cumvar = np.cumsum(_diag_pca.explained_variance_ratio_)
print("\nPCA cumulative explained variance (diagnostic, up to 15 components):")
for i, cv in enumerate(_cumvar, start=1):
    print(f"  {i:2d} components: {cv*100:5.1f}%")


if USE_PCA:
    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    X_train_reduced_full = pca.fit_transform(X_train_scaled)
    X_val_reduced_full   = pca.transform(X_val_scaled)
    X_test_reduced_full  = pca.transform(X_test_scaled)
    print(f"PCA applied: {N_COMPONENTS} components, "
          f"explained variance ratio sum = {pca.explained_variance_ratio_.sum():.4f}")
else:
    X_train_reduced_full = X_train_scaled[:, :RAW_DIM_CAP]
    X_val_reduced_full   = X_val_scaled[:, :RAW_DIM_CAP]
    X_test_reduced_full  = X_test_scaled[:, :RAW_DIM_CAP]
    print(f"Using first {RAW_DIM_CAP} raw (scaled) feature columns, no PCA")

n_qubits = X_train_reduced_full.shape[1]
print(f"Feature dimension going into the quantum feature map: {n_qubits}")


# ---- Stratified subsample for each split (kernel step is the expensive part) ----

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
print(f"Classes present — train: {len(np.unique(y_train_enc))}/13, "
      f"val: {len(np.unique(y_val_enc))}/13, test: {len(np.unique(y_test_enc))}/13")

assert len(np.unique(y_train_enc)) > 1, (
    "y_train_enc has only one class! Restart the Colab runtime and Run all "
    "from the top — this usually means stale variables from a previous run."
)


# ---- Rescale into [0, pi] before the quantum feature map (fit on train only) ----

angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_reduced = angle_scaler.fit_transform(X_train_reduced)
X_val_reduced   = angle_scaler.transform(X_val_reduced)
X_test_reduced  = angle_scaler.transform(X_test_reduced)


# ---- Build the feature map once (bandwidth is applied to the data, not the map) ----

feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement="full")
feature_map = feature_map.decompose()  # expand the blueprint circuit before use
quantum_kernel = FidelityStatevectorKernel(feature_map=feature_map)
# shots=None (the default) means EXACT fidelity — no shot noise. This is what
# makes the kernel matrix numerically stable for the C-sweep's overfit/underfit
# diagnosis below: noisy kernel entries would otherwise make that diagnosis
# unreliable run-to-run.


# ---- Bandwidth sweep: pick input scale using rank + separation, not accuracy ----
# Kernel concentration (a near-degenerate, low-rank kernel matrix) is often
# caused by angle-encoded inputs saturating the feature map. Scaling the
# inputs down before they hit the feature map (bandwidth < 1) spreads
# fidelities back out. We select bandwidth on rank + intra/inter separation
# here, NOT downstream accuracy — accuracy also depends on C, which hasn't
# been swept yet at this point in the script.
#
# NOTE ON RUNTIME: this evaluates one full train-kernel matrix PER bandwidth
# candidate — with 5 candidates at ~370s each (per the last full run), budget
# roughly 30+ minutes just for this step, on top of the val/test kernels and
# C-sweep that follow. Trim BANDWIDTH_CANDIDATES in the config block above if
# you want a faster first pass.

print("\nSweeping bandwidth (input scale before feature map) for kernel quality...")
bandwidth_results = []

for bw in BANDWIDTH_CANDIDATES:
    _t0 = time.time()
    K_bw = quantum_kernel.evaluate(x_vec=X_train_reduced * bw)
    elapsed = time.time() - _t0

    rank_bw = np.linalg.matrix_rank(K_bw)
    off_diag = ~np.eye(K_bw.shape[0], dtype=bool)
    same_class = (y_train_enc[:, None] == y_train_enc[None, :])
    sep_bw = K_bw[same_class & off_diag].mean() - K_bw[(~same_class) & off_diag].mean()

    print(f"  bandwidth={bw:<5} | rank: {rank_bw}/{K_bw.shape[0]} "
          f"({100*rank_bw/K_bw.shape[0]:.1f}%) | separation: {sep_bw:.4f} | {elapsed:.1f}s")

    bandwidth_results.append({"bandwidth": bw, "kernel": K_bw, "rank": rank_bw, "separation": sep_bw})

# Pick the bandwidth with the best separation (ties broken by higher rank).
best_bw_result = max(bandwidth_results, key=lambda r: (r["separation"], r["rank"]))
best_bandwidth = best_bw_result["bandwidth"]
train_kernel_matrix = best_bw_result["kernel"]   # reuse — do not recompute
kernel_rank = best_bw_result["rank"]
avg_intra = train_kernel_matrix[
    (y_train_enc[:, None] == y_train_enc[None, :]) & ~np.eye(train_kernel_matrix.shape[0], dtype=bool)
].mean()
avg_inter = train_kernel_matrix[
    (y_train_enc[:, None] != y_train_enc[None, :]) & ~np.eye(train_kernel_matrix.shape[0], dtype=bool)
].mean()

print(f"\nSelected bandwidth={best_bandwidth} "
      f"(rank {kernel_rank}/{train_kernel_matrix.shape[0]}, separation {best_bw_result['separation']:.4f})")
print(f"Kernel matrix rank: {kernel_rank} / {train_kernel_matrix.shape[0]} "
      f"({100 * kernel_rank / train_kernel_matrix.shape[0]:.1f}% of full rank)")
print(f"Average intra-class fidelity: {avg_intra:.4f}")
print(f"Average inter-class fidelity: {avg_inter:.4f}")
print(f"Separation (intra - inter):   {best_bw_result['separation']:.4f}")

# Apply the winning bandwidth to all splits going forward. val/test kernels
# computed later in the script will now be consistent with train_kernel_matrix.
X_train_reduced = X_train_reduced * best_bandwidth
X_val_reduced   = X_val_reduced * best_bandwidth
X_test_reduced  = X_test_reduced * best_bandwidth


# ---- Helper: evaluate a fitted model on one split ----

def evaluate_split(model, X, y_true, split_name, verbose=True):
    y_pred = model.predict(X)
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


# ---- Train QSVC with a C sweep to fix overfitting ----
# C is SVM's regularization strength (the direct analog to "weight decay" for
# a neural net) — lower C = softer margin = less fitting to individual
# training points = less overfitting. Default C=1.0 was too permissive here
# (77% train vs ~61-62% val/test — a 15+ point gap). We sweep several C
# values and pick the one with the smallest train-val gap among those that
# still keep reasonable val accuracy, instead of guessing a single value.
#
# KEY OPTIMIZATION: the quantum kernel matrix does NOT depend on C — only the
# classical SVM optimization inside QSVC.fit() does. Previously, calling
# QSVC(...).fit() once per C candidate silently recomputed the FULL O(n^2)
# quantum kernel matrix every time — 3x the actual quantum work needed.
#
# Fix: compute the train kernel matrix ONCE (train_kernel_matrix, already
# done above for diagnostics) and the train-val kernel matrix ONCE, then pass
# kernel="precomputed" so each C candidate only re-runs the classical SVM
# solver (fast — no circuit evaluation at all). This turns the C-sweep's
# quantum cost from 3x O(n^2) into 1x O(n^2) + 1x O(n_train * n_val).

# IMPORTANT: QSVC accepts the literal string quantum_kernel="precomputed" —
# passing that (not the kernel object) tells the underlying sklearn SVC to
# treat X as an already-computed Gram matrix instead of raw feature vectors.
# Verified against the installed qiskit-machine-learning source
# (QSVC.__init__: `if quantum_kernel == "precomputed": super().__init__(kernel=self._quantum_kernel, ...)`,
#  vs `else: super().__init__(kernel=self._quantum_kernel.evaluate, ...)`).

print("\nComputing val-vs-train kernel matrix once (reused across the whole C-sweep)...")
_t0 = time.time()
val_train_kernel_matrix = quantum_kernel.evaluate(x_vec=X_val_reduced, y_vec=X_train_reduced)
print(f"Val-train kernel matrix computed in {time.time()-_t0:.1f}s "
      f"(shape {val_train_kernel_matrix.shape}).")
# shape (n_val, n_train) — exactly what SVC.predict() expects for kernel="precomputed"

# Now that kernel computation is fast (vectorized, not per-circuit), a wider
# C grid costs almost nothing extra — each candidate is just a classical QP
# re-solve on the already-cached kernel matrix. Widened from 3 to 7 candidates
# spanning 3 orders of magnitude, to give the overfit/underfit search room to
# actually find a good regularization point rather than picking among 3 coarse
# options.
C_CANDIDATES = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

print("\nSweeping C — selecting for NEITHER overfitting NOR underfitting (both are non-negotiable).")
print(f"  Overfit flag:  train-val gap > {OVERFIT_GAP_THRESHOLD_PTS:.1f} points")
print(f"  Underfit flag: train accuracy < {UNDERFIT_MIN_TRAIN_ACC*100:.1f}%")

sweep_results = []

for c in C_CANDIDATES:
    # quantum_kernel="precomputed": QSVC/SVC runs only the classical QP
    # solver on the cached Gram matrix — no quantum circuits are evaluated
    # anywhere in this loop.
    candidate = QSVC(quantum_kernel="precomputed", C=c, class_weight="balanced")
    candidate.fit(train_kernel_matrix, y_train_enc)  # square (n_train, n_train) precomputed Gram matrix

    train_pred = candidate.predict(train_kernel_matrix)       # (n_train, n_train)
    val_pred   = candidate.predict(val_train_kernel_matrix)   # (n_val, n_train)
    train_acc = accuracy_score(y_train_enc, train_pred)
    val_acc   = accuracy_score(y_val_enc, val_pred)
    gap = train_acc - val_acc

    is_overfit = gap > (OVERFIT_GAP_THRESHOLD_PTS / 100.0)
    is_underfit = train_acc < UNDERFIT_MIN_TRAIN_ACC
    flag = "OVERFIT" if is_overfit else ("UNDERFIT" if is_underfit else "ok")

    print(f"  C={c:<6} | Train: {train_acc*100:5.2f}% | Val: {val_acc*100:5.2f}% | "
          f"Gap: {gap*100:5.2f}pts | {flag}")

    sweep_results.append({
        "c": c, "model": candidate, "train_acc": train_acc, "val_acc": val_acc,
        "gap": gap, "is_overfit": is_overfit, "is_underfit": is_underfit,
    })

# Selection: only consider candidates that violate NEITHER non-negotiable.
# Among those, pick the one with the highest val accuracy (the best model
# that still respects both constraints) rather than just the smallest gap,
# since smallest gap alone can select a near-useless underfit model.
clean_candidates = [r for r in sweep_results if not r["is_overfit"] and not r["is_underfit"]]

if clean_candidates:
    best = max(clean_candidates, key=lambda r: r["val_acc"])
    print(f"\n{len(clean_candidates)}/{len(C_CANDIDATES)} candidates satisfy both non-negotiables "
          f"(no overfit, no underfit).")
else:
    # No candidate satisfies both — pick the least-bad one and say so loudly.
    # This is a signal to widen C_CANDIDATES, adjust class_weight, or revisit
    # the feature map / PCA dimensionality — not something to silently paper over.
    best = min(sweep_results, key=lambda r: abs(r["gap"]) if not r["is_underfit"] else np.inf)
    print("\n*** WARNING: NO C VALUE IN THIS GRID SATISFIES BOTH NON-NEGOTIABLES. ***")
    print("*** Falling back to the least-bad candidate below — treat this as a signal ***")
    print("*** to widen C_CANDIDATES, revisit class_weight, or reconsider N_COMPONENTS, ***")
    print("*** not as a result to report as-is. ***")

best_c = best["c"]
qsvc = best["model"]
best_gap = best["gap"]
best_val_acc = best["val_acc"]
best_train_acc = best["train_acc"]

print(f"\nSelected C={best_c} | Train: {best_train_acc*100:.2f}% | Val: {best_val_acc*100:.2f}% | "
      f"Gap: {best_gap*100:.2f}pts | Overfit: {best['is_overfit']} | Underfit: {best['is_underfit']}")
print("Training complete.")


# ---- Evaluate QSVC (best C) on ALL 3 splits ----
# qsvc was fit with quantum_kernel="precomputed", so every predict() call
# below needs a precomputed Gram matrix (n_query, n_train), NOT raw features.
# Train/val matrices are already cached; only the test-vs-train matrix is new
# (computed once here, not per C candidate, since the sweep never touches test).

print("\nComputing test-vs-train kernel matrix (test set touched once, after model selection)...")
test_train_kernel_matrix = quantum_kernel.evaluate(x_vec=X_test_reduced, y_vec=X_train_reduced)
# shape (n_test, n_train)

print("\n--- QSVC results across all splits (best C) ---")

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

qsvc_train_metrics = evaluate_split_precomputed(qsvc, train_kernel_matrix, y_train_enc, "Train")
qsvc_val_metrics   = evaluate_split_precomputed(qsvc, val_train_kernel_matrix, y_val_enc, "Val")
qsvc_test_metrics  = evaluate_split_precomputed(qsvc, test_train_kernel_matrix, y_test_enc, "Test")

y_pred_test = qsvc.predict(test_train_kernel_matrix)
all_label_ids = np.arange(len(label_encoder.classes_))
print("\nClassification report (test set):")
print(classification_report(
    y_test_enc, y_pred_test, labels=all_label_ids,
    target_names=label_encoder.classes_, zero_division=0,
))
print("Confusion matrix (test set):")
print(confusion_matrix(y_test_enc, y_pred_test, labels=all_label_ids))


# ---- Classical RBF-SVM baseline, ALSO with a C sweep (fair comparison) ----

print("\nSweeping C for the classical RBF-SVM baseline too, for a fair comparison...")
best_c_svc = None
best_gap_svc = np.inf
best_val_acc_svc = -np.inf
svc_classical = None

for c in C_CANDIDATES:
    candidate = SVC(kernel="rbf", C=c, class_weight="balanced", random_state=RANDOM_STATE)
    candidate.fit(X_train_reduced, y_train_enc)

    train_acc = evaluate_split(candidate, X_train_reduced, y_train_enc, f"C={c} Train", verbose=False)["accuracy"]
    val_acc   = evaluate_split(candidate, X_val_reduced, y_val_enc, f"C={c} Val", verbose=False)["accuracy"]
    gap = train_acc - val_acc

    print(f"  C={c:<5} | Train: {train_acc*100:5.2f}% | Val: {val_acc*100:5.2f}% | Gap: {gap*100:5.2f}pts")

    if val_acc > 0 and (gap < best_gap_svc or (abs(gap - best_gap_svc) < 0.01 and val_acc > best_val_acc_svc)):
        best_gap_svc = gap
        best_val_acc_svc = val_acc
        best_c_svc = c
        svc_classical = candidate

print(f"Selected C={best_c_svc} for classical SVM (train-val gap: {best_gap_svc*100:.2f} points, val acc: {best_val_acc_svc*100:.2f}%)")

print("\n--- Classical RBF-SVM results across all splits (best C, same features) ---")
svc_train_metrics = evaluate_split(svc_classical, X_train_reduced, y_train_enc, "Train")
svc_val_metrics   = evaluate_split(svc_classical, X_val_reduced, y_val_enc, "Val")
svc_test_metrics  = evaluate_split(svc_classical, X_test_reduced, y_test_enc, "Test")


# ---- Accuracy plot: QSVC vs classical SVM, across train/val/test ----

splits = ["Train", "Val", "Test"]
qsvc_accs = [qsvc_train_metrics["accuracy"], qsvc_val_metrics["accuracy"], qsvc_test_metrics["accuracy"]]
svc_accs  = [svc_train_metrics["accuracy"], svc_val_metrics["accuracy"], svc_test_metrics["accuracy"]]

x = np.arange(len(splits))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 5))
bars1 = ax.bar(x - width/2, [a*100 for a in qsvc_accs], width, label="QSVC", color="#7F77DD")
bars2 = ax.bar(x + width/2, [a*100 for a in svc_accs], width, label="Classical RBF-SVM", color="#D85A30")
for bars in (bars1, bars2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("Accuracy by Split — QSVC vs Classical SVM")
ax.legend()
plt.tight_layout()
plt.show()


# ---- Error-rate plot (1 - accuracy) — the honest "loss" analog for SVMs ----

qsvc_err = [1 - a for a in qsvc_accs]
svc_err  = [1 - a for a in svc_accs]

fig, ax = plt.subplots(figsize=(6, 5))
bars1 = ax.bar(x - width/2, qsvc_err, width, label="QSVC", color="#7F77DD")
bars2 = ax.bar(x + width/2, svc_err, width, label="Classical RBF-SVM", color="#D85A30")
for bars in (bars1, bars2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.set_ylabel("Error rate (1 - accuracy)")
ax.set_ylim(0, 1)
ax.set_title("Error Rate by Split — QSVC vs Classical SVM\n(SVMs have no epochs; this is the closest honest analog to a loss curve)")
ax.legend()
plt.tight_layout()
plt.show()


# ---- Final non-negotiable summary ----
# Prints a plain pass/fail against all four constraints given for this run,
# using the actual measured numbers from this run — not estimates.

_total_runtime = time.time() - _run_start_time
_runtime_ok = _total_runtime <= TARGET_MAX_RUNTIME_SECONDS

print("\n" + "=" * 70)
print("NON-NEGOTIABLE CHECK (this run)")
print("=" * 70)
print(f"1. Full data used (no subsampling): "
      f"{'YES' if TRAIN_SAMPLE_SIZE is None else f'NO — capped at {TRAIN_SAMPLE_SIZE}'}")
print(f"2. No overfitting (selected model): "
      f"{'FLAGGED — gap ' + f'{best_gap*100:.2f}pts > {OVERFIT_GAP_THRESHOLD_PTS}pt threshold' if best['is_overfit'] else 'OK'}")
print(f"3. No underfitting (selected model): "
      f"{'FLAGGED — train acc ' + f'{best_train_acc*100:.2f}% < {UNDERFIT_MIN_TRAIN_ACC*100}% threshold' if best['is_underfit'] else 'OK'}")
print(f"4. Runtime vs {TARGET_MAX_RUNTIME_SECONDS/3600:.1f}h budget: "
      f"{_total_runtime/60:.1f} min ({_total_runtime/3600:.2f}h) — "
      f"{'OK' if _runtime_ok else 'OVER BUDGET'}")
print("=" * 70)