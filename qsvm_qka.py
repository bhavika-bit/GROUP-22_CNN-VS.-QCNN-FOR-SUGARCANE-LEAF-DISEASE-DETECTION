# ======================================================
# QKA on NCA-12 (winning config) — NOT a retrain, built to run quick
#
# WHY NCA-12 AND NOT LDA-12:
# NCA-12 beat LDA-12 on both models' test accuracy (QSVC 94.38% vs 94.11%,
# classical SVM 94.52% vs 94.45%) and had the smaller QSVC-vs-classical gap
# (-0.14pts vs -0.34pts) -- the slightly stronger base to test alignment on.
#
# THREE THINGS THIS SCRIPT DELIBERATELY DOES NOT RETRAIN:
#   1. NCA itself -- the reduced features are cached to .npy after the first
#      fit (NCA_CACHE_DIR). Reruns load the cache and skip the ~72s NCA fit
#      entirely. Delete the cache dir if your source CSVs change.
#   2. The classical RBF-SVM baseline -- your qsvm_nca12.py run already
#      produced this (C=5.0, Test Acc 94.52%). Hardcoded below as
#      PRIOR_CLASSICAL_RESULTS, not refit.
#   3. The fixed (untrained) quantum kernel QSVC -- also already produced
#      (bw=0.08, C=1.0, Test Acc 94.38%). Hardcoded as PRIOR_FIXED_QSVC_RESULTS,
#      not recomputed. This also means NO bw x C sweep runs here -- bw=0.08
#      is fixed from that prior result, which is most of why this script is
#      fast: the 109-minute full sweep becomes a single bandwidth.
#
# WHAT ACTUALLY RUNS (the only "new training" in this script):
#   - A trainable per-feature Ry(theta) scaling layer in front of the same
#     ZZFeatureMap(bw=0.08, reps=1, entanglement="full"), optimized via
#     COBYLA to minimize SVC-loss (maximize kernel-target alignment).
#   - Training happens on a SMALL stratified subsample (QKA_TRAIN_SUBSAMPLE,
#     default 250) for a SMALL number of iterations (QKA_OPTIMIZER_MAXITER,
#     default 25) -- this is the part that would be prohibitively slow at
#     full scale (6750 rows), so it's deliberately kept cheap.
#   - The learned theta is then used ONCE to compute the final full-size
#     train/val/test kernel matrices (unavoidable at full size for a valid
#     test-set number -- this is the single biggest remaining cost, roughly
#     equivalent to ONE bandwidth from the original sweep, ~30-40 min).
#
# EXPECTED RUNTIME: ~35-50 min total (vs 109 min for the original full
# NCA-12 sweep) -- dominated by the single final kernel-matrix computation,
# not by QKA training itself (which finishes in a few minutes on the
# subsample). Set EVAL_SUBSAMPLE_SIZE below if you want an even faster
# sanity-check run before committing to the full-size final evaluation.
# ======================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, classification_report, confusion_matrix
)

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
try:
    from qiskit_machine_learning.kernels import TrainableFidelityStatevectorKernel as _TrainableKernelCls
    _USING_STATEVECTOR_KERNEL = True
except ImportError:
    from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel as _TrainableKernelCls
    _USING_STATEVECTOR_KERNEL = False
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.utils.loss_functions import SVCLoss
from qiskit_machine_learning.algorithms import QSVC


# ---- CONFIG ----
TRAIN_CSV = r"C:\Users\bhavi\OneDrive\Documents\Group22_Sugarcane_CNN_Project\efficientnetb0_features_train.csv"
VAL_CSV   = r"C:\Users\bhavi\OneDrive\Documents\Group22_Sugarcane_CNN_Project\efficientnetb0_features_val.csv"
TEST_CSV  = r"C:\Users\bhavi\OneDrive\Documents\Group22_Sugarcane_CNN_Project\efficientnetb0_features_test.csv"

N_COMPONENTS = 12                 # matched to LDA ceiling / NCA's own winning width
BEST_BW = 0.08                    # fixed from the already-completed NCA-12 sweep -- NOT re-swept
ZZ_REPS = 1
ZZ_ENTANGLEMENT = "full"
C_CANDIDATES = [0.5, 1.0, 5.0, 10.0, 20.0]   # cheap quick sweep, kernel is only computed once
RANDOM_STATE = 42

QKA_TRAIN_SUBSAMPLE = 250          # small on purpose -- alignment training set, not the final eval set
QKA_OPTIMIZER_MAXITER = 25         # small on purpose -- keeps the COBYLA loop fast

# Set to an int (e.g. 1500) for a faster sanity-check run; None = full dataset
# (what you want for a number that's actually comparable to the prior results).
EVAL_SUBSAMPLE_SIZE = None

NCA_CACHE_DIR = "./nca12_cache"

# ---- Results already obtained from qsvm_nca12.py -- reused, not retrained ----
PRIOR_CLASSICAL_RESULTS = {"train": 0.9957, "val": 0.9611, "test": 0.9452}
PRIOR_FIXED_QSVC_RESULTS = {"train": 0.9957, "val": 0.9639, "test": 0.9438}

_run_start = time.time()
print("=" * 70)
print("QKA on NCA-12 -- fixed bandwidth, cached NCA, reused prior baselines")
print(f"Trainable kernel backend: "
      f"{'TrainableFidelityStatevectorKernel (fast)' if _USING_STATEVECTOR_KERNEL else 'TrainableFidelityQuantumKernel (Sampler-based, slower)'}")
print("=" * 70)


# ======================================================
# STEP 1 — Load data, get NCA-12 features (cached -- NOT refit if cache exists)
# ======================================================
os.makedirs(NCA_CACHE_DIR, exist_ok=True)
_cache_paths = {k: os.path.join(NCA_CACHE_DIR, f"{k}.npy")
                for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "classes"]}
_cache_complete = all(os.path.exists(p) for p in _cache_paths.values())

if _cache_complete:
    print(f"\nLoading cached NCA-12 features from {NCA_CACHE_DIR}/ -- NCA is NOT being refit.")
    X_train = np.load(_cache_paths["X_train"])
    y_train = np.load(_cache_paths["y_train"])
    X_val   = np.load(_cache_paths["X_val"])
    y_val   = np.load(_cache_paths["y_val"])
    X_test  = np.load(_cache_paths["X_test"])
    y_test  = np.load(_cache_paths["y_test"])
    classes = np.load(_cache_paths["classes"], allow_pickle=True)
else:
    print(f"\nNo cache found in {NCA_CACHE_DIR}/ -- fitting NCA once (one-time cost, ~70s).")
    train_df = pd.read_csv(TRAIN_CSV).dropna(subset=["label"])
    val_df   = pd.read_csv(VAL_CSV).dropna(subset=["label"])
    test_df  = pd.read_csv(TEST_CSV).dropna(subset=["label"])

    feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
    X_train_full = train_df[feature_cols].values
    y_train_full = train_df["label"].values
    X_val_full   = val_df[feature_cols].values
    y_val_full   = val_df["label"].values
    X_test_full  = test_df[feature_cols].values
    y_test_full  = test_df["label"].values

    label_encoder = LabelEncoder()
    all_labels = pd.concat([train_df["label"], val_df["label"], test_df["label"]]).unique()
    label_encoder.fit(all_labels)
    classes = label_encoder.classes_
    y_train = label_encoder.transform(y_train_full)
    y_val   = label_encoder.transform(y_val_full)
    y_test  = label_encoder.transform(y_test_full)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_val_scaled   = scaler.transform(X_val_full)
    X_test_scaled  = scaler.transform(X_test_full)

    _t0 = time.time()
    nca = NCA(n_components=N_COMPONENTS, max_iter=50, random_state=RANDOM_STATE)
    X_train = nca.fit_transform(X_train_scaled, y_train)
    X_val   = nca.transform(X_val_scaled)
    X_test  = nca.transform(X_test_scaled)
    print(f"NCA fit complete in {time.time() - _t0:.1f}s")

    np.save(_cache_paths["X_train"], X_train); np.save(_cache_paths["y_train"], y_train)
    np.save(_cache_paths["X_val"], X_val);     np.save(_cache_paths["y_val"], y_val)
    np.save(_cache_paths["X_test"], X_test);   np.save(_cache_paths["y_test"], y_test)
    np.save(_cache_paths["classes"], classes)
    print(f"Cached reduced features to {NCA_CACHE_DIR}/ for future runs.")

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} | dims: {X_train.shape[1]}")

# Optional eval subsample (stratified) -- only affects the FINAL kernel
# matrices / final metrics, not the QKA alignment training step below.
def stratified_take(X, y, size, seed):
    if size is not None and size < len(X):
        X_sub, _, y_sub, _ = train_test_split(X, y, train_size=size, stratify=y, random_state=seed)
        return X_sub, y_sub
    return X, y

X_train_eval, y_train_eval = stratified_take(X_train, y_train, EVAL_SUBSAMPLE_SIZE, RANDOM_STATE)
X_val_eval, y_val_eval     = stratified_take(X_val, y_val, EVAL_SUBSAMPLE_SIZE, RANDOM_STATE)
X_test_eval, y_test_eval   = stratified_take(X_test, y_test, EVAL_SUBSAMPLE_SIZE, RANDOM_STATE)
if EVAL_SUBSAMPLE_SIZE is not None:
    print(f"*** EVAL_SUBSAMPLE_SIZE={EVAL_SUBSAMPLE_SIZE} set -- final metrics are NOT directly "
          f"comparable to the full-dataset prior results. Set to None for the real comparison. ***")


# ======================================================
# STEP 2 — Fixed angle encoding at bw=0.08 (no sweep)
# ======================================================
angle_scaler = MinMaxScaler(feature_range=(0, np.pi * BEST_BW))
Xtr_ang = angle_scaler.fit_transform(X_train_eval)
Xval_ang = angle_scaler.transform(X_val_eval)
Xtest_ang = angle_scaler.transform(X_test_eval)

n_qubits = X_train.shape[1]
base_feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=ZZ_REPS,
                                 entanglement=ZZ_ENTANGLEMENT).decompose()


# ======================================================
# STEP 3 — QKA training (the only genuinely new training in this script)
# ======================================================
print("\n" + "=" * 70)
print("QKA: training trainable Ry(theta) scaling layer")
print("=" * 70)

training_params = ParameterVector("theta", n_qubits)
scaling_circuit = QuantumCircuit(n_qubits)
for i in range(n_qubits):
    scaling_circuit.ry(training_params[i], i)
qka_feature_map = scaling_circuit.compose(base_feature_map)

trainable_kernel = _TrainableKernelCls(feature_map=qka_feature_map, training_parameters=training_params)

rng = np.random.RandomState(RANDOM_STATE)
idx_by_class = {c: np.where(y_train_eval == c)[0] for c in np.unique(y_train_eval)}
per_class_n = max(1, QKA_TRAIN_SUBSAMPLE // len(idx_by_class))
subsample_idx = np.concatenate([
    rng.choice(idxs, size=min(per_class_n, len(idxs)), replace=False)
    for idxs in idx_by_class.values()
])
rng.shuffle(subsample_idx)
X_qka = Xtr_ang[subsample_idx]
y_qka = y_train_eval[subsample_idx]
print(f"Alignment training subsample: {len(subsample_idx)} rows (~{per_class_n}/class) -- "
      f"NOT the full 6750-row training set.")

kta_history = []

class LoggingSVCLoss(SVCLoss):
    """SVCLoss that records every evaluation so we can plot the curve."""
    def evaluate(self, parameter_values, quantum_kernel, data, labels):
        value = super().evaluate(parameter_values, quantum_kernel, data, labels)
        kta_history.append(value)
        n = len(kta_history)
        if n == 1 or n % 5 == 0:
            print(f"  eval {n:<3} | SVC-loss: {value:.6f}")
        return value

_t0 = time.time()
qk_trainer = QuantumKernelTrainer(
    quantum_kernel=trainable_kernel,
    loss=LoggingSVCLoss(C=1.0),
    optimizer=COBYLA(maxiter=QKA_OPTIMIZER_MAXITER),   # no callback
    initial_point=np.zeros(n_qubits),
)
qka_result = qk_trainer.fit(X_qka, y_qka)
aligned_kernel = qka_result.quantum_kernel
theta = qka_result.optimal_point
print(f"QKA training done in {time.time()-_t0:.1f}s | "
      f"loss {kta_history[0]:.4f} -> {qka_result.optimal_value:.4f} "
      f"({'improved' if qka_result.optimal_value < kta_history[0] else 'did not improve'})")
print(f"theta: {np.round(theta, 4)}")


# ======================================================
# STEP 4 — ONE final kernel-matrix computation (the unavoidable cost)
# ======================================================
print("\nComputing final train/val/test kernel matrices with the aligned feature map "
      "(single pass, no sweep)...")
_t0 = time.time()
K_train = aligned_kernel.evaluate(Xtr_ang)
K_val   = aligned_kernel.evaluate(Xval_ang, Xtr_ang)
K_test  = aligned_kernel.evaluate(Xtest_ang, Xtr_ang)
print(f"Done in {(time.time()-_t0)/60:.1f} min")

print("\nQuick C-sweep on the (already-computed) aligned kernel...")
best = None
for c in C_CANDIDATES:
    clf = SVC(kernel="precomputed", C=c, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(K_train, y_train_eval)
    val_acc = accuracy_score(y_val_eval, clf.predict(K_val))
    if best is None or val_acc > best["val_acc"]:
        best = {"c": c, "val_acc": val_acc, "clf": clf}
clf_aligned = best["clf"]
print(f"Selected C={best['c']} (val acc {best['val_acc']*100:.2f}%)")

def summarize(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"[{name}] Acc: {acc*100:.2f}% | F1: {f1:.2f} | MCC: {mcc:.4f}")
    return acc

aligned_acc = {
    "train": summarize(y_train_eval, clf_aligned.predict(K_train), "Train"),
    "val":   summarize(y_val_eval,   clf_aligned.predict(K_val),   "Val"),
    "test":  summarize(y_test_eval,  clf_aligned.predict(K_test),  "Test"),
}


# ======================================================
# SUMMARY (prior results reused as-is, not recomputed)
# ======================================================
print("\n" + "=" * 70)
print("SUMMARY -- Test accuracy")
print("=" * 70)
print(f"{'Classical RBF-SVM (prior, not retrained)':<42} {PRIOR_CLASSICAL_RESULTS['test']*100:.2f}%")
print(f"{'QSVC fixed kernel (prior, not retrained)':<42} {PRIOR_FIXED_QSVC_RESULTS['test']*100:.2f}%")
print(f"{'QSVC QKA-aligned (this run)':<42} {aligned_acc['test']*100:.2f}%")
print(f"\nQKA vs fixed kernel: {(aligned_acc['test']-PRIOR_FIXED_QSVC_RESULTS['test'])*100:+.2f}pts")
print(f"QKA vs classical SVM: {(aligned_acc['test']-PRIOR_CLASSICAL_RESULTS['test'])*100:+.2f}pts")


# ======================================================
# PLOTS
# ======================================================
print("\nGenerating plots...")
models = ["Classical\nRBF-SVM\n(prior)", "QSVC\nfixed kernel\n(prior)", "QSVC\nQKA-aligned\n(this run)"]
train_vals = [PRIOR_CLASSICAL_RESULTS["train"], PRIOR_FIXED_QSVC_RESULTS["train"], aligned_acc["train"]]
val_vals   = [PRIOR_CLASSICAL_RESULTS["val"],   PRIOR_FIXED_QSVC_RESULTS["val"],   aligned_acc["val"]]
test_vals  = [PRIOR_CLASSICAL_RESULTS["test"],  PRIOR_FIXED_QSVC_RESULTS["test"],  aligned_acc["test"]]

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(models)); width = 0.25
for i, (label, vals) in enumerate([("Train", train_vals), ("Val", val_vals), ("Test", test_vals)]):
    bars = ax.bar(x + (i - 1) * width, [v*100 for v in vals], width, label=label)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("Accuracy: Classical vs Fixed-Kernel QSVC vs QKA-Aligned QSVC (NCA-12)")
ax.legend(title="Split"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("qka_nca12_accuracy_comparison.png", dpi=150)
print("Saved qka_nca12_accuracy_comparison.png")

fig, ax = plt.subplots(figsize=(7, 5))
errors = [(1 - v) * 100 for v in test_vals]
bars = ax.bar(models, errors, color=["#969696", "#fc9272", "#de2d26"])
ax.bar_label(bars, fmt="%.2f%%", padding=3)
ax.set_ylabel("Test Error Rate (%)"); ax.set_title("Test Error Rate — lower is better")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("qka_nca12_error_rate_comparison.png", dpi=150)
print("Saved qka_nca12_error_rate_comparison.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(kta_history) + 1), kta_history, marker="o", markersize=3, color="#08519c")
ax.set_xlabel("COBYLA iteration"); ax.set_ylabel("SVC loss (lower = better alignment)")
ax.set_title(f"QKA Training Curve (subsample={len(subsample_idx)}, {QKA_OPTIMIZER_MAXITER} iters)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("qka_nca12_training_curve.png", dpi=150)
print("Saved qka_nca12_training_curve.png")

plt.show()

print("\n" + "=" * 70)
print(f"Total runtime: {(time.time()-_run_start)/60:.1f} min")
print("=" * 70)