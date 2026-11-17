# ======================================================
# fine-tuned.py -- VQC on NCA-12 features (tuned version of vqc_nca12.py)
#
# WHAT YOUR LAST RUN SHOWED (why these changes):
#   * Training loss flattened at ~0.44. Most of that is the label-smoothing +
#     class-weight "floor", not unfinished learning. So the loss was not the
#     problem; the curve was already flat.
#   * Val accuracy bounced between ~95.2% and 96.1% from epoch 2 onward. With
#     1440 val rows, one sample is 0.07 pts and the standard error is ~0.5 pts,
#     so that bouncing is mostly noise plus a learning rate that is still too
#     big late in training.
#   * VQC and SVM share the same 12 NCA numbers, so both are capped by those
#     numbers. Tuning the circuit can tighten the VQC; it cannot add
#     information the 12 features do not contain.
#
# WHAT CHANGED (everything else is kept from vqc_nca12.py):
#   1. EMA of weights      : evaluate/select on a running average of the weights
#                            (smooths the val-accuracy zig-zag).
#   2. LR schedule         : 1 warm-up epoch, then cosine decay to ~1% of peak,
#                            stepped per batch (the old one stepped per epoch).
#   3. Separate LRs        : circuit weights and the linear head get their own LR.
#   4. Gradient clipping   : max-norm 1.0.
#   5. Input encoding      : angle = pi * sigmoid(z-score). No hard clipping of
#                            val/test values (old code clipped them to the
#                            train min/max range, which flattens outliers).
#   6. Feature noise       : small Gaussian noise on training inputs
#                            (regularisation; the train-val gap was ~3 pts).
#   7. Early stopping      : patience on the EMA validation score.
#   8. Checkpoint cache    : each seed is saved; rerunning reloads it instead of
#                            retraining (your 3-seed run took ~3 hours).
#   9. Fairer baseline     : the SVM now also gets a gamma sweep, not only C.
#  10. Honest statistics   : bootstrap 95% CI for test accuracy and an exact
#                            McNemar test of VQC vs SVM on the same test rows.
#  11. Convergence report  : per seed, prints whether the loss and val accuracy
#                            actually settled, using stated numeric criteria.
#
# STATUS: NOT RUN. I could not execute this on your data/machine. Set
#   QUICK_TEST = True first (checks the code path and prints sec/epoch).
#   Whether it beats the SVM is an empirical question; expect it to be close
#   to the SVM, not clearly above it.
#
# INSTALL:  pip install pennylane torch scipy
# ======================================================

import os
import math
import time
import json
import contextlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# ---- CONFIG ----

TRAIN_CSV = r"./features/efficientnetb0_features_train.csv"
VAL_CSV   = r"./features/efficientnetb0_features_val.csv"
TEST_CSV  = r"./features/efficientnetb0_features_test.csv"

N_COMPONENTS = 12
NCA_MAX_ITER = 200

# Circuit
N_LAYERS = 4
REUPLOAD = True
TRAINABLE_SCALE = True
INIT_STD = 0.5
READOUT = ("Z", "X", "Y", "ZZ")
HEAD_HIDDEN = 0                 # 0 = linear head (keeps the model quantum-heavy)
ENSEMBLE_SEEDS = [7, 123]

# Optimisation
EPOCHS = 30                     # upper bound; early stopping usually ends sooner
BATCH_SIZE = 64
LR_CIRCUIT = 0.02
LR_HEAD = 0.01
WEIGHT_DECAY = 1e-4             # applied to the head only
LABEL_SMOOTHING = 0.02          # was 0.05
WARMUP_EPOCHS = 1
MIN_LR_FRACTION = 0.01          # cosine decays to this fraction of peak LR
GRAD_CLIP = 1.0
FEATURE_NOISE = 0.05            # std of Gaussian noise added to z-scored inputs (train only)
EMA_DECAY = 0.995               # ~200-step memory (~2 epochs at batch 64)

# Early stopping / model selection (all on VALIDATION, never test)
PATIENCE = 10
MIN_EPOCHS = 10
TRAIN_EVAL_ROWS = 1440          # rows used for the clean per-epoch train accuracy

# Convergence criteria (used only for the printed report)
CONV_WINDOW = 5                 # look at the last 5 epochs
CONV_LOSS_REL_CHANGE = 0.01     # train loss moved < 1% across the window
CONV_VAL_STD_PTS = 0.30         # val accuracy std < 0.30 pts across the window

USE_CACHE = True                # reload finished seeds from disk (delete the .pt files to retrain)
CKPT_DIR = "./vqc_ckpt"

QUICK_TEST = False
TRAIN_SAMPLE_SIZE = None
VAL_SAMPLE_SIZE   = None
TEST_SAMPLE_SIZE  = None

# Classical baseline
C_CANDIDATES = [0.5, 1.0, 5.0, 10.0, 20.0]
GAMMA_MULTIPLIERS = [0.5, 1.0, 2.0]     # times the sklearn "scale" gamma
OVERFIT_GAP_THRESHOLD_PTS = 8.0
UNDERFIT_MIN_TRAIN_ACC = 0.45
RANDOM_STATE = 42
BOOTSTRAP_N = 2000

PRIOR_TEST_ACC = {"Classical RBF-SVM (NCA-12)": 94.52,
                  "QSVC fixed kernel (NCA-12)": 94.38,
                  "QSVC + QKA (NCA-12)": 94.52,
                  "Previous VQC ensemble (vqc_nca12.py)": 94.32}

if QUICK_TEST:
    EPOCHS = 2
    MIN_EPOCHS = 1
    TRAIN_SAMPLE_SIZE = 600
    VAL_SAMPLE_SIZE = 300
    TRAIN_EVAL_ROWS = 300
    USE_CACHE = False

os.makedirs(CKPT_DIR, exist_ok=True)
_run_start_time = time.time()

print("=" * 70)
print("fine-tuned.py -- VQC on NCA-12 features")
print(f"layers={N_LAYERS} reupload={REUPLOAD} epochs<={EPOCHS} batch={BATCH_SIZE} "
      f"lr_circuit={LR_CIRCUIT} lr_head={LR_HEAD} seeds={ENSEMBLE_SEEDS}")
if QUICK_TEST:
    print("*** QUICK_TEST is ON: results are NOT meaningful. ***")
print("=" * 70)


# ---- Load data ----

train_df = pd.read_csv(TRAIN_CSV).dropna(subset=["label"])
val_df   = pd.read_csv(VAL_CSV).dropna(subset=["label"])
test_df  = pd.read_csv(TEST_CSV).dropna(subset=["label"])
print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")

feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
X_train_full = train_df[feature_cols].values
X_val_full   = val_df[feature_cols].values
X_test_full  = test_df[feature_cols].values
print(f"Feature dimension in source CSVs: {len(feature_cols)}")

label_encoder = LabelEncoder()
label_encoder.fit(pd.concat([train_df["label"], val_df["label"], test_df["label"]]).unique())
y_train_enc = label_encoder.transform(train_df["label"].values)
y_val_enc   = label_encoder.transform(val_df["label"].values)
y_test_enc  = label_encoder.transform(test_df["label"].values)
n_classes = len(label_encoder.classes_)
print("Classes:", list(label_encoder.classes_))


# ---- Scale + NCA (fit on train only) ----

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_val_scaled   = scaler.transform(X_val_full)
X_test_scaled  = scaler.transform(X_test_full)

_t0 = time.time()
nca = NCA(n_components=N_COMPONENTS, max_iter=NCA_MAX_ITER, random_state=RANDOM_STATE, verbose=0)
X_train_reduced_full = nca.fit_transform(X_train_scaled, y_train_enc)
X_val_reduced_full   = nca.transform(X_val_scaled)
X_test_reduced_full  = nca.transform(X_test_scaled)
print(f"NCA applied: {N_COMPONENTS} components (fit on y_train only) | fit time: {time.time() - _t0:.1f}s")

n_qubits = X_train_reduced_full.shape[1]


def stratified_take(X, y, size, random_state):
    if size is not None and size < len(X):
        X_sub, _, y_sub, _ = train_test_split(X, y, train_size=size, stratify=y, random_state=random_state)
        return X_sub, y_sub
    return X, y

X_train_reduced, y_train_enc = stratified_take(X_train_reduced_full, y_train_enc, TRAIN_SAMPLE_SIZE, RANDOM_STATE)
X_val_reduced, y_val_enc     = stratified_take(X_val_reduced_full, y_val_enc, VAL_SAMPLE_SIZE, RANDOM_STATE)
X_test_reduced, y_test_enc   = stratified_take(X_test_reduced_full, y_test_enc, TEST_SAMPLE_SIZE, RANDOM_STATE)
print(f"Using {len(X_train_reduced)} train / {len(X_val_reduced)} val / {len(X_test_reduced)} test samples")
assert len(np.unique(y_train_enc)) > 1, "y_train_enc has only one class!"


def evaluate_preds(y_true, y_pred, split_name, verbose=True):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    if verbose:
        print(f"[{split_name}] Acc: {acc*100:.2f}% | Precision: {precision:.2f} | "
              f"Recall: {recall:.2f} | F1: {f1:.2f} | MCC: {mcc:.4f}")
    return {"split": split_name, "accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1, "mcc": mcc}


# ======================================================
# INPUT PREP: z-score with TRAIN statistics. The squash to an angle
# (pi * sigmoid) happens inside the model, so noise can be added in z-space.
# ======================================================

z_scaler = StandardScaler().fit(X_train_reduced)
Z_tr = torch.tensor(z_scaler.transform(X_train_reduced))
Z_va = torch.tensor(z_scaler.transform(X_val_reduced))
Z_te = torch.tensor(z_scaler.transform(X_test_reduced))
ytr_t = torch.tensor(y_train_enc, dtype=torch.long)
yva_t = torch.tensor(y_val_enc, dtype=torch.long)

_counts = np.bincount(y_train_enc, minlength=n_classes).astype(np.float64)
class_w = torch.tensor(len(y_train_enc) / (n_classes * np.maximum(_counts, 1)))

# fixed subset of train rows for a clean (no-noise, EMA-weights) train accuracy
_rng = np.random.RandomState(RANDOM_STATE)
_eval_idx = _rng.choice(len(Z_tr), size=min(TRAIN_EVAL_ROWS, len(Z_tr)), replace=False)


# ======================================================
# THE VQC
# ======================================================

def vqc_circuit(x, weights, scale):
    """x: (batch, n_qubits) angles. weights: (N_LAYERS, n_qubits, 3). scale: (n_qubits,)."""
    for layer in range(N_LAYERS):
        if layer == 0 or REUPLOAD:
            for q in range(n_qubits):
                qml.RY(x[:, q] * scale[q], wires=q)
        for q in range(n_qubits):
            qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])
    obs = []
    if "Z" in READOUT:  obs += [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
    if "X" in READOUT:  obs += [qml.expval(qml.PauliX(q)) for q in range(n_qubits)]
    if "Y" in READOUT:  obs += [qml.expval(qml.PauliY(q)) for q in range(n_qubits)]
    if "ZZ" in READOUT: obs += [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % n_qubits)) for q in range(n_qubits)]
    return obs


N_FEATS = n_qubits * sum(k in READOUT for k in ("Z", "X", "Y")) + (n_qubits if "ZZ" in READOUT else 0)


class VQCClassifier(nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weights = nn.Parameter(INIT_STD * torch.randn(N_LAYERS, n_qubits, 3, generator=g))
        scale = torch.ones(n_qubits)
        if TRAINABLE_SCALE:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)
        if HEAD_HIDDEN > 0:
            self.head = nn.Sequential(nn.Linear(N_FEATS, HEAD_HIDDEN), nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(HEAD_HIDDEN, n_classes))
        else:
            self.head = nn.Linear(N_FEATS, n_classes)
        dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(vqc_circuit, dev, interface="torch", diff_method="backprop")

    def forward(self, z):
        x = math.pi * torch.sigmoid(z)                      # z-score -> angle in (0, pi), no hard clip
        out = self.qnode(x, self.weights, self.scale)
        feats = torch.stack([o.reshape(z.shape[0]) for o in out], dim=1)
        return self.head(feats)


@torch.no_grad()
def predict_logits(model, Z, chunk=256):
    model.eval()
    return torch.cat([model(Z[i:i + chunk]) for i in range(0, len(Z), chunk)])


@contextlib.contextmanager
def use_weights(model, state):
    """Temporarily load `state` (name -> tensor) into the model's parameters."""
    backup = {k: p.detach().clone() for k, p in model.named_parameters()}
    with torch.no_grad():
        for k, p in model.named_parameters():
            p.copy_(state[k])
    try:
        yield
    finally:
        with torch.no_grad():
            for k, p in model.named_parameters():
                p.copy_(backup[k])


def convergence_report(hist):
    """Numeric check of whether training settled. Criteria are set in CONFIG."""
    if len(hist) < CONV_WINDOW + 1:
        return "too few epochs to judge"
    last = hist[-CONV_WINDOW:]
    loss_change = abs(last[-1]["train_loss"] - last[0]["train_loss"]) / max(last[0]["train_loss"], 1e-12)
    val_std = float(np.std([h["val_acc"] for h in last]) * 100)
    ok_loss = loss_change < CONV_LOSS_REL_CHANGE
    ok_val = val_std < CONV_VAL_STD_PTS
    verdict = "CONVERGED" if (ok_loss and ok_val) else "not fully settled"
    return (f"{verdict} | last {CONV_WINDOW} epochs: train-loss change {loss_change*100:.2f}% "
            f"(need <{CONV_LOSS_REL_CHANGE*100:.0f}%), val-acc std {val_std:.2f} pts (need <{CONV_VAL_STD_PTS:.2f})")


def _cfg_tag():
    return (f"L{N_LAYERS}_ru{int(REUPLOAD)}_ts{int(TRAINABLE_SCALE)}_h{HEAD_HIDDEN}_"
            f"{''.join(READOUT)}_nca{N_COMPONENTS}_n{len(Z_tr)}")


def train_one(seed):
    ckpt = os.path.join(CKPT_DIR, f"seed{seed}_{_cfg_tag()}.pt")
    torch.manual_seed(seed)
    model = VQCClassifier(seed)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if USE_CACHE and os.path.exists(ckpt):
        blob = torch.load(ckpt)
        with torch.no_grad():
            for k, p in model.named_parameters():
                p.copy_(blob["state"][k])
        print(f"\n[seed {seed}] loaded cached checkpoint {ckpt} (delete it to retrain)")
        return model, blob["hist"], n_p

    print(f"\n[seed {seed}] VQC: {N_LAYERS} layers x {n_qubits} qubits | {n_p} trainable params | {N_FEATS} readout features")
    circuit_params = [p for n, p in model.named_parameters() if not n.startswith("head")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("head")]
    opt = torch.optim.AdamW([
        {"params": circuit_params, "lr": LR_CIRCUIT, "weight_decay": 0.0},
        {"params": head_params,    "lr": LR_HEAD,    "weight_decay": WEIGHT_DECAY},
    ])

    steps_per_epoch = math.ceil(len(Z_tr) / BATCH_SIZE)
    total_steps = EPOCHS * steps_per_epoch
    warm_steps = max(1, WARMUP_EPOCHS * steps_per_epoch)

    def lr_lambda(step):
        if step < warm_steps:
            return (step + 1) / warm_steps
        prog = (step - warm_steps) / max(1, total_steps - warm_steps)
        return MIN_LR_FRACTION + (1 - MIN_LR_FRACTION) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(weight=class_w, label_smoothing=LABEL_SMOOTHING)
    val_loss_fn = nn.CrossEntropyLoss()

    ema = {k: p.detach().clone() for k, p in model.named_parameters()}
    gen = torch.Generator().manual_seed(seed)
    best_key, best_state, best_epoch, bad_epochs = (-1.0, 0.0), None, 0, 0
    hist, step = [], 0

    for epoch in range(EPOCHS):
        model.train(); t0 = time.time()
        perm = torch.randperm(len(Z_tr), generator=gen)
        run_loss, run_n = 0.0, 0
        for i in range(0, len(Z_tr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            zb = Z_tr[idx]
            if FEATURE_NOISE > 0:
                zb = zb + FEATURE_NOISE * torch.randn(zb.shape, generator=gen)
            opt.zero_grad()
            loss = loss_fn(model(zb), ytr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); sched.step(); step += 1
            d = min(EMA_DECAY, (1 + step) / (10 + step))
            with torch.no_grad():
                for k, p in model.named_parameters():
                    ema[k].mul_(d).add_(p.detach(), alpha=1 - d)
            run_loss += loss.item() * len(idx); run_n += len(idx)

        # evaluate the EMA weights (this is the model we keep)
        with use_weights(model, ema):
            vl = predict_logits(model, Z_va)
            tl = predict_logits(model, Z_tr[_eval_idx])
        val_acc = float((vl.argmax(1).numpy() == y_val_enc).mean())
        val_loss = val_loss_fn(vl, yva_t).item()
        train_acc = float((tl.argmax(1).numpy() == y_train_enc[_eval_idx]).mean())
        cur_lr = opt.param_groups[0]["lr"]
        hist.append({"epoch": epoch + 1, "train_loss": run_loss / run_n, "train_acc": train_acc,
                     "val_acc": val_acc, "val_loss": val_loss, "lr": cur_lr})

        improved = (val_acc, -val_loss) > best_key
        if improved:
            best_key = (val_acc, -val_loss)
            best_state = {k: v.clone() for k, v in ema.items()}
            best_epoch, bad_epochs = epoch + 1, 0
        else:
            bad_epochs += 1
        print(f"  epoch {epoch + 1:>3}/{EPOCHS} | loss {hist[-1]['train_loss']:.4f} | train {train_acc*100:5.2f}% "
              f"| val {val_acc*100:5.2f}% | val-loss {val_loss:.3f} | lr {cur_lr:.4f} | {time.time() - t0:.0f}s"
              f"{'  *best' if improved else ''}")
        if epoch + 1 >= MIN_EPOCHS and bad_epochs >= PATIENCE:
            print(f"  early stop: no improvement for {PATIENCE} epochs")
            break

    with torch.no_grad():
        for k, p in model.named_parameters():
            p.copy_(best_state[k])
    print(f"[seed {seed}] best val {best_key[0]*100:.2f}% at epoch {best_epoch}")
    print(f"[seed {seed}] convergence: {convergence_report(hist)}")
    if not QUICK_TEST:
        torch.save({"state": best_state, "hist": hist}, ckpt)
    return model, hist, n_p


print("\nTraining (selection on VALIDATION only; test is not used here)...")
_t_train = time.time()
models, histories = [], []
for _s in ENSEMBLE_SEEDS:
    m, h, n_params = train_one(_s)
    models.append(m); histories.append(h)
print(f"\nTraining stage finished in {(time.time() - _t_train) / 60:.1f} min")


def ens_pred(Z):
    probs = torch.stack([torch.softmax(predict_logits(m, Z), 1) for m in models]).mean(0)
    return probs.argmax(1).numpy()


# ---- Final evaluation ----

print(f"\n--- VQC results across all splits (ensemble of {len(models)}) ---")
if len(models) > 1:
    for _s, _m in zip(ENSEMBLE_SEEDS, models):
        print(f"  seed {_s}: val {accuracy_score(y_val_enc, predict_logits(_m, Z_va).argmax(1).numpy())*100:.2f}%")
vqc_train_metrics = evaluate_preds(y_train_enc, ens_pred(Z_tr), "Train")
vqc_val_metrics   = evaluate_preds(y_val_enc,   ens_pred(Z_va), "Val")
y_pred_test = ens_pred(Z_te)
vqc_test_metrics  = evaluate_preds(y_test_enc,  y_pred_test,    "Test")

_gap = vqc_train_metrics["accuracy"] - vqc_val_metrics["accuracy"]
_flag = ("OVERFIT" if _gap > OVERFIT_GAP_THRESHOLD_PTS / 100.0
         else ("UNDERFIT" if vqc_train_metrics["accuracy"] < UNDERFIT_MIN_TRAIN_ACC else "ok"))
print(f"Train-val gap: {_gap*100:.2f}pts | {_flag}")

all_label_ids = np.arange(n_classes)
print("\nClassification report (test set):")
print(classification_report(y_test_enc, y_pred_test, labels=all_label_ids,
                            target_names=label_encoder.classes_, zero_division=0))
print("Confusion matrix (test set):")
print(confusion_matrix(y_test_enc, y_pred_test, labels=all_label_ids))


# ---- Classical RBF-SVM baseline (C and gamma both swept on VALIDATION) ----

print("\nSweeping C and gamma for the classical RBF-SVM baseline (same NCA features)...")
_gamma0 = 1.0 / (X_train_reduced.shape[1] * X_train_reduced.var())
best_cfg, best_val_acc_svc, svc_classical = None, -np.inf, None
for c in C_CANDIDATES:
    for gm in GAMMA_MULTIPLIERS:
        cand = SVC(kernel="rbf", C=c, gamma=_gamma0 * gm, class_weight="balanced", random_state=RANDOM_STATE)
        cand.fit(X_train_reduced, y_train_enc)
        va = accuracy_score(y_val_enc, cand.predict(X_val_reduced))
        if va > best_val_acc_svc:
            best_val_acc_svc, best_cfg, svc_classical = va, (c, gm), cand
print(f"Selected C={best_cfg[0]}, gamma={best_cfg[1]}x'scale' for classical SVM (val acc: {best_val_acc_svc*100:.2f}%)")

print("\n--- Classical RBF-SVM results across all splits ---")
svc_train_metrics = evaluate_preds(y_train_enc, svc_classical.predict(X_train_reduced), "Train")
svc_val_metrics   = evaluate_preds(y_val_enc,   svc_classical.predict(X_val_reduced),   "Val")
y_pred_svc = svc_classical.predict(X_test_reduced)
svc_test_metrics  = evaluate_preds(y_test_enc,  y_pred_svc, "Test")


# ---- Is the VQC-vs-SVM difference real? (same test rows, paired) ----

print("\n--- Statistical check on the test set (VQC ensemble vs RBF-SVM) ---")
vqc_ok = (y_pred_test == y_test_enc)
svc_ok = (y_pred_svc == y_test_enc)
b = int((vqc_ok & ~svc_ok).sum())     # VQC right, SVM wrong
c_ = int((~vqc_ok & svc_ok).sum())    # SVM right, VQC wrong
p_mcn = binomtest(b, b + c_, 0.5).pvalue if (b + c_) > 0 else 1.0
print(f"Rows only VQC got right: {b} | rows only SVM got right: {c_} | exact McNemar p = {p_mcn:.3f}")

rng = np.random.RandomState(0)
n_te = len(y_test_enc)
boot_v, boot_s, boot_d = [], [], []
for _ in range(BOOTSTRAP_N):
    ii = rng.randint(0, n_te, n_te)
    av, as_ = vqc_ok[ii].mean(), svc_ok[ii].mean()
    boot_v.append(av); boot_s.append(as_); boot_d.append(av - as_)
ci = lambda a: (np.percentile(a, 2.5) * 100, np.percentile(a, 97.5) * 100)
print(f"VQC test acc {vqc_ok.mean()*100:.2f}% (95% CI {ci(boot_v)[0]:.2f}-{ci(boot_v)[1]:.2f})")
print(f"SVM test acc {svc_ok.mean()*100:.2f}% (95% CI {ci(boot_s)[0]:.2f}-{ci(boot_s)[1]:.2f})")
print(f"Difference VQC-SVM {(vqc_ok.mean()-svc_ok.mean())*100:+.2f} pts (95% CI {ci(boot_d)[0]:+.2f} to {ci(boot_d)[1]:+.2f})")
print("If that interval contains 0 (or McNemar p > 0.05), the data do not show a real difference.")


# ---- Plots ----

splits = ["Train", "Val", "Test"]
vqc_accs = [vqc_train_metrics["accuracy"], vqc_val_metrics["accuracy"], vqc_test_metrics["accuracy"]]
svc_accs = [svc_train_metrics["accuracy"], svc_val_metrics["accuracy"], svc_test_metrics["accuracy"]]
x = np.arange(len(splits)); width = 0.35

fig, ax = plt.subplots(figsize=(6, 5))
b1 = ax.bar(x - width/2, [a*100 for a in vqc_accs], width, label="VQC (NCA-12 features)", color="#7F77DD")
b2 = ax.bar(x + width/2, [a*100 for a in svc_accs], width, label="Classical RBF-SVM (same features)", color="#D85A30")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{bar.get_height():.1f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(splits); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(80, 101)
ax.set_title("Accuracy by Split - VQC vs Classical SVM")
ax.legend(loc="lower left"); plt.tight_layout(); plt.savefig("ft_accuracy_comparison.png", dpi=150); plt.show()

colors = ["#7F77DD", "#3FA796", "#D8A030", "#B04A8A", "#4A7FB0"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for j, (s, h) in enumerate(zip(ENSEMBLE_SEEDS, histories)):
    ep = [r["epoch"] for r in h]; col = colors[j % len(colors)]
    axes[0].plot(ep, [r["train_loss"] for r in h], marker="o", markersize=3, color=col, label=f"seed {s}")
    axes[1].plot(ep, [r["val_loss"] for r in h], marker="o", markersize=3, color=col, label=f"seed {s}")
    axes[2].plot(ep, [r["val_acc"] * 100 for r in h], marker="o", markersize=3, color=col, label=f"val seed {s}")
    axes[2].plot(ep, [r["train_acc"] * 100 for r in h], linestyle="--", color=col, alpha=0.5)
axes[0].set_title("Train loss (smoothed-label, weighted)"); axes[1].set_title("Validation loss (plain CE, EMA weights)")
axes[2].set_title("Accuracy (solid = val, dashed = train)")
for a in axes:
    a.set_xlabel("Epoch"); a.grid(alpha=0.3); a.legend()
axes[2].set_ylabel("Accuracy (%)")
plt.tight_layout(); plt.savefig("ft_training_curves.png", dpi=150); plt.show()


# ---- Save ----

with open("ft_results.json", "w") as f:
    json.dump({"seeds": ENSEMBLE_SEEDS, "layers": N_LAYERS, "params_per_model": n_params,
               "vqc": {"train": vqc_train_metrics, "val": vqc_val_metrics, "test": vqc_test_metrics},
               "svm": {"train": svc_train_metrics, "val": svc_val_metrics, "test": svc_test_metrics,
                       "C": best_cfg[0], "gamma_multiplier": best_cfg[1]},
               "mcnemar_p": p_mcn, "only_vqc_right": b, "only_svm_right": c_,
               "histories": histories}, f, indent=1, default=float)


# ---- Summary ----

_total = time.time() - _run_start_time
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"NCA-{N_COMPONENTS} -> {n_qubits}-qubit VQC | layers={N_LAYERS} | readout={READOUT} | "
      f"params/model={n_params} | seeds={ENSEMBLE_SEEDS}")
print(f"VQC ensemble test accuracy : {vqc_test_metrics['accuracy']*100:.2f}%")
print(f"RBF-SVM test accuracy      : {svc_test_metrics['accuracy']*100:.2f}%  (this run, C and gamma tuned)")
print("Earlier results (not recomputed):")
for k, v in PRIOR_TEST_ACC.items():
    print(f"  {k:<40}{v:.2f}%   (this VQC {vqc_test_metrics['accuracy']*100 - v:+.2f} pts)")
print(f"McNemar p = {p_mcn:.3f} (VQC vs SVM on the same test rows)")
print(f"Total runtime: {_total/60:.1f} min ({_total/3600:.2f}h)")
print("=" * 70)