"""
Milk Adulteration Detection — ESP32 OPTIMIZED PIPELINE
=======================================================
Outputs:
  - model.h                  (C header optimized for ESP32 SRAM/Flash)
  - plot_confusion.png       (Confusion matrix)
  - plot_feature_importance.png
  - plot_roc.png             (ROC curves per class)
  - plot_noise_robustness.png
  - plot_cv_scores.png
  - plot_class_metrics.png   (per-class precision/recall/F1)
  - training_report.json
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, f1_score, precision_score, recall_score)
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
import warnings, json
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH       = "amul_realistic_v2.csv"
AUG_COPIES      = 3
AUG_NOISE_STD   = 0.002
AUG_PH_NOISE    = 0.015
ROBUSTNESS_LEVELS = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012]
CLASS_WEIGHTS   = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.4}
CLASS_NAMES     = {0: "Pure Milk", 1: "Water Added", 2: "Detergent", 3: "Urea", 4: "Starch"}

# Plot theme
BG    = '#0f1117'
PANEL = '#1a1d27'
TEXT  = '#e0e0e0'
GRID  = '#2a2d3a'
COLORS = ['#00d4ff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8']

def style_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT, pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)

# ─────────────────────────────────────────────
# 1. LOAD & FEATURE ENGINEER
# ─────────────────────────────────────────────
print("=" * 60)
print("  MILK ADULTERATION — ESP32 OPTIMIZED TRAINING PIPELINE")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Loaded: {df.shape[0]} samples, {df.shape[1]} columns")

ch_cols = [f'ch{i}' for i in range(14)]

# Normalise channels by intensity
for ch in ch_cols:
    df[f'{ch}_norm'] = df[ch] / (df['total_intensity'] + 1e-8)

ch_norm_cols   = [f'{ch}_norm' for ch in ch_cols]
ch_norm_matrix = df[ch_norm_cols].values
ch_matrix      = df[ch_cols].values

# Log intensity (clipped)
df['log_intensity'] = np.clip(np.log(df['total_intensity'] + 1e-8), 6, 9)

# Spectral shape features
df['spectral_width']    = ch_matrix.std(axis=1)
df['peak_to_mean']      = ch_matrix.max(axis=1) / (ch_matrix.mean(axis=1) + 1e-8)
df['spectral_contrast'] = (ch_matrix.max(axis=1) - ch_matrix.min(axis=1)) / (ch_matrix.max(axis=1) + 1e-8)
df['spectral_entropy']  = -np.sum(ch_norm_matrix * np.log(ch_norm_matrix + 1e-8), axis=1)
df['blue_nir_ratio']    = (df['ch0'] + df['ch1']) / (df['ch10'] + df['ch11'] + 1e-8)
df['vis_nir_ratio']     = df[['ch0','ch1','ch2','ch3','ch4','ch5']].mean(axis=1) / \
                           df[['ch8','ch9','ch10','ch11']].mean(axis=1)

# Intensity relative features
df['intensity_per_width'] = df['total_intensity'] / (df['spectral_width'] + 1e-8)
df['intensity_per_peak']  = df['total_intensity'] / (df['peak_to_mean']   + 1e-8)

# pH features
df['pH']             = np.clip(df['pH'], 6.0, 8.0)
df['pH_dev']         = np.abs(df['pH'] - 6.68)
df['pH_x_intensity'] = df['pH'] * df['log_intensity']
df['pH_x_entropy']   = df['pH'] * df['spectral_entropy']

# Drop raw channels and temp
df = df.drop(columns=ch_cols + ['temp'], errors='ignore')

X = df.drop('label', axis=1).values.astype(np.float32)
y = df['label'].values
feature_names = df.drop('label', axis=1).columns.tolist()

print(f"    Features after engineering: {len(feature_names)}")
print(f"    Class distribution: { {k: int((y==k).sum()) for k in range(5)} }")

# ─────────────────────────────────────────────
# 2. SPLIT
# ─────────────────────────────────────────────
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\n[2] Train: {len(X_train_raw)}  |  Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. AUGMENTATION (training only)
# ─────────────────────────────────────────────
channel_cols_idx = [i for i, f in enumerate(feature_names) if 'norm' in f]
ph_col_idx       = feature_names.index('pH')
ph_dev_col_idx   = feature_names.index('pH_dev')
int_col_idx      = feature_names.index('log_intensity')

def augment(X, y, copies):
    aug_X, aug_y = [X], [y]
    for _ in range(copies):
        Xn = X.copy()
        Xn[:, channel_cols_idx] += np.random.normal(0, AUG_NOISE_STD,
                                             (len(X), len(channel_cols_idx)))
        ph_noise = np.random.normal(0, AUG_PH_NOISE, len(X))
        Xn[:, ph_col_idx]     += ph_noise
        Xn[:, ph_dev_col_idx]  = np.abs(Xn[:, ph_col_idx] - 6.68)
        Xn[:, int_col_idx]    += np.random.normal(0, 0.03, len(X))
        Xn[:, channel_cols_idx] = np.clip(Xn[:, channel_cols_idx], 0, 1)
        Xn[:, ph_col_idx]       = np.clip(Xn[:, ph_col_idx], 6.0, 8.0)
        aug_X.append(Xn)
        aug_y.append(y)
    return np.vstack(aug_X), np.concatenate(aug_y)

X_train, y_train = augment(X_train_raw, y_train_raw, AUG_COPIES)
X_train, y_train = shuffle(X_train, y_train, random_state=42)
print(f"\n[3] Augmented training set: {len(X_train_raw)} → {len(X_train)} samples")

# ─────────────────────────────────────────────
# 4. TRAIN (OPTIMIZED FOR ESP32)
# ─────────────────────────────────────────────
print("\n[4] Training Optimized Random Forest...")
clf = RandomForestClassifier(
    n_estimators=45,          # Reduced from 220 to strictly control size
    max_depth=9,              # Capped depth to prevent exponential node growth
    min_samples_leaf=12,      # Forces simpler, more generalized trees
    max_features='sqrt',
    class_weight=CLASS_WEIGHTS,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
print("    Done.")

# ─────────────────────────────────────────────
# 5. CORE METRICS
# ─────────────────────────────────────────────
train_acc  = clf.score(X_train, y_train)
test_acc   = clf.score(X_test, y_test)
gap        = train_acc - test_acc
y_pred     = clf.predict(X_test)
y_prob     = clf.predict_proba(X_test)
cm         = confusion_matrix(y_test, y_pred)

cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores  = cross_val_score(clf, X_train_raw, y_train_raw, cv=cv)

# Per-class metrics
f1s  = f1_score(y_test, y_pred, average=None)
prec = precision_score(y_test, y_pred, average=None)
rec  = recall_score(y_test, y_pred, average=None)

# Noise robustness
robust_accs = []
for noise in ROBUSTNESS_LEVELS:
    Xn = X_test.copy()
    if noise > 0:
        Xn[:, channel_cols_idx] += np.random.normal(0, noise, (len(Xn), len(channel_cols_idx)))
    robust_accs.append(clf.score(Xn, y_test))

# Feature importance
importances   = clf.feature_importances_
feat_sorted   = sorted(zip(feature_names, importances), key=lambda x: -x[1])

# Ablation test
print("\n[5] Feature Ablation Test:")
baseline = clf.score(X_test, y_test)
ablation = {}
top_feats = [f for f,_ in feat_sorted[:8]]
for feat in top_feats:
    idx = feature_names.index(feat)
    Xa  = X_test.copy()
    Xa[:, idx] = X_test[:, idx].mean()
    acc = clf.score(Xa, y_test)
    drop = baseline - acc
    ablation[feat] = drop
    flag = "⚠️ " if drop > 0.05 else "✅"
    print(f"  {flag} Without {feat:25s}: {acc:.4f}  (drop: {drop:+.4f})")

print(f"\n{'='*60}")
print(f"  Train Accuracy : {train_acc:.4f}")
print(f"  Test  Accuracy : {test_acc:.4f}")
print(f"  Gap            : {gap:.4f}  {'✅ Healthy' if gap < 0.05 else '⚠️ Overfit'}")
print(f"  CV Mean        : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Noise Drop     : {robust_accs[0]:.4f} → {robust_accs[-1]:.4f}  (Δ={robust_accs[0]-robust_accs[-1]:.4f})")
print(f"{'='*60}")
print(f"\n{classification_report(y_test, y_pred, target_names=[CLASS_NAMES[i] for i in range(5)])}")
print("Confusion Matrix:")
print(cm)

# ─────────────────────────────────────────────
# 6. PLOTS (Kept identical to original)
# ─────────────────────────────────────────────
print("\n[6] Generating plots...")

# ── Plot 1: Confusion Matrix ──────────────────
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar(im, ax=ax).ax.tick_params(colors=TEXT)
labels = [CLASS_NAMES[i] for i in range(5)]
ax.set_xticks(range(5)); ax.set_xticklabels(labels, rotation=25, ha='right', color=TEXT, fontsize=9)
ax.set_yticks(range(5)); ax.set_yticklabels(labels, color=TEXT, fontsize=9)
for i in range(5):
    for j in range(5):
        color = 'white' if cm[i,j] > cm.max()*0.5 else TEXT
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)
ax.set_xlabel('Predicted Label', color=TEXT, fontsize=10)
ax.set_ylabel('True Label', color=TEXT, fontsize=10)
ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold', color=TEXT, pad=10)
for spine in ax.spines.values(): spine.set_edgecolor(GRID)
plt.tight_layout()
plt.savefig('plot_confusion.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("    Saved: plot_confusion.png")

# ── Plot 2: Feature Importance ────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

fnames_plot = [f for f,_ in feat_sorted]
fimps_plot  = [v for _,v in feat_sorted]
bar_colors  = [COLORS[i % len(COLORS)] for i in range(len(fnames_plot))]
bars = ax.barh(fnames_plot[::-1], fimps_plot[::-1],
               color=bar_colors[::-1], edgecolor='none', height=0.65)
for bar, val in zip(bars, fimps_plot[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', color=TEXT, fontsize=8)
ax.axvline(x=0.15, color='#ff6b6b', linestyle='--', linewidth=1.2, alpha=0.7, label='Dependence threshold (0.15)')
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
style_ax(ax, 'Feature Importance')
ax.set_xlabel('Importance Score')
ax.tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig('plot_feature_importance.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("    Saved: plot_feature_importance.png")

# ── Plot 3: ROC Curves ────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

y_test_bin = label_binarize(y_test, classes=[0,1,2,3,4])
for i in range(5):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=COLORS[i], lw=2,
            label=f'{CLASS_NAMES[i]}  (AUC = {roc_auc:.3f})')

ax.plot([0,1],[0,1], '--', color=GRID, lw=1.2)
ax.fill_between([0,1],[0,1], alpha=0.04, color='white')
style_ax(ax, 'ROC Curves — One vs Rest')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID, loc='lower right')
plt.tight_layout()
plt.savefig('plot_roc.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("    Saved: plot_roc.png")

# ── Plot 4: Noise Robustness ──────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

ax.plot(ROBUSTNESS_LEVELS, robust_accs, 'o-', color='#00d4ff',
        lw=2.5, markersize=7, markerfacecolor='#ff6b6b', markeredgecolor='white', markeredgewidth=0.5)
ax.fill_between(ROBUSTNESS_LEVELS, robust_accs, min(robust_accs)-0.01,
                alpha=0.12, color='#00d4ff')
ax.axhline(y=0.90, color='#ffd43b', linestyle='--', lw=1.2, alpha=0.7, label='90% threshold')
for noise, acc in zip(ROBUSTNESS_LEVELS, robust_accs):
    ax.annotate(f'{acc:.3f}', (noise, acc),
                textcoords="offset points", xytext=(0, 9),
                ha='center', color=TEXT, fontsize=8)
style_ax(ax, 'Noise Robustness Sweep')
ax.set_xlabel('Added Channel Noise (σ)')
ax.set_ylabel('Test Accuracy')
ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
plt.tight_layout()
plt.savefig('plot_noise_robustness.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("    Saved: plot_noise_robustness.png")

# ── Plot 5: Cross-Validation Scores ──────────
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

cv_x = np.arange(1, 6)
bars = ax.bar(cv_x, cv_scores, color=COLORS[0], edgecolor='none', width=0.55, alpha=0.85)
ax.axhline(cv_scores.mean(), color='#ffd43b', lw=1.8, linestyle='--',
           label=f'Mean: {cv_scores.mean():.4f}')
ax.fill_between([0.5, 5.5],
                cv_scores.mean() - cv_scores.std(),
                cv_scores.mean() + cv_scores.std(),
                alpha=0.15, color='#ffd43b', label=f'±Std: {cv_scores.std():.4f}')
for x, s in zip(cv_x, cv_scores):
    ax.text(x, s + 0.001, f'{s:.4f}', ha='center', color=TEXT, fontsize=9)
ax.set_xticks(cv_x)
ax.set_xticklabels([f'Fold {i}' for i in cv_x], fontsize=9)
ax.set_ylim(max(0.85, cv_scores.min()-0.02), 1.01)
style_ax(ax, '5-Fold Cross-Validation')
ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
plt.tight_layout()
plt.savefig('plot_cv_scores.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("    Saved: plot_cv_scores.png")

# ── Plot 6: Per-Class Precision / Recall / F1 ─
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

x         = np.arange(5)
width     = 0.25
labels_short = [CLASS_NAMES[i] for i in range(5)]
b1 = ax.bar(x - width, prec, width, label='Precision', color=COLORS[0], edgecolor='none', alpha=0.9)
b2 = ax.bar(x,         rec,  width, label='Recall',    color=COLORS[1], edgecolor='none', alpha=0.9)
b3 = ax.bar(x + width, f1s,  width, label='F1-Score',  color=COLORS[2], edgecolor='none', alpha=0.9)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.2f}', ha='center', va='bottom', color=TEXT, fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(labels_short, rotation=15, fontsize=9)
ax.set_ylim(0, 1.12)
ax.axhline(y=0.9, color='#ffd43b', linestyle='--', lw=1, alpha=0.5)
style_ax(ax, 'Per-Class Precision / Recall / F1')
ax.set_ylabel('Score')
ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
plt.tight_layout()
plt.savefig('plot_class_metrics.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("    Saved: plot_class_metrics.png")

# ── Plot 7: Ablation Test ─────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

abl_names = list(ablation.keys())
abl_drops = list(ablation.values())
abl_colors = ['#ff6b6b' if d > 0.05 else '#51cf66' for d in abl_drops]
bars = ax.bar(abl_names, abl_drops, color=abl_colors, edgecolor='none', width=0.6)
ax.axhline(y=0.05, color='#ff6b6b', linestyle='--', lw=1.2, alpha=0.7, label='Danger threshold (0.05)')
for bar, val in zip(bars, abl_drops):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', color=TEXT, fontsize=8.5)
ax.set_xticklabels(abl_names, rotation=20, ha='right', fontsize=8)
style_ax(ax, 'Feature Ablation — Accuracy Drop When Feature Removed')
ax.set_ylabel('Accuracy Drop')
ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
plt.tight_layout()
plt.savefig('plot_ablation.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("    Saved: plot_ablation.png")

# ─────────────────────────────────────────────
# 7. EXPORT MODEL AS C HEADER (OPTIMIZED)
# ─────────────────────────────────────────────
print("\n[7] Exporting optimized model.h ...")

def export_model_h(clf, feature_names, class_names, filename="model.h"):
    n_trees    = len(clf.estimators_)
    n_features = len(feature_names)
    n_classes  = clf.n_classes_
    lines = []

    lines += [
        "/* =====================================================",
        " * Milk Adulteration Detection Model (Optimized for ESP32)",
        f" * Random Forest: {n_trees} trees | {n_features} features | {n_classes} classes",
        f" * Test Accuracy : {test_acc:.4f}",
        f" * CV Mean       : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}",
        " * Generated by train_final.py",
        " * =====================================================*/",
        "",
        "#ifndef MILK_MODEL_H",
        "#define MILK_MODEL_H",
        "",
        "#include <stdint.h>  // Required for space-saving int types",
        "",
        f"#define N_FEATURES  {n_features}",
        f"#define N_CLASSES   {n_classes}",
        f"#define N_TREES     {n_trees}",
        "",
        "/* ── Class Labels ── */",
    ]
    for i, name in class_names.items():
        lines.append(f'#define CLASS_{i}_LABEL "{name}"')
    lines.append("")

    lines.append("/* ── Feature Index Map ── */")
    for i, fname in enumerate(feature_names):
        safe = fname.upper().replace('-','_').replace('/','_').replace('(','').replace(')','')
        lines.append(f"#define FEAT_{safe}  {i}")
    lines.append("")

    # Per-tree arrays using int8_t and int16_t to save massive space
    for t_idx, tree in enumerate(clf.estimators_):
        t = tree.tree_
        leaf_class = np.argmax(t.value[:, 0, :], axis=1)
        lines += [
            f"/* ── Tree {t_idx} ── */",
            f"static const int8_t  t{t_idx}_feat[]  = {{{','.join(map(str, t.feature))}}};",
            # Dropped the float precision from .6f to .4f to save plain text size in the .h file
            f"static const float   t{t_idx}_thr[]   = {{{','.join(f'{v:.4f}f' for v in t.threshold)}}};",
            f"static const int16_t t{t_idx}_left[]  = {{{','.join(map(str, t.children_left))}}};",
            f"static const int16_t t{t_idx}_right[] = {{{','.join(map(str, t.children_right))}}};",
            f"static const int8_t  t{t_idx}_val[]   = {{{','.join(map(str, leaf_class))}}};",
            "",
        ]

    # Inference helpers
    lines += [
        "/* ── Single Tree Predict ── */",
        "static inline int _tree_predict(",
        "    const int8_t* feat, const float* thr,",
        "    const int16_t* left, const int16_t* right,",
        "    const int8_t* val,  const float* x) {",
        "    int n = 0;",
        "    while (feat[n] >= 0)",
        "        n = (x[feat[n]] <= thr[n]) ? left[n] : right[n];",
        "    return val[n];",
        "}",
        "",
        "/* ── Forest Predict (majority vote) ── */",
        "static inline int predict_adulteration(const float* x) {",
        f"    int votes[{n_classes}] = {{0}};",
    ]
    for t_idx in range(n_trees):
        lines.append(
            f"    votes[_tree_predict(t{t_idx}_feat,t{t_idx}_thr,"
            f"t{t_idx}_left,t{t_idx}_right,t{t_idx}_val,x)]++;"
        )
    lines += [
        f"    int best=0;",
        f"    for(int i=1;i<{n_classes};i++) if(votes[i]>votes[best]) best=i;",
        "    return best;",
        "}",
        "",
        "/* ── Class Name Helper ── */",
        "static inline const char* class_name(int c) {",
        "    switch(c) {",
    ]
    for i, name in class_names.items():
        lines.append(f'        case {i}: return "{name}";')
    lines += [
        '        default: return "Unknown";',
        "    }",
        "}",
        "",
        "#endif /* MILK_MODEL_H */",
    ]

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"    Saved: {filename}")

export_model_h(clf, feature_names, CLASS_NAMES)

# ─────────────────────────────────────────────
# 8. SAVE JSON REPORT
# ─────────────────────────────────────────────
from sklearn.metrics import classification_report as cr_fn
report_dict = cr_fn(y_test, y_pred,
                    target_names=[CLASS_NAMES[i] for i in range(5)],
                    output_dict=True)
report = {
    "train_accuracy":   round(train_acc, 4),
    "test_accuracy":    round(test_acc,  4),
    "gap":              round(gap, 4),
    "cv_mean":          round(float(cv_scores.mean()), 4),
    "cv_std":           round(float(cv_scores.std()),  4),
    "noise_robustness": {str(k): round(v,4) for k,v in zip(ROBUSTNESS_LEVELS, robust_accs)},
    "ablation":         {k: round(v,4) for k,v in ablation.items()},
    "features_used":    feature_names,
    "classification_report": report_dict,
}
with open("training_report.json", 'w') as f:
    json.dump(report, f, indent=2)
print("    Saved: training_report.json")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("  ALL DONE")
print(f"{'='*60}")
print(f"  Test Accuracy  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  CV Mean ± Std  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Train/Test Gap : {gap:.4f}  {'✅' if gap<0.05 else '⚠️'}")
print(f"  Noise Drop     : {robust_accs[0]-robust_accs[-1]:.4f}")
print(f"\n  Files saved:")
print(f"    model.h")
print(f"    training_report.json")
print(f"    plot_confusion.png")
print(f"    plot_feature_importance.png")
print(f"    plot_roc.png")
print(f"    plot_noise_robustness.png")
print(f"    plot_cv_scores.png")
print(f"    plot_class_metrics.png")
print(f"    plot_ablation.png")
print(f"{'='*60}\n")