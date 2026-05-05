"""
SCALA-Guard ML Model Training
Run: python train_model.py
"""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

print("=" * 60)
print("  SCALA-Guard ML Model Training v2.0")
print("=" * 60)

# ─── Synthetic Dataset ──────────────────────────────────────────
np.random.seed(42)
N_SAMPLES = 2000

# Benign packages (label=0)
benign_syscalls = np.random.randint(50, 300, size=N_SAMPLES // 2)
benign_network = np.random.randint(0, 2, size=N_SAMPLES // 2)
benign_file_access = np.random.randint(0, 3, size=N_SAMPLES // 2)
benign_exfil_kb = np.zeros(N_SAMPLES // 2)
benign_proc_spawn = np.zeros(N_SAMPLES // 2)

# Malicious packages (label=1)
malicious_syscalls = np.random.randint(400, 2000, size=N_SAMPLES // 2)
malicious_network = np.random.randint(2, 10, size=N_SAMPLES // 2)
malicious_file_access = np.random.randint(3, 10, size=N_SAMPLES // 2)
malicious_exfil_kb = np.random.randint(10, 500, size=N_SAMPLES // 2)
malicious_proc_spawn = np.random.randint(1, 5, size=N_SAMPLES // 2)

X_benign = np.column_stack([
    benign_syscalls, benign_network, benign_file_access,
    benign_exfil_kb, benign_proc_spawn
])
X_malicious = np.column_stack([
    malicious_syscalls, malicious_network, malicious_file_access,
    malicious_exfil_kb, malicious_proc_spawn
])

X = np.vstack([X_benign, X_malicious])
y = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2))

# ─── Train/Test Split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Dataset: {len(X)} samples ({N_SAMPLES//2} benign, {N_SAMPLES//2} malicious)")
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ─── Model Pipeline ─────────────────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ))
])

print("\n🔧 Training Random Forest (200 trees)...")
pipeline.fit(X_train, y_train)

# ─── Evaluation ─────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")

print("\n" + "=" * 40)
print("  📈 Evaluation Results")
print("=" * 40)
print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))
print(f"5-Fold CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ─── Save Model ─────────────────────────────────────────────────
joblib.dump(pipeline, "scala_guard_model.pkl")
print("\n✅ Model saved: scala_guard_model.pkl")
print("   Features: [syscall_count, network_connections, file_access_count, exfil_kb, proc_spawn]")

