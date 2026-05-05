# train_ml.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SCALA-Guard ML Model Training (Improved Version)")
print("="*70)

# =========================
# 1. Load Dataset
# =========================
DATA_PATH = "final_selected_features.csv"

try:
    df = pd.read_csv(DATA_PATH)
    print(f"\n✅ Dataset Loaded Successfully")
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"📋 Columns: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"\n❌ Error: {DATA_PATH} not found!")
    print("Please make sure the CSV file exists in the current directory.")
    exit(1)

# Check label distribution
print(f"\n📈 Label Distribution:")
label_counts = df['label'].value_counts()
print(label_counts)
print(f"Benign (0): {label_counts.get(0, 0)}")
print(f"Malicious (1): {label_counts.get(1, 0)}")
malicious_percentage = (label_counts.get(1, 0) / len(df)) * 100
print(f"Malicious Percentage: {malicious_percentage:.2f}%")

if malicious_percentage < 10:
    print(f"\n⚠️ Warning: Dataset is highly imbalanced! Only {malicious_percentage:.2f}% malicious samples.")
    print("Using class weights to handle imbalance...")

# =========================
# 2. Features and Target
# =========================
TARGET_COLUMN = "label"
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

feature_cols = X.columns.tolist()
print(f"\n🔧 Features ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i}. {col}")

# =========================
# 3. Handle Class Imbalance
# =========================
print("\n⚖️ Handling Class Imbalance...")

# Calculate class weights
classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = {classes[0]: class_weights[0], classes[1]: class_weights[1]}
print(f"Class Weights: Benign({classes[0]})={class_weights[0]:.2f}, Malicious({classes[1]})={class_weights[1]:.2f}")

# =========================
# 4. Train Test Split (Stratified)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 80-20 split for better training
    random_state=42,
    stratify=y  # Maintain class distribution
)

print(f"\n📊 Split Summary:")
print(f"Training Shape: {X_train.shape}")
print(f"Testing Shape: {X_test.shape}")
print(f"Training - Benign: {(y_train==0).sum()}, Malicious: {(y_train==1).sum()}")
print(f"Testing - Benign: {(y_test==0).sum()}, Malicious: {(y_test==1).sum()}")

# =========================
# 5. Feature Scaling
# =========================
print("\n📐 Scaling Features...")
scaler = RobustScaler()  # Robust to outliers
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Scaling completed using RobustScaler")

# =========================
# 6. Model Training with Multiple Algorithms
# =========================
print("\n" + "="*70)
print("🤖 Training Multiple Models")
print("="*70)

# Model 1: Random Forest (optimized for imbalance)
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle imbalance
)

# Model 2: Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

# Model 3: Logistic Regression with class weight
lr_model = LogisticRegression(
    C=1.5,
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    solver='lbfgs'
)

# Model 4: SVM with RBF kernel
svm_model = SVC(
    C=1.5,
    kernel='rbf',
    gamma='scale',
    probability=True,
    random_state=42,
    class_weight='balanced'
)

models = {
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'Logistic Regression': lr_model,
    'SVM': svm_model
}

results = {}
best_accuracy = 0
best_model = None
best_model_name = ""

for name, model in models.items():
    print(f"\n📊 Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }
    
    print(f"  ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ✅ F1-Score: {f1:.4f}")
    print(f"  ✅ Precision: {precision:.4f}")
    print(f"  ✅ Recall: {recall:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"  ✅ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"\n🏆 Best Individual Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# =========================
# 7. Ensemble Model (Best Performance)
# =========================
print("\n" + "="*70)
print("🎯 Creating Ensemble Model")
print("="*70)

ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model),
        ('lr', lr_model),
        ('svm', svm_model)
    ],
    voting='soft',  # Use probability voting
    weights=[2, 2, 1, 1]  # Give more weight to tree-based models
)

ensemble_model.fit(X_train_scaled, y_train)
y_pred_ensemble = ensemble_model.predict(X_test_scaled)
y_pred_proba_ensemble = ensemble_model.predict_proba(X_test_scaled)[:, 1]

# Ensemble metrics
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
ensemble_f1 = f1_score(y_test, y_pred_ensemble)
ensemble_precision = precision_score(y_test, y_pred_ensemble)
ensemble_recall = recall_score(y_test, y_pred_ensemble)
ensemble_auc = roc_auc_score(y_test, y_pred_proba_ensemble)

print(f"\n📊 Ensemble Model Results:")
print(f"  ✅ Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
print(f"  ✅ F1-Score: {ensemble_f1:.4f}")
print(f"  ✅ Precision: {ensemble_precision:.4f}")
print(f"  ✅ Recall: {ensemble_recall:.4f}")
print(f"  ✅ ROC-AUC: {ensemble_auc:.4f}")

# Cross-validation for ensemble
cv_scores_ensemble = cross_val_score(ensemble_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"  ✅ CV Accuracy: {cv_scores_ensemble.mean():.4f} (+/- {cv_scores_ensemble.std()*2:.4f})")

# =========================
# 8. Detailed Classification Report
# =========================
print("\n" + "="*70)
print("📋 Detailed Classification Report - Ensemble Model")
print("="*70)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['Benign (0)', 'Malicious (1)']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_ensemble)
print("\nConfusion Matrix:")
print("                 Predicted")
print("              Benign  Malicious")
print(f"Actual Benign    {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"Actual Malicious {cm[1,0]:4d}    {cm[1,1]:4d}")

# Calculate metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
print(f"\n📊 Confusion Matrix Analysis:")
print(f"  True Negatives (Benign correct): {tn}")
print(f"  False Positives (Benign wrong): {fp}")
print(f"  False Negatives (Malicious wrong): {fn}")
print(f"  True Positives (Malicious correct): {tp}")

if fn > tp:
    print(f"\n⚠️ Warning: Model is missing many malicious samples!")
    print(f"   False Negatives: {fn}, True Positives: {tp}")
    print(f"   Consider collecting more malicious samples for training.")

# =========================
# 9. Feature Importance Analysis
# =========================
print("\n" + "="*70)
print("🔍 Feature Importance Analysis")
print("="*70)

# Use Random Forest for feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nAll Features Importance:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']:35s}: {row['importance']:.4f}")

print("\nTop 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  ⭐ {row['feature']:35s}: {row['importance']:.4f}")

# =========================
# 10. Save Model and Artifacts
# =========================
print("\n" + "="*70)
print("💾 Saving Model and Artifacts")
print("="*70)

# Save the best ensemble model
MODEL_PATH = "malicious_detection_model.pkl"
joblib.dump(ensemble_model, MODEL_PATH)
print(f"✅ Model saved -> {MODEL_PATH}")

# Save feature columns
FEATURES_PATH = "feature_columns.pkl"
joblib.dump(feature_cols, FEATURES_PATH)
print(f"✅ Feature columns saved -> {FEATURES_PATH}")

# Save scaler
SCALER_PATH = "scaler.pkl"
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Scaler saved -> {SCALER_PATH}")

# Save training metrics
metrics = {
    'ensemble_accuracy': ensemble_accuracy,
    'ensemble_f1_score': ensemble_f1,
    'ensemble_precision': ensemble_precision,
    'ensemble_recall': ensemble_recall,
    'ensemble_auc': ensemble_auc,
    'feature_importance': feature_importance.to_dict(),
    'confusion_matrix': cm.tolist(),
    'class_distribution': {
        'benign': int(label_counts.get(0, 0)),
        'malicious': int(label_counts.get(1, 0))
    },
    'best_individual_model': best_model_name,
    'individual_model_performance': {
        name: {
            'accuracy': results[name]['accuracy'],
            'f1_score': results[name]['f1_score'],
            'precision': results[name]['precision'],
            'recall': results[name]['recall']
        } for name in results
    }
}
joblib.dump(metrics, 'model_metrics.pkl')
print(f"✅ Metrics saved -> model_metrics.pkl")

# =========================
# 11. Final Summary
# =========================
print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\n📊 Final Model Performance:")
print(f"  🏆 Best Model: Ensemble (RF + GB + LR + SVM)")
print(f"  📈 Test Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
print(f"  📈 F1-Score: {ensemble_f1:.4f}")
print(f"  📈 Precision: {ensemble_precision:.4f}")
print(f"  📈 Recall: {ensemble_recall:.4f}")
print(f"  📈 ROC-AUC: {ensemble_auc:.4f}")
print(f"\n📁 Files Created:")
print(f"  1. malicious_detection_model.pkl - Main ML model")
print(f"  2. feature_columns.pkl - Feature names")
print(f"  3. scaler.pkl - Feature scaler")
print(f"  4. model_metrics.pkl - Training metrics")
print(f"\n🔧 Features Used: {len(feature_cols)}")
print(f"📊 Training Samples: {len(X_train)}")
print(f"🧪 Testing Samples: {len(X_test)}")

# Test prediction with sample
print("\n" + "="*70)
print("🧪 Testing Model with Sample Data")
print("="*70)

# Test with first few samples from test set
print("\nSample Predictions (First 10 test samples):")
test_samples = X_test_scaled[:min(10, len(X_test))]
test_actuals = y_test[:min(10, len(y_test))]

for i in range(len(test_samples)):
    sample = test_samples[i:i+1]
    pred = ensemble_model.predict(sample)[0]
    prob = ensemble_model.predict_proba(sample)[0][1]
    actual = test_actuals.iloc[i] if hasattr(test_actuals, 'iloc') else test_actuals[i]
    status = "✓" if pred == actual else "✗"
    print(f"  Sample {i+1}: Predicted={pred} ({'Malicious' if pred==1 else 'Benign'}), "
          f"Confidence={prob:.2%}, Actual={actual} {status}")

# Verify model loading
print("\n" + "="*70)
print("🔍 Verifying Model Loading")
print("="*70)

try:
    test_model = joblib.load(MODEL_PATH)
    test_features = joblib.load(FEATURES_PATH)
    test_scaler = joblib.load(SCALER_PATH)
    print("✅ All artifacts can be loaded successfully!")
    print(f"   Model type: {type(test_model).__name__}")
    print(f"   Features count: {len(test_features)}")
    print(f"   Scaler type: {type(test_scaler).__name__}")
except Exception as e:
    print(f"❌ Error loading artifacts: {e}")

print("\n🎉 Model is ready for integration with the API!")
print("="*70)