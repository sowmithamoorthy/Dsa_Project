import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Create folders if they don't exist
os.makedirs('models', exist_ok=True)

# Load Dataset
df = pd.read_csv('data/Intrusion_detection.csv')

# Preprocessing
df.fillna(method='ffill', inplace=True)

# Label Encoding for categorical columns (if any)
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Save label encoders
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Features and Target
# Modify this part to match the correct label column
if 'Label' in df.columns:
    y = df['Label']
    X = df.drop(['Label'], axis=1)
else:
    raise Exception("Target variable not found! Please ensure 'Label' column exists.")

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Use only Decision Tree
model = DecisionTreeClassifier()

print("\nTraining decision_tree...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Classification Report for decision_tree:")
print(classification_report(y_test, y_pred))

# Save model
with open('models/decision_tree.pkl', 'wb') as f:
    pickle.dump(model, f)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f'Confusion Matrix - decision_tree')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('models/decision_tree_confusion_matrix.png')
plt.close()

# For multiclass ROC Curve
# Assuming y_test and model are already defined:

# Get predicted probabilities (required for ROC Curve)
y_prob = model.predict_proba(X_test)

from sklearn.metrics import roc_auc_score

# For multiclass, we need to use One-vs-Rest (OvR)
n_classes = len(np.unique(y_test))
fpr = {}
tpr = {}
roc_auc = {}

# Loop through each class and calculate ROC curve and AUC
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = roc_auc_score(y_test == i, y_prob[:, i])

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multiclass Classification')
plt.legend()
plt.savefig('models/multiclass_roc_curve.png')
plt.close()


# Feature Importance for Decision Tree
if hasattr(model, 'feature_importances_'):
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    plt.figure(figsize=(8,6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances - Decision Tree')
    plt.savefig('models/decision_tree_feature_importance.png')
    plt.close()

print("\nTraining complete and models saved.")
