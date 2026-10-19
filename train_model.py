import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("loan_approval.csv")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Drop ID
if 'Applicant_ID' in df.columns:
    df = df.drop('Applicant_ID', axis=1)

target_col = 'Loan_Approved'
print(f"Target ({target_col}):")
print(df[target_col].value_counts())

# Fill missing PROPERLY (no inplace warnings)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Encode ALL categoricals except target FIRST
label_encoders = {}
object_cols = ['Employment_Status', 'Marital_Status', 'Loan_Purpose', 'Property_Area', 
               'Education_Level', 'Gender', 'Employer_Category']
for col in object_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Encode target CORRECTLY: Yes=1, No=0
df[target_col] = df[target_col].map({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0, 1: 1, 0: 0}).fillna(0).astype(int)
print("Target after mapping:")
print(df[target_col].value_counts())

X = df.drop(target_col, axis=1)
y = df[target_col]
print("Features shape:", X.shape)

# Create model_files directory
os.makedirs('model_files', exist_ok=True)

# Scale numeric
scaler = StandardScaler()
numeric_cols = X.select_dtypes(include='number').columns
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
pickle.dump(scaler, open('model_files/scaler.pkl', 'wb'))

# SMOTE balance (298 Yes → 1000 each)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("After SMOTE:", y_res.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train enhanced RF
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Feature importance
importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
importance.to_csv('model_files/importance.csv', index=False)
print("Top 5 features:")
print(importance.head())

# Save ALL
pickle.dump(model, open('model_files/loan_model.pkl', 'wb'))
pickle.dump(label_encoders, open('model_files/encoders.pkl', 'wb'))
pickle.dump(X.columns.tolist(), open('model_files/columns.pkl', 'wb'))
print("🎉 Model trained perfectly! Run: streamlit run app.py")