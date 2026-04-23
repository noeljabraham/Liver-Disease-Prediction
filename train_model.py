import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("liver_patient_dataset.csv")

# Clean column names (VERY important)
df.columns = df.columns.str.strip()

# Debug checks
print("Columns:", df.columns)
print("Unique target values:", df['Selector'].unique())

# Encode target
df['Selector'] = df['Selector'].map({
    'Liver Disease': 1,
    'No Liver Disease': 0
})

# Encode gender
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})

# Drop missing values
df = df.dropna()

# Features and target
X = df.drop('Selector', axis=1)
y = df['Selector']

# Check class balance
print("Class distribution:\n", y.value_counts())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model trained and saved successfully!")