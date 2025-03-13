import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('preprocessed_data.csv')

# Print column names for debugging
print("Columns in the dataset:", df.columns.tolist())

# Define features (X) and target (y)
# Corrected column names based on your dataset
X = df[['Usage Time (mins)', 'Count of Survey Attempts']]
y = (df['Count of Survey Attempts'] > 0).astype(int)  # Binary target: 1 if attempts > 0, else 0

# Train a model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model to a file
joblib.dump(model, 'model.pkl')

print("Model trained and saved as model.pkl")
