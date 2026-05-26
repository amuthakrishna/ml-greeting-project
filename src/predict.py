import joblib
import numpy as np

# Load model
model = joblib.load("model/intent_model.pkl")

# Input
text = input("Enter text: ").lower().strip()

# Predict probabilities
probs = model.predict_proba([text])[0]

# Max probability
max_prob = np.max(probs)

# Prediction
prediction = model.predict([text])[0]

# Show confidence
print(f"Confidence Score: {max_prob:.2f}")

# Better threshold
THRESHOLD = 0.34

# Logic
if max_prob < THRESHOLD:
    print("Prediction: The data is not available in the dataset.")
else:
    print("Prediction:", prediction)