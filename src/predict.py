import joblib
import numpy as np

# Load trained model
model = joblib.load("model/intent_model.pkl")

# User input
text = input("Enter text: ").lower().strip()

# Predict probabilities
probs = model.predict_proba([text])[0]

# Highest probability
max_prob = np.max(probs)

# Predicted label
prediction = model.predict([text])[0]

# Print confidence score
#print(f"Confidence Score: {max_prob:.2f}")

# Threshold
THRESHOLD = 0.50

# Prediction logic
if max_prob < THRESHOLD:
    print("Prediction: Please provide the correrct intent")
else:
    print("Prediction:", prediction)