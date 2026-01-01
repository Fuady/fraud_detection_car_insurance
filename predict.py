import pickle
import pandas as pd
from flask import Flask, request, jsonify

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "models/fraud_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Insurance Fraud Detection API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected input:
    {
        "data": [
            {
                "age": 35,
                "policy_state": "CA",
                "incident_severity": "Total Loss",
                ...
            }
        ]
    }
    """

    try:
        payload = request.get_json()

        if "data" not in payload:
            return jsonify({"error": "Missing 'data' field"}), 400

        df = pd.DataFrame(payload["data"])

        # Predictions
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]

        response = {
            "prediction": prediction.tolist(),
            "fraud_probability": probability.tolist()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
