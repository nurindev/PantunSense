from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load model and encoders
model = joblib.load('output/pantun_quality_model.joblib')
rhyme_encoder = joblib.load('output/rhyme_type_encoder.joblib')
quality_encoder = joblib.load('output/quality_encoder.joblib')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    avg_syllables = float(data['avg_syllables'])
    line_count = int(data['line_count'])
    rhyme_type = data['rhyme_type']

    tips = "" 

    # If rhyme_type is not in encoder, use a fallback (e.g. "Other" or the first class)
    if rhyme_type not in rhyme_encoder.classes_:
        fallback = "Other" if "Other" in rhyme_encoder.classes_ else rhyme_encoder.classes_[0]
        rhyme_type_used = fallback
        reason = "We couldn't recognize the rhyme pattern, so we used the closest match."
    else:
        rhyme_type_used = rhyme_type
        reason = ""

    rhyme_type_encoded = rhyme_encoder.transform([rhyme_type_used])[0]
    features = np.array([[avg_syllables, line_count, rhyme_type_encoded]])
    pred = model.predict(features)[0]
    quality = quality_encoder.inverse_transform([pred])[0]

    if not reason:
        if line_count != 4:
            reason = "A good pantun has 4 lines."
            tips = "Write your pantun in 4 lines. "
        elif rhyme_type_used != "ABAB":
            reason = "A good pantun usually uses an ABAB rhyme pattern."
            tips = "Try to use an ABAB rhyme pattern. "
        elif not (8 <= avg_syllables <= 12):
            reason = "A good pantun usually has 8-12 syllables per line."
            tips = "Aim for 8-12 syllables per line. "
        elif not (9 <= avg_syllables <= 10):
            reason = "Great! Your pantun follows the traditional structure."
            tips = "For an even better pantun, try to have 9 or 10 syllables in each line."
        else:
            reason = "Excellent! Your pantun matches the ideal traditional structure: 4 lines, ABAB rhyme, and 9-10 syllables per line."
            tips = "" 

    # "Poor" for very short or obviously non-pantun input
    if line_count < 2 or avg_syllables < 5:
        quality = "Poor"
        reason = (
            "This doesn't look like a pantun. "
            "A good pantun has 4 lines and each line usually has 8-12 syllables with nature metaphors."
        )
        tips = "Write your pantun in 4 lines. Aim for 8-12 syllables per line. Try to use an ABAB rhyme pattern."

    # Add improvement tips for Moderate/Poor if not already set
    if quality in ["Moderate", "Poor"]:
        if line_count != 4:
            tips += "Write your pantun in 4 lines. "
        if not (8 <= avg_syllables <= 12):
            tips += "Aim for 8-12 syllables per line. "
        if rhyme_type_used != "ABAB":
            tips += "Try to use an ABAB rhyme pattern. "

    return jsonify({'quality': quality, 'reason': reason, 'tips': tips.strip()})

if __name__ == '__main__':
    app.run(port=5000, debug=True)