# app.py (Flask backend for PantunSense)
from flask_cors import CORS
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re

app = Flask(__name__)
CORS(app)

# Load trained model and encoders
model = joblib.load('output/pantun_quality_model.joblib')
rhyme_encoder = joblib.load('output/rhyme_type_encoder.joblib')
quality_encoder = joblib.load('output/quality_encoder.joblib')

# --- Helper Functions ---
def count_syllables(line):
    return len(re.findall(r'[aeiouAEIOU]+', line))

def get_rhyme_suffix(word):
    word = re.sub(r'[^a-zA-Z]', '', word.lower())
    return [word[-i:] for i in range(1, min(len(word), 3) + 1)]

def is_rhyme_match(word1, word2):
    suffixes1 = get_rhyme_suffix(word1)
    suffixes2 = get_rhyme_suffix(word2)
    return any(s1 == s2 for s1 in suffixes1 for s2 in suffixes2)

def get_rhyme_scheme(lines):
    if len(lines) != 4:
        return "Invalid"
    try:
        words = [line.split()[-1] for line in lines]
        abab = is_rhyme_match(words[0], words[2]) and is_rhyme_match(words[1], words[3])
        aaaa = all(is_rhyme_match(words[i], words[0]) for i in range(1, 4))
        if abab:
            return "ABAB"
        elif aaaa:
            return "AAAA"
        else:
            return "Other"
    except:
        return "Invalid"

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("pantun", "")
    lines = text.strip().split("\n")
    lines = [l.strip() for l in lines if l.strip()]

    if not lines:
        return jsonify({"quality": "Poor", "reason": "Pantun cannot be empty.", "tips": "Write a 4-line traditional pantun with ABAB rhyme."})

    avg_syllables = sum(count_syllables(l) for l in lines) / len(lines)
    rhyme_type = get_rhyme_scheme(lines)

    tips = ""
    reason = ""
    fallback = "Other" if "Other" in rhyme_encoder.classes_ else rhyme_encoder.classes_[0]
    rhyme_type_used = rhyme_type if rhyme_type in rhyme_encoder.classes_ else fallback

    try:
        rhyme_encoded = rhyme_encoder.transform([rhyme_type_used])[0]
    except:
        rhyme_encoded = 0

    features = np.array([[avg_syllables, len(lines), rhyme_encoded]])
    pred = model.predict(features)[0]
    quality = quality_encoder.inverse_transform([pred])[0]

    if len(lines) < 2 or avg_syllables < 5:
        quality = "Poor"
        reason = "This doesn't look like a pantun."
        tips = "A good pantun has 4 lines with 8-12 syllables each and uses ABAB rhyme."
    else:
        if len(lines) != 4:
            reason = "A good pantun has 4 lines."
            tips += "Write your pantun in 4 lines. "
        elif rhyme_type_used != "ABAB":
            reason = "A good pantun usually uses an ABAB rhyme pattern."
            tips += "Try to use an ABAB rhyme pattern. "
        elif not (8 <= avg_syllables <= 12):
            reason = "A good pantun usually has 8-12 syllables per line."
            tips += "Aim for 8-12 syllables per line. "
        else:
            reason = "Excellent! Your pantun matches the ideal traditional structure."
            tips = ""

    return jsonify({'quality': quality, 'reason': reason, 'tips': tips.strip()})

if __name__ == "__main__":
    app.run(debug=True)
