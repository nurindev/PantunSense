# app.py (Flask backend for PantunSense)
from flask_cors import CORS
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
import os

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

def load_nature_words(filepath=None):
    if filepath is None:
        filepath = os.path.join(app.root_path, 'static', 'nature_words.txt')
    with open(filepath, "r", encoding="utf-8") as f:
        return set(word.strip().lower() for word in f.readlines() if word.strip())
        
nature_words = load_nature_words()


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
        return jsonify({
            "quality": "Poor",
            "reason": "Pantun cannot be empty.",
            "tips": "Write a 4-line traditional pantun with ABAB rhyme and nature metaphors."
        })

    avg_syllables = sum(count_syllables(l) for l in lines) / len(lines)
    rhyme_type = get_rhyme_scheme(lines)
    has_nature_metaphor = any(word in text.lower() for word in nature_words)

    # Check if gibberish or clearly not pantun
    if len(lines) < 2 or avg_syllables < 5:
        return jsonify({
            "quality": "Poor",
            "reason": "This doesn't look like a pantun.",
            "tips": "Write 4 lines with proper words, using nature metaphors and ABAB rhyme."
        })

    # Evaluate structure
    conditions_met = {
        "line_count": len(lines) == 4,
        "rhyme_abab": rhyme_type == "ABAB",
        "syllable_range": 8 <= avg_syllables <= 12,
        "has_nature": has_nature_metaphor
    }

    total_passed = sum(conditions_met.values())

    if total_passed == 4:
        quality = "Good"
        reason = "Excellent! Your pantun matches all traditional structure elements."
        tips = ""
    elif total_passed >= 2:
        quality = "Moderate"
        reason = "Your pantun meets some structural criteria but could be improved."
        tips = ""
        if not conditions_met["line_count"]:
            tips += "Make sure your pantun has 4 lines. "
        if not conditions_met["rhyme_abab"]:
            tips += "Try to use an ABAB rhyme pattern. "
        if not conditions_met["syllable_range"]:
            tips += "Each line should have 8–12 syllables. "
        if not conditions_met["has_nature"]:
            tips += "Use natural elements or metaphors to strengthen imagery. "
    else:
        quality = "Poor"
        reason = "Your pantun doesn't follow the typical structure."
        tips = "Write 4 lines with 8–12 syllables, ABAB rhyme, and nature metaphors."

    return jsonify({
        "quality": quality,
        "reason": reason,
        "tips": tips.strip()
    })

if __name__ == "__main__":
    app.run(debug=True)
