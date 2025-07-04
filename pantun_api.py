from flask_cors import CORS
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
import os
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

# Load trained model and encoders
try:
    model = joblib.load('output/pantun_quality_model.joblib')
    rhyme_encoder = joblib.load('output/rhyme_type_encoder.joblib')
    quality_encoder = joblib.load('output/quality_encoder.joblib')
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    rhyme_encoder = None
    quality_encoder = None

# --- Helper Functions ---
def count_syllables(line):
    vowels = 'aeiouAEIOU'
    syllable_count = 0
    prev_char_was_vowel = False
    
    for char in line:
        if char in vowels:
            if not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False
            
    return syllable_count if syllable_count > 0 else 1

def get_rhyme_suffix(word):
    word = re.sub(r'[^a-zA-Z]', '', word.lower())
    return [word[-i:] for i in range(1, min(len(word), 4))] 

def is_rhyme_match(word1, word2):
    suffixes1 = get_rhyme_suffix(word1)
    suffixes2 = get_rhyme_suffix(word2)
    return any(s1 == s2 for s1 in suffixes1 for s2 in suffixes2)

def get_rhyme_scheme(lines):
    if len(lines) != 4:
        return "Other"
    
    try:
        last_words = [line.strip().split()[-1] for line in lines if line.strip()]
        if len(last_words) != 4:
            return "Other"

        abab = (
            is_rhyme_match(last_words[0], last_words[2]) and
            is_rhyme_match(last_words[1], last_words[3]) and
            not is_rhyme_match(last_words[0], last_words[1])
        )

        aaaa = all(is_rhyme_match(last_words[0], w) for w in last_words[1:])

        if abab:
            return "ABAB"
        elif aaaa:
            return "AAAA"
        else:
            return "Other"
    except:
        return "Other"

def load_nature_words(filepath=None):
    if filepath is None:
        filepath = os.path.join(app.root_path, 'static', 'nature_words.txt')
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return set(word.strip().lower() for word in f.readlines() if word.strip())
    except:
        return set()

nature_words = load_nature_words()

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("pantun", "").strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if not lines:
        return jsonify({
            "quality": "Poor",
            "reason": "Pantun cannot be empty.",
            "tips": "Write a 4-line traditional pantun with ABAB rhyme and nature metaphors."
        })

    # Calculate metrics
    syllable_counts = [count_syllables(line) for line in lines]
    avg_syllables = sum(syllable_counts) / len(lines) if lines else 0
    rhyme_type = get_rhyme_scheme(lines)
    has_nature_metaphor = any(word in text.lower() for word in nature_words)

    # Quality assessment
    conditions = {
        "4_lines": len(lines) == 4,
        "syllables": 8 <= avg_syllables <= 12,
        "nature": has_nature_metaphor,
        "rhyme": rhyme_type == "ABAB"  # Only ABAB is considered best
    }

    met_conditions = sum(conditions.values())

    if all(conditions.values()):
        quality = "Good"
        reason = "Excellent! Your pantun matches all traditional structure elements."
        tips = ""
    elif met_conditions >= 2:
        quality = "Moderate"
        reason = "Your pantun meets some structural criteria but could be improved."
        tips = []

        if not conditions["4_lines"]:
            tips.append("Make sure your pantun has exactly 4 lines.")
        if not conditions["syllables"]:
            tips.append("Aim for 8-12 syllables per line (current avg: {:.1f}).".format(avg_syllables))
        if not conditions["nature"]:
            tips.append("Include nature metaphors for stronger imagery.")
        if not conditions["rhyme"]:
            tips.append("Try using the ABAB rhyme scheme.")

        tips = " ".join(tips)
    else:
        quality = "Poor"
        reason = "Your pantun doesn't follow the typical structure."
        tips = "Write 4 lines with 8-12 syllables each, using ABAB rhyme and nature metaphors."

    return jsonify({
        "quality": quality,
        "reason": reason,
        "tips": tips.strip(),
        "rhyme_scheme": rhyme_type,
        "structure": {
            "line_count": len(lines),
            "avg_syllables": round(avg_syllables, 1),
            "rhyme_scheme": rhyme_type,
            "has_nature": has_nature_metaphor
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
