from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os
from urllib.parse import urljoin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Configure Chrome options
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

# Initialize driver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

base_url = "https://malaycivilization.com.my/items/browse?collection=10"
base_domain = "https://malaycivilization.com.my"
pantuns = []
seen_pantuns = set()

# Classification & ML helper functions 
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
        abab_rhyme = is_rhyme_match(words[0], words[2]) and is_rhyme_match(words[1], words[3])
        aaaa_rhyme = all(is_rhyme_match(words[i], words[0]) for i in range(1, 4))
        if abab_rhyme:
            return "ABAB"
        elif aaaa_rhyme:
            return "AAAA"
        else:
            return "Other"
    except IndexError:
        return "Invalid"

def contains_nature_metaphor(text):
    alam_keywords = [
    "laut", "lautan", "ombak", "taufan", "semesta",  "pantai", "rimba", "sungai", "arus", "bulan", "bintang", "matahari", "pelangi",
    "hujan", "langit", "embun", "bumi", "awan", "ribut", "redup", "angin", "petir", "gunung", "bukit", "hutan", "dahan","pohon", "bunga", "daun",
    "mentari", "bayu", "panas", "senja", "gelombang", "lurah", "tebing", "hilir", "cuaca", "perdu", "rumpun",
    "kaya", "herba", "gelemat", "kantan", "lengkuas", "turi", "kuda", "kelam", "kala", "ular", "rusa", "kedidi", "senduduk", "pandan", "sena", "sirih", "sunti", "baldu", "gerimis",
    "lukah", "muara", "kenari", "undan", "pala", "merekah", "tabir", "sulaman", "terang", "ungka", "sayat", "kenjah", "semalu", "mengkudu", ""
    "palas", "empulur", "duku", "selasih", "cermai", "belukar", "meranti", "leban", "rambai", "bidara", "keladi", "padi", "sawah",
    "halaman", "ijuk", "rotan", "bunga melati", "talas", "rumbia", "pegaga", "bunga selasih", "belukar", "kanji", "kurma", "periuk kera", "bersirih", "pisang emas", "nangka", "pelam", "jambu", "sena", "mengkudu", "kandis",
    "ikan", "berudu", "merbah", "gagak", "angsa", "unggas", "lipan bara", "puyuh", "rerama", "camar", "balam", "pipit", "monyet", "buaya", "singa", "ular", "agas", "ayam"
    ]
    return any(word in text.lower() for word in alam_keywords)

def classify_pantun(pantun_lines, full_text):
    reason = []
    if len(pantun_lines) != 4:
        return "Poor", "Number of lines is not 4"

    word_counts = [len(line.split()) for line in pantun_lines]
    if not all(4 <= wc <= 5 for wc in word_counts):
        reason.append("Word count per line is not in the range of 4-5")

    syllables = [count_syllables(line) for line in pantun_lines]
    if not all(8 <= s <= 12 for s in syllables):
        reason.append("Syllable count per line is not in the range of 8-12")

    rhyme_type = get_rhyme_scheme(pantun_lines)
    if rhyme_type == "AAAA":
        reason.append("End rhyme A-A-A-A")
    elif rhyme_type == "Other":
        reason.append("Rhyme doesn't follow ABAB or AAAA pattern")

    try:
        last_words = [line.split()[-1].lower() for line in pantun_lines]
        if all(word == last_words[0] for word in last_words):
            reason.append("Word repetition at line endings")
    except:
        return "Poor", "Error detecting ending words"

    if not contains_nature_metaphor(full_text):
        reason.append("No nature elements")

    if not reason and rhyme_type == "ABAB":
        return "Good", "Good structure, ABAB rhyme, sufficient syllables & words, contains good methapors"
    elif len(reason) <= 2:
        return "Moderate", "; ".join(reason)
    else:
        return "Poor", "; ".join(reason)

def extract_features(pantun_lines):
    avg_syllables = sum(count_syllables(line) for line in pantun_lines) / len(pantun_lines)
    rhyme_type = get_rhyme_scheme(pantun_lines)
    return {
        "avg_syllables": avg_syllables,
        "rhyme_type": rhyme_type,
        "line_count": len(pantun_lines)
    }

# Web scraping
def extract_pantun_text(item):
    pantun_link = item.find('a', class_='permalink')
    if not pantun_link:
        return None

    lines = []
    for element in pantun_link.find_all(text=True, recursive=True):
        stripped = element.strip()
        if stripped:
            lines.append(stripped)

    text = '\n'.join(lines)
    text = re.sub(r'\n\s+', '\n', text).strip()
    pantun_lines = [line.strip() for line in text.split('\n') if line.strip()]

    if len(pantun_lines) == 4:
        return '\n'.join(pantun_lines)
    print(f"Skipped pantun: Expected 4 lines, got {len(pantun_lines)}")
    return None

try:
    for page in range(1, 790): 
        url = f"{base_url}&page={page}"
        print(f"\nScraping page {page}...")

        try:
            driver.get(url)
            time.sleep(random.uniform(2, 4))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            items = soup.find_all('div', class_='item hentry')

            if not items:
                print("No pantun items found - check class name")
                break

            print(f"Found {len(items)} pantun containers")

            for item in items:
                try:
                    pantun_text = extract_pantun_text(item)
                    if not pantun_text:
                        continue

                    pantun_lines = pantun_text.split('\n')
                    pantun_id = item.find('a', class_='permalink')['href'].split('/')[-1]

                    if pantun_text in seen_pantuns:
                        print(f"  - Duplicate skipped: {pantun_id}")
                        continue

                    seen_pantuns.add(pantun_text)

                    quality, reason = classify_pantun(pantun_lines, pantun_text)
                    features = extract_features(pantun_lines)

                    pantuns.append({
                        'id': len(pantuns) + 1,
                        'pantun': pantun_text,
                        'quality': quality,
                        'reason': reason,
                        'avg_syllables': features['avg_syllables'],
                        'rhyme_type': features['rhyme_type'],
                        'line_count': features['line_count']
                    })

                    print(f"  - Collected pantun {len(pantuns)}")
                    print(f"    Sample: {pantun_text[:50]}...")

                except Exception as e:
                    print(f"Error processing item: {str(e)}")
                    continue

            time.sleep(random.uniform(3, 6))

        except Exception as e:
            print(f"Error on page {page}: {str(e)}")
            continue

finally:
    driver.quit()

# Save to CSV
os.makedirs('output', exist_ok=True)
csv_path = os.path.join('output', 'pantun_dataset.csv')
df = pd.DataFrame(pantuns, columns=['id', 'pantun', 'quality', 'reason', 'avg_syllables', 'rhyme_type', 'line_count'])
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n Successfully saved {len(df)} pantuns to '{csv_path}'")

# ML Classification
print("\n Training machine learning model for pantun quality classification...")

X = df[['avg_syllables', 'line_count', 'rhyme_type']].copy()
X['rhyme_type'] = LabelEncoder().fit_transform(X['rhyme_type'])
y = LabelEncoder().fit_transform(df['quality'])

if len(set(y)) < 2:
    print("Not enough classes to train ML model. Need at least 2 different quality labels.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n Classification Results:")
    print(classification_report(y_test, y_pred))
