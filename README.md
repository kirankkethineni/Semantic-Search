# Semantic-Search: Knowledge-Driven Plant Disease Classification

**IEEE Transactions on AgriFood Electronics · 2025**

> Kiran K. Kethineni · Saraju P. Mohanty · Elias Kougianos  
> Department of Computer Science & Engineering / Electrical Engineering  
> University of North Texas, USA

[Project Page](https://kirankkethineni.github.io/Semantic-Search) · [IEEE Paper](https://ieeexplore.ieee.org) · [GitHub](https://github.com/kirankkethineni/Semantic-Search)

---

## Overview

Instead of training a separate CNN output neuron for every disease class, Semantic-Search classifies plant diseases through **semantic understanding + knowledge base lookup**. A CNN detects *what the leaf looks like* (spots, stripes, powdery coating, etc.) and *what colors are present* — then queries a structured database to find the best-matching disease. Adding a new disease requires only a plain-text sentence description and a database insert. No retraining. No new labeled images.

| Metric | Value |
|---|---|
| End-to-end accuracy (21 disease classes) | **90%** |
| CNN semantic classification accuracy | **95.28%** |
| Segmentation mean IoU | **0.73** |
| NER F1-score | **98%** |
| Images evaluated | **11,000** |
| Inference time (NVIDIA L4 GPU) | **132 ms** |
| Retraining required to add a new disease | **0** |

---

## Core Idea

Every plant leaf disease can be described as a combination of a **semantic shape** (spots, stripes, patches...) and a set of **colors**. This structured representation is compact, class-agnostic, and human-readable.

**Conventional CNN approach:**
- Trains a separate output neuron per disease class
- Requires thousands of labeled images for each new disease
- Full model retraining to add any new class
- Black-box — no explanation for its output

**Semantic-Search approach:**
- CNN detects semantic features (shape + color), not disease names
- New diseases added via a one-sentence text description
- Zero retraining — knowledge base updated with a SQL `INSERT`
- Explainable — output includes the matched semantics and colors

---

## System Architecture

The system has two distinct workflows sharing a common SQLite knowledge base.

**Learning Pathway (adding a new disease):**
```
Text Description → Tokenization (spaCy tok2vec) → NER Extraction → SQL INSERT → Knowledge Base Updated
```

Example input:
```
"apple crops affected by cedar rust disease have spots with brown, yellow colors"
```

Extracted entities:
- `PLANT` → apple
- `DISEASE` → cedar rust
- `SEMANTIC` → spots
- `COLOR` → brown, yellow

**Inference Pathway (classifying a leaf image):**
```
Leaf Image + Crop Name → CNN Classification → Segmentation (if spots/stripes) → HSV Color Filters → DB Query (Weighted Jaccard) → Disease Label + Explanation
```

---

## Repository Structure

```
Semantic-Search/
├── Semantic_Classification.ipynb   # Step 1 — Train semantic shape CNN
├── Semantic_Segmentation.ipynb     # Step 2 — Train U-Net segmentation model
├── Spacy.ipynb                     # Step 3 — Train spaCy NER model
├── TAFER1.ipynb                    # Step 4 — Run end-to-end Flask app
└── docs/                           # GitHub Pages project site
```

---

## Setup

**Requirements:** Python 3.8+, Jupyter, TensorFlow/Keras, spaCy, Flask, OpenCV, SQLite3, pyngrok

```bash
pip install tensorflow spacy flask opencv-python pyngrok
python -m spacy download en_core_web_trf
```

---

## Running the Notebooks (in order)

### Step 1 — `Semantic_Classification.ipynb`

Trains a lightweight CNN (87K parameters) to classify leaf images into 7 semantic shape categories:

| Category | Description |
|---|---|
| Spots | Circular / irregular discolorations |
| Stripes | Linear streaks of different color |
| Flecks | Tiny scattered specks |
| Patches | Irregular blotch areas |
| Powdery | Dust-like white/gray coating |
| Mosaic | Mixed normal and discolored regions |
| Velvety | Soft, thick, plush-like texture |

**Model architecture:**
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(),
    SeparableConv2D(32,  (3,3), activation='relu'), MaxPooling2D(),
    SeparableConv2D(64,  (3,3), activation='relu'), MaxPooling2D(),
    SeparableConv2D(128, (3,3), activation='relu'), MaxPooling2D(),
    # ... additional 128-filter blocks ...
    GlobalAveragePooling2D(),
    Dense(32, activation='relu'),
    Dense(7,  activation='softmax')   # 7 semantic classes
])
# Total: 87,207 trainable parameters
# Optimizer: Adam | Loss: sparse_categorical_crossentropy | Input: 256×256 RGB
```

Save the trained model as `semantic_classification_model.h5`.

---

### Step 2 — `Semantic_Segmentation.ipynb`

Trains a U-Net-style segmentation model (1.45M parameters) to generate binary masks of diseased leaf regions. Used only for spot and stripe classes at inference — powdery, velvety, and mosaic patterns cover the whole leaf and don't need localization.

**Architecture:** Encoder with `SeparableConv2D + MaxPooling2D` → Decoder with `Conv2DTranspose (upsampling) + Concatenate (skip connections) + SeparableConv2D` → binary mask output `(None, 256, 256, 2)`.

Trained on 3,000 images. Achieves **93% accuracy** and **0.73 mean IoU**.

Save the trained model as `semantic_segmentation_model.h5`.

---

### Step 3 — `Spacy.ipynb`

Trains the spaCy NER model on the tok2vec pipeline to recognize four entity types from disease descriptions.

**Training data:** 15,400 documents (80% train / 20% dev), generated by augmenting 120 real disease descriptions from PlantVillage using GPT-based paraphrasing → 36,000 total records.

**NER performance:**

| Entity Type | F1 Score | Example Values |
|---|---|---|
| PLANT | 99.77% | Apple, Tomato, Corn, Grape, Pepper |
| DISEASE | 99.35% | Powdery Mildew, Rust, Blight, Black Rot |
| COLOR | 98.34% | Yellow, Brown, Dark Green, Gray |
| SEMANTIC | 99.27% | Spots, Stripes, Patches, Powdery |
| **Overall** | **98.98%** | |

**Inference example:**
```python
import spacy
nlp = spacy.load("path/to/trained_ner_model")

doc = nlp("apple crops affected by cedar rust have spots with brown, yellow colors")
for ent in doc.ents:
    print(ent.text, ent.label_)
# apple      → PLANT
# cedar rust → DISEASE
# spots      → SEMANTIC
# brown      → COLOR
# yellow     → COLOR
```

Export the trained pipeline for use in the Flask app.

---

### Step 4 — `TAFER1.ipynb`

The full end-to-end Flask application. Loads all three trained models, initializes the SQLite knowledge base, and exposes the following routes:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Home — displays all disease records in the knowledge base |
| `/add_disease` | GET / POST | Add Disease — user enters a plain-text description |
| `/entities` | POST | Confirm NER extraction and commit to database |
| `/classify` | GET / POST | Classify Leaf — upload image + select crop type |
| `/api/classify` | POST | Internal JSON API called by the classify route |

Run the notebook to start the Flask server. Uses **pyngrok** for remote access — the cell prints a public URL you can open in any browser.

---

## Knowledge Base

The SQLite database has four tables:

| Table | Columns | Purpose |
|---|---|---|
| `diseases` | id, plant_name, disease_name, semantics_id | One row per disease |
| `colors` | id, value | Lookup table of valid color names |
| `semantics` | id, value | Lookup table: Spots, Flecks, Mosaic, Stripes, Curls, Powdery, Velvety |
| `disease_colors` | disease_id, color_id | Many-to-many: each disease can have multiple colors |

**Adding a record (Algorithm 1):**
```python
# 1. Resolve semantics_id
cur.execute("SELECT id FROM semantics WHERE value = ?", (semantic,))
semantics_id = cur.fetchone()[0]

# 2. Insert disease record
cur.execute("INSERT INTO diseases (plant_name, disease_name, semantics_id) VALUES (?,?,?)",
            (plant, disease, semantics_id))
disease_id = cur.lastrowid

# 3. Link each color
for color in colors_list:
    cur.execute("SELECT id FROM colors WHERE value = ?", (color,))
    color_id = cur.fetchone()[0]
    cur.execute("INSERT OR IGNORE INTO disease_colors VALUES (?,?)", (disease_id, color_id))
conn.commit()
```

**Classifying via Weighted Jaccard similarity (Algorithm 2):**
```python
# Weights: first (most dominant) color gets weight 1.0, decreasing by 0.2
weight_dict = {color: max(0.1, 1.0 - 0.2 * i) for i, color in enumerate(colors_list)}

best_score, best_record = -1, None
for record in sql_results:  # filtered by plant_name + semantics
    record_colors = set(record[4].split(','))
    union = colors_list_set | record_colors
    weighted_intersection = sum(weight_dict[c] for c in colors_list_set & record_colors)
    score = weighted_intersection / len(union)
    if score > best_score:
        best_score, best_record = score, record
```

---

## Results

Evaluated on 11,000 images from PlantVillage + PlantDoc across **21 disease classes** and **11 plant species**.

| Disease | Accuracy | F1 |
|---|---|---|
| Corn rust | 100% | 99.67% |
| Orange citrus bacterial spot | 100% | 100% |
| Pepper bacterial spot | 100% | 100% |
| Tomato mosaic virus | 100% | 99.01% |
| Potato blight | 99.60% | 99.79% |
| Squash mildew | 98.00% | 98.99% |
| Tomato curl virus | 97.80% | 96.87% |
| Grape esca | 95.14% | 97.51% |
| Cherry mildew | 91.67% | 95.65% |
| Apple scab | 84.81% | 88.35% |
| ... (21 classes total) | | |
| **Average** | **90%** | **91%** |

**Comparison with prior work:**

| Approach | Semantic Understanding | Retraining for New Classes | Accuracy |
|---|---|---|---|
| CNN-based models | Limited (pixel-level) | Required | 97% |
| Histogram-based SVM | No | Required | 80% |
| Bag of Visual Words | No | Required | 82% |
| Knowledge Graph | Yes (complex) | No | 83% |
| **Semantic-Search (ours)** | **Yes — Shape + Color + NLP** | **No — DB update only** | **90%** |

The 7-point accuracy gap vs. CNN baselines is the trade-off for gaining zero retraining, transparent decisions, and instant extensibility to new disease classes.

---

## Citation

```bibtex
@article{kethineni2025semantic,
  title   = {Semantic-Search: A Knowledge-Driven Classification Method
             for Accurate Identification of Plant Diseases},
  author  = {Kethineni, Kiran K. and Mohanty, Saraju P. and Kougianos, Elias},
  journal = {IEEE Transactions on AgriFood Electronics},
  year    = {2025},
  doi     = {10.1109/TAFE.2025.XXXXXXX}
}
```

---

Kiran K. Kethineni · Saraju P. Mohanty · Elias Kougianos  
[Smart Electronics Systems Laboratory](https://smartelectronics.unt.edu) · University of North Texas
