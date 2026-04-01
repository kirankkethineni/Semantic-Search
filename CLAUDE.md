# Project: Semantic-Search

## Language & Conventions
- Python project (Jupyter notebooks)
- Always use snake_case (Python convention)

## Paper
- Title: "Semantic-Search: A Knowledge-Driven Classification Method for Accurate Identification of Plant Diseases"
- Authors: Kiran K. Kethineni, Saraju P. Mohanty, Elias Kougianos
- Published: IEEE Transactions on AgriFood Electronics, 2025
- Institution: University of North Texas
- Project page: https://kirankkethineni.github.io/Semantic-Search

## Repository Structure
- `Semantic_Classification.ipynb` — Step 1: trains 87K-param SeparableConv2D CNN to classify leaf images into 7 semantic shape categories (Spots, Stripes, Flecks, Patches, Powdery, Mosaic, Velvety). Saves `semantic_classification_model.h5`.
- `Semantic_Segmentation.ipynb` — Step 2: trains 1.45M-param U-Net segmentation model for binary leaf disease masks. Used only for spot/stripe classes at inference. Saves `semantic_segmentation_model.h5`.
- `Spacy.ipynb` — Step 3: trains spaCy tok2vec NER model to extract PLANT, DISEASE, COLOR, SEMANTIC entities from plain-text disease descriptions. 98% F1. Exports trained pipeline.
- `TAFER1.ipynb` — Step 4: end-to-end Flask app. Loads all three models, initializes SQLite knowledge base, serves 5 routes. Uses pyngrok for remote access.
- `docs/index.html` — GitHub Pages project site (full paper walkthrough, screenshots, results tables)

## System Architecture
Two workflows sharing a SQLite knowledge base:

Learning (add new disease):
  Text Description → spaCy NER → SQL INSERT → Knowledge Base (no retraining)

Inference (classify leaf):
  Image + Crop → CNN Classification → Segmentation (spots/stripes only) → HSV Color Filters → Weighted Jaccard DB Query → Disease Label

## Knowledge Base (SQLite)
- `diseases` (id, plant_name, disease_name, semantics_id)
- `colors` (id, value)
- `semantics` (id, value) — Spots, Flecks, Mosaic, Stripes, Curls, Powdery, Velvety
- `disease_colors` (disease_id, color_id) — many-to-many

## Flask Routes (TAFER1.ipynb)
- `/` GET — home, lists all disease records
- `/add_disease` GET/POST — enter plain-text disease description
- `/entities` POST — confirm NER extraction, commits SQL INSERT
- `/classify` GET/POST — upload leaf image + select crop, runs full pipeline
- `/api/classify` POST — internal JSON API

## Key Results
- 90% end-to-end accuracy on 21 disease classes, 11K images (PlantVillage + PlantDoc)
- CNN classification: 95.28% validation accuracy
- Segmentation: 93% accuracy, 0.73 mean IoU
- NER: 98.98% F1
- Inference: 132ms on NVIDIA L4 GPU
- Zero retraining required to add new disease classes

## Git / GitHub
- Remote: https://github.com/kirankkethineni/Semantic-Search
- Branch: main
- GitHub Pages: https://kirankkethineni.github.io/Semantic-Search
- Ignore: .idea/, docs/.idea/, .claude/
