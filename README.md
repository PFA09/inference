# 🎙️ Architecture Modulaire ASR - Speech Recognition

## 📋 Vue d'ensemble

Architecture modulaire Python pour **Automatic Speech Recognition (ASR)** optimisée pour la parole pathologique (dysarthrie).

## 📁 Structure du projet

```
inference/
├── inference.py          # Module d'inférence ASR (Wav2Vec2)
├── eda.py               # Analyse Exploratoire des Données
├── requirements.txt     # Dépendances Python
├── README.md           # Documentation
└── reports/
    └── eda_report.pdf  # Rapport EDA généré
```

---

## 🔧 Modules

### 1. **inference.py** - Inférence ASR

Module pour la reconnaissance vocale avec Wav2Vec2.

```python
from inference import SpeechInference

# Initialisation
inference = SpeechInference(
    audio_path="path/to/audio.wav",
    mode="letter"  # 'letter', 'word', ou 'sentences'
)

# Prédiction
result = inference.predict()
print(result)  # "DV93NIXIZ"
```

**Modes**:
- `letter` : Lettres/chiffres individuels (recommandé pour la dysarthrie)
- `word` : Mots en majuscules
- `sentences` : Phrases complètes

**Fonctionnalités**:
- ✅ Prétraitement audio: 16kHz mono
- ✅ Pas de suppression de silences (adapté à la dysarthrie)
- ✅ Mapping robuste français: `zéro`→`0`, `tiret`→`-`, `i grec`→`y`, etc.
- ✅ Auto-détection CPU/GPU (CUDA)
- ✅ Sortie totalement silencieuse (zéro warnings/logs)

---

### 2. **eda.py** - Analyse Exploratoire des Données

Module pour générer des rapports EDA professionnels en PDF.

```python
from eda import DataAnalyzer

# Initialisation
analyzer = DataAnalyzer(
    json_path="path/to/labels.json",
    audio_directory="path/to/audio/dir"
)

# Génération du rapport PDF
analyzer.generate_eda_pdf("eda_report.pdf")
```

**Rapport généré inclut**:
- 📊 **Statistiques globales**: fichiers, types, durées (min/max/moyenne/médiane)
- 📈 **Graphiques haute résolution**:
  - Répartition des classes (type de label + top 10 labels)
  - Distribution des durées avec courbes moyennes/médianes
  - Répartition par session et speaker
- 📋 **Tableaux professionnels**: statistiques globales et par type

---

## 🚀 Installation

### Étape 1: Créer l'environnement

```bash
conda create -n PFA python=3.10 -y
conda activate PFA
```

### Étape 2: Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 📊 Format des données JSON attendu

```json
[
  {
    "file": "audio001.wav",
    "label": "D",
    "type": "Letter",
    "speaker": "client",
    "session": 1,
    "micro": "labri"
  },
  {
    "file": "audio002.wav",
    "label": "1",
    "type": "Digit",
    "speaker": "client",
    "session": 1,
    "micro": "labri"
  }
]
```

---

## 💻 Exemples d'utilisation

### Exemple 1: Inférence simple

```bash
python << 'PYTHON'
from inference import SpeechInference

# Inférence sur un fichier audio
inference = SpeechInference("audio.wav", mode="letter")
result = inference.predict()
print(f"Résultat: {result}")
PYTHON
```

### Exemple 2: Génération du rapport EDA

```bash
python << 'PYTHON'
from eda import DataAnalyzer

# Analyser les données et générer le rapport
analyzer = DataAnalyzer("labels.json", "audio_directory")
analyzer.generate_eda_pdf("eda_report.pdf")
print("Rapport généré: eda_report.pdf")
PYTHON
```

### Exemple 3: Batch processing

```python
from inference import SpeechInference
import os
import json

audio_dir = "path/to/audio"
results = {}

for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        inference = SpeechInference(
            os.path.join(audio_dir, filename),
            mode="letter"
        )
        results[filename] = inference.predict()

# Sauvegarder les résultats
with open("predictions.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## 📦 Dépendances

### Core:
- `torch >= 2.1.0` - Framework deep learning
- `librosa >= 0.10.0` - Traitement audio (sans CUDA)
- `numpy >= 1.24.0` - Calculs numériques
- `pandas >= 2.0.0` - Manipulation de données

### ASR:
- `transformers >= 4.35.0` - Modèles Wav2Vec2 (HuggingFace)

### Visualization & Reporting:
- `matplotlib >= 3.7.0` - Graphiques
- `seaborn >= 0.13.0` - Graphiques statistiques
- `reportlab >= 4.0.0` - Génération PDF

### Audio:
- `soundfile >= 0.12.0` - Lecture durée audio

Voir `requirements.txt` pour plus de détails.

---

## 🎯 Points clés

### Inférence:
- ✅ **Librosa** au lieu de torchaudio (plus stable, pas de dépendances CUDA coûteuses)
- ✅ **Sortie complètement silencieuse** : tous les warnings supprimés
- ✅ **Modèle par défaut** : `jonatasgrosman/wav2vec2-large-xlsr-53-french`
- ✅ **Lazy loading** : modèles chargés au premier appel

### EDA:
- ✅ **PDF natif** : utilise reportlab für une meilleure qualité
- ✅ **Graphiques vectorisés** : PNG haute résolution (100 dpi)
- ✅ **Sans X11** : moteur Agg de matplotlib (pour serveurs headless)

### Parole pathologique:
- ✅ **Pas de suppression de silences** (adapté à la dysarthrie)
- ✅ **Mapping robuste** pour les lettres difficiles (h, y, ç)

---

## 📝 Remarques techniques

- **Au lieu de torchaudio** : Utilise librosa pour la stabilité
- **Suppression des warnings** : Gérée au niveau du système d'exploitation (FD redirection)
- **GPU auto-détection** : CUDA utilisé automatiquement si disponible

---

**Version**: 1.0  
**Dernière mise à jour**: Avril 2026  
**Auteur**: Data Science Team - PFA
