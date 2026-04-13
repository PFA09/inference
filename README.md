# ASR Dysarthrie — Pipeline Wav2Vec2 + LoRA

Fine-tuning et inférence d'un modèle de reconnaissance vocale (ASR) adapté à la parole dysarthrique, basé sur [wav2vec2-large-xlsr-53-french](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french) et la technique LoRA.

---

## Prérequis

- Python 3.10
- GPU CUDA recommandé (le pipeline fonctionne aussi en CPU, mais l'entraînement sera très lent)
- Conda (pour créer l'environnement isolé)

---

## Installation

```bash
conda create -n PFA python=3.10 -y
conda activate PFA
pip install -r requirements.txt
```

> **Note GPU** : la version de `torch` listée dans `requirements.txt` est compilée avec CUDA 13.0.  
> Si votre driver est différent, installez d'abord torch manuellement depuis [pytorch.org](https://pytorch.org/get-started/locally/) avant de lancer `pip install -r requirements.txt`.

> **Note compatibilité** : `accelerate` est épinglé à `0.27.2` car les versions ≥ 0.28 sont incompatibles avec `transformers==4.35.2` (`dispatch_batches` supprimé).

---

## Structure des données

Le pipeline attend un fichier `labels.json` et les fichiers audio dans le **même dossier** :

```
data/
├── labels.json
├── audio001.wav
├── audio002.wav
└── ...
```

Format de `labels.json` — liste de dicts :

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
  ...
]
```

| Champ | Description |
|-------|-------------|
| `file` | Nom du fichier audio (relatif à `labels.json`, ou chemin absolu) |
| `label` | Vérité terrain (lettre, chiffre, mot, phrase…) |
| `type` | `Letter`, `Digit`, `Special`, `Word`, ou `Sentences` |
| `speaker` | Identifiant du locuteur |
| `session` | Numéro de session (utilisé pour le split stratifié) |
| `micro` | Identifiant du micro |

---

## Lancer l'entraînement

Éditez les chemins dans `train.py` :

```python
cfg = ASRConfig(
    mode="letter",          # "letter" | "word" | "sentences"
    paths=PathConfig(
        data_json="/chemin/vers/data/labels.json",
        output_dir="/chemin/vers/training_outputs",
    ),
)
```

Puis lancez :

```bash
conda activate PFA
cd /chemin/vers/ce/dossier
python train.py
```

Le pipeline effectue dans l'ordre :

1. Chargement du modèle de base + injection LoRA
2. Augmentation des données (pitch shift, bruit, décalage temporel)
3. Split train/validation stratifié par session
4. Fine-tuning avec early stopping sur le WER
5. Fusion des adapteurs LoRA dans le modèle de base
6. Évaluation automatique sur le split de validation
7. Génération d'un rapport PDF dans `training_outputs/reports/`

**Artefacts produits :**

```
training_outputs/
├── checkpoint-*/          # checkpoints intermédiaires
├── final_merged_model/    # modèle fusionné prêt pour l'inférence
├── validation_split.json  # enregistrements utilisés en validation
└── reports/
    └── evaluation_letter.pdf
```

---

## Lancer l'inférence

### Inférence sur un fichier audio

```bash
conda activate PFA
python inference.py chemin/vers/audio.wav --mode letter
```

Avec un modèle fine-tuné :

```bash
python inference.py chemin/vers/audio.wav \
    --mode letter \
    --model-path training_outputs/final_merged_model
```

**Arguments :**

| Argument | Défaut | Description |
|----------|--------|-------------|
| `audio_path` | *(requis)* | Chemin vers le fichier audio (`.wav`, `.flac`, `.mp3`…) |
| `--mode` | `letter` | Mode : `letter`, `word`, ou `sentences` |
| `--model-path` | *(modèle HF de base)* | Chemin vers un modèle fine-tuné local |

### Modes d'inférence

| Mode | Types JSON utilisés | Description |
|------|---------------------|-------------|
| `letter` | `Letter`, `Digit`, `Special` | Reconnaissance lettre/chiffre par lettre/chiffre |
| `word` | `Word` | Reconnaissance de mots isolés |
| `sentences` | `Sentences` | Reconnaissance de phrases |

---

## Configuration

Tous les hyperparamètres sont dans `config.py`. Les principales classes :

| Classe | Rôle |
|--------|------|
| `AudioConfig` | Fréquence d'échantillonnage, durée min/max |
| `AugmentConfig` | Probabilités et amplitudes des augmentations |
| `ModelConfig` | Nom du modèle HF, SpecAugment, dropout, LoRA |
| `TrainingConfig` | Epochs, LR, batch size, early stopping… |
| `PathConfig` | Chemins vers les données et les sorties |
| `ASRConfig` | Façade qui regroupe tout, + `mode` |

Exemple de surcharge :

```python
from config import ASRConfig, PathConfig, TrainingConfig

cfg = ASRConfig(
    mode="word",
    paths=PathConfig(
        data_json="/data/labels.json",
        output_dir="/outputs/run_02",
    ),
    training=TrainingConfig(
        num_epochs=20,
        learning_rate=5e-5,
    ),
)
```

---

## Structure du projet

```
inference/
├── config.py                  # Configuration centralisée
├── train.py                   # Point d'entrée entraînement
├── inference.py               # Point d'entrée inférence CLI
├── requirements.txt
│
├── audio/
│   ├── preprocessor.py        # Chargement et normalisation audio
│   └── augmenter.py           # Augmentation (train uniquement)
│
├── data/
│   ├── dataset.py             # Construction datasets HuggingFace
│   └── collator.py            # Collateur CTC avec padding dynamique
│
├── model/
│   ├── loader.py              # Chargement modèle, injection LoRA, fusion
│   ├── decoder.py             # Décodage CTC logits → texte brut
│   └── postprocessor.py       # Post-traitement sémantique (mapping lettre)
│
├── pipeline/
│   ├── trainer.py             # Orchestration du fine-tuning
│   ├── inference.py           # Façade d'inférence fichier par fichier
│   └── evaluator.py           # Évaluation sur le split de validation
│
└── reporting/
    └── pdf_report.py          # Génération du rapport PDF
```
