"""
config.py — Configuration centralisée du pipeline ASR dysarthrie.

Toutes les hyperparamètres, chemins, et paramètres de modèle sont définis ici.
Aucun autre module ne doit contenir de valeurs "magiques".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

InferenceMode = Literal["letter", "word", "sentences"]
Mode = InferenceMode  # alias utilisé dans pipeline/inference.py et postprocessor.py


# ──────────────────────────────────────────────────────────────────────────────
# Audio
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AudioConfig:
    """Paramètres de prétraitement audio."""

    sample_rate: int = 16_000
    """Fréquence d'échantillonnage cible (Hz). Wav2Vec2 attend 16 kHz."""

    mono: bool = True
    """Conversion en mono avant extraction de features."""

    max_duration_s: float = 30.0
    """Durée maximale acceptée (secondes). Les fichiers plus longs sont rejetés."""

    min_duration_s: float = 0.05
    """Durée minimale acceptée (secondes). En-dessous, le fichier est ignoré."""

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate doit être positif, reçu : {self.sample_rate}")
        if self.max_duration_s <= self.min_duration_s:
            raise ValueError("max_duration_s doit être > min_duration_s")


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AugmentConfig:
    """
    Paramètres d'augmentation de données.

    Appliqué UNIQUEMENT pendant l'entraînement, jamais à l'inférence.
    Toutes les probabilités sont indépendantes et cumulables.
    """

    enabled: bool = True
    """Active ou désactive toutes les augmentations."""

    # ── Décalage temporel ──────────────────────────────────────────────────
    shift_prob: float = 0.3
    """Probabilité d'appliquer un décalage temporel (time shift)."""

    shift_max_fraction: float = 0.1
    """Fraction maximale de la durée du signal décalée (0.1 = 10 %)."""

    # ── Bruit gaussien ─────────────────────────────────────────────────────
    noise_prob: float = 0.4
    """Probabilité d'ajouter du bruit gaussien."""

    noise_min_snr_db: float = 20.0
    """SNR minimum en dB pour le bruit ajouté. Plus haut = moins de bruit."""

    noise_max_snr_db: float = 40.0
    """SNR maximum en dB pour le bruit ajouté."""

    # ── Pitch shift ────────────────────────────────────────────────────────
    pitch_prob: float = 0.3
    """Probabilité d'appliquer un pitch shift."""

    pitch_semitones_range: float = 2.0
    """Amplitude maximale du pitch shift (±N demi-tons)."""

    def validate(self) -> None:
        for attr, val in [
            ("shift_prob", self.shift_prob),
            ("noise_prob", self.noise_prob),
            ("pitch_prob", self.pitch_prob),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr} doit être entre 0 et 1, reçu : {val}")
        if self.noise_min_snr_db > self.noise_max_snr_db:
            raise ValueError("noise_min_snr_db doit être ≤ noise_max_snr_db")


# ──────────────────────────────────────────────────────────────────────────────
# LoRA
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LoraConfig:
    """
    Paramètres LoRA (Low-Rank Adaptation).

    Calibrés pour un corpus ~800 samples sur un modèle 300 M paramètres.
    """

    r: int = 16
    """Rang des matrices de décomposition. r=16 > r=8 pour un modèle 300M."""

    alpha: int = 32
    """Facteur d'échelle LoRA. Conventionnellement alpha = 2 × r."""

    dropout: float = 0.15
    """Dropout LoRA. Légèrement plus fort que la valeur standard (0.10)
    pour compenser la petite taille du corpus."""

    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"]
    )
    """Modules de l'attention sur lesquels appliquer LoRA."""

    bias: str = "none"
    """Stratégie de biais LoRA. 'none' = seuls les poids A/B sont entraînés."""


# ──────────────────────────────────────────────────────────────────────────────
# Modèle
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    """
    Paramètres du modèle Wav2Vec2 (architecture, SpecAugment, dropout, LoRA).
    """

    base_model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
    """Modèle de base HuggingFace. Fine-tuné XLSR-53 sur le français."""

    # ── SpecAugment ────────────────────────────────────────────────────────
    mask_time_prob: float = 0.08
    """Fraction de timesteps masqués (SpecAugment temporel)."""

    mask_time_length: int = 10
    """Longueur de chaque masque temporel (frames)."""

    mask_feature_prob: float = 0.05
    """Fraction de features fréquentielles masquées."""

    mask_feature_length: int = 10
    """Longueur de chaque masque fréquentiel."""

    # ── Dropout ────────────────────────────────────────────────────────────
    activation_dropout: float = 0.1
    hidden_dropout: float = 0.1
    feat_proj_dropout: float = 0.1
    attention_dropout: float = 0.1
    final_dropout: float = 0.1

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora: LoraConfig = field(default_factory=LoraConfig)


# ──────────────────────────────────────────────────────────────────────────────
# Entraînement
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainingConfig:
    """
    Hyperparamètres d'entraînement.

    Justification de chaque valeur documentée inline — issue de l'EDA
    (corpus dysarthrique mono-locuteur, ~1 228 fichiers courts).
    """

    # ── Données ──────────────────────────────────────────────────────────────
    val_split: float = 0.20
    """Fraction de validation. 20 % sur petit corpus > 15 % pour signal fiable."""

    random_seed: int = 42
    """Graine globale (split, augmentation, init)."""

    # ── Batch / accumulation ─────────────────────────────────────────────────
    per_device_train_batch_size: int = 4
    """Batch size par GPU. Petit = plus de steps sur corpus court."""

    per_device_eval_batch_size: int = 4

    gradient_accumulation_steps: int = 4
    """Batch effectif = 4 × 4 = 16. Même throughput que batch=8 avec accum=2."""

    # ── Optimiseur ───────────────────────────────────────────────────────────
    num_epochs: int = 15
    """Nombre maximal d'époques. L'early stopping arrête avant si besoin."""

    learning_rate: float = 1e-4
    """LR conservateur (3e-4 risque l'overshoot sur ~800 samples)."""

    weight_decay: float = 0.01
    """Régularisation L2. Absent dans l'original — nécessaire ici."""

    warmup_ratio: float = 0.15
    """Warmup plus long car les steps sont courts (fichiers ~0.7 s)."""

    # ── Early stopping ───────────────────────────────────────────────────────
    early_stopping_patience: int = 5
    """Patience augmentée (3 → 5) car le set de val est petit et bruité."""

    metric_for_best_model: str = "wer"
    """Métrique de sélection du meilleur checkpoint."""

    greater_is_better: bool = False
    """False car WER/CER = plus bas est meilleur."""

    # ── Checkpointing ────────────────────────────────────────────────────────
    eval_steps: int = 20
    """Évaluation toutes les N steps."""

    save_steps: int = 20
    """Sauvegarde alignée sur l'évaluation."""

    save_total_limit: int = 3
    """Garder au maximum 3 checkpoints sur disque."""

    logging_steps: int = 10


# ──────────────────────────────────────────────────────────────────────────────
# Chemins
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    """
    Chemins du projet.

    Mutable (pas frozen) car calculés dynamiquement selon l'environnement.
    """

    data_json: Path
    """JSON d'entraînement. Format attendu : liste de dicts {file, label, type, session, micro, speaker}."""

    output_dir: Path
    """Répertoire racine des artefacts d'entraînement."""

    @property
    def audio_dir(self) -> Path:
        """Répertoire audio = même dossier que le JSON."""
        return self.data_json.parent

    @property
    def merged_model_dir(self) -> Path:
        return self.output_dir / "final_merged_model"

    @property
    def validation_json(self) -> Path:
        return self.output_dir / "validation_split.json"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    def makedirs(self) -> None:
        """Crée tous les répertoires nécessaires."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.merged_model_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        self.data_json = Path(self.data_json)
        self.output_dir = Path(self.output_dir)
        if not self.data_json.exists():
            raise FileNotFoundError(f"JSON introuvable : {self.data_json}")


# ──────────────────────────────────────────────────────────────────────────────
# Config globale (façade)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ASRConfig:
    """
    Point d'entrée unique pour toute la configuration.

    Usage :
        cfg = ASRConfig(
            mode="letter",
            paths=PathConfig(
                data_json=Path("/data/labels.json"),
                output_dir=Path("/outputs/run_01"),
            )
        )
    """

    mode: InferenceMode
    """Mode d'inférence : 'letter', 'word' ou 'sentences'."""

    paths: PathConfig
    audio: AudioConfig = field(default_factory=AudioConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Mapping des types JSON → mode d'inférence (utilisé par dataset.py et evaluator.py)
    MODE_TO_TYPES: dict = field(
        default_factory=lambda: {
            "letter": ["Letter", "Digit", "Special"],
            "word": ["Word"],
            "sentences": ["Sentences"],
        }
    )

    def validate(self) -> None:
        """Valide la cohérence de la configuration."""
        if self.mode not in ("letter", "word", "sentences"):
            raise ValueError(f"Mode invalide : {self.mode!r}")
        self.audio.validate()
        self.augment.validate()
