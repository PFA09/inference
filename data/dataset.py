"""
data/dataset.py — Construction des datasets HuggingFace pour l'entraînement.

Responsabilités :
- Charger et filtrer le JSON selon le mode
- Prétraiter les waveforms + labels
- Split stratifié par session (2 sessions dans le corpus)
- Exporter le JSON de validation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Mapping mode → types JSON acceptés
MODE_TO_TYPES: Dict[str, List[str]] = {
    "letter": ["Letter", "Digit", "Special"],
    "word": ["Word"],
    "sentences": ["Sentences"],
}


class DatasetBuilder:
    """
    Construit les datasets train/val à partir d'un fichier JSON annoté.

    Parameters
    ----------
    json_path:
        Chemin vers labels.json.
    mode:
        Mode du pipeline ("letter", "word", "sentences").
    processor:
        Instance Wav2Vec2Processor (déjà chargée par ModelLoader).
    augmenter:
        Instance AudioAugmenter (None = pas d'augmentation).
    val_split:
        Fraction de validation (0.20 recommandé sur petit corpus).
    seed:
        Graine pour la reproductibilité du split.
    """

    def __init__(
        self,
        json_path: str | Path,
        mode: str,
        processor,
        augmenter=None,
        val_split: float = 0.20,
        seed: int = 42,
    ) -> None:
        self._json_path = Path(json_path)
        self._audio_dir = self._json_path.parent
        self._mode = mode
        self._processor = processor
        self._augmenter = augmenter
        self._val_split = val_split
        self._seed = seed

    # ── API publique ───────────────────────────────────────────────────────

    def build(self) -> Tuple:
        """
        Construit et retourne (train_dataset, val_dataset, val_records).

        val_records est la liste de dicts utilisée pour générer le JSON
        de validation (chemin, label, type, speaker, session, micro).

        Returns
        -------
        tuple[Dataset, Dataset, list[dict]]
        """
        from datasets import Dataset

        raw_rows = self._load_json()
        filtered = self._filter_by_mode(raw_rows)
        if not filtered:
            raise ValueError(
                f"Aucune ligne pour le mode '{self._mode}' dans {self._json_path}"
            )
        logger.info("Lignes après filtrage mode='%s' : %d", self._mode, len(filtered))

        processed, skipped = self._preprocess_all(filtered)
        if not processed:
            raise ValueError("Aucun fichier audio prétraité avec succès.")
        logger.info(
            "Prétraitement : %d ok, %d ignorés.", len(processed), skipped
        )

        dataset = Dataset.from_list(processed)
        train_ds, val_ds = self._stratified_split(dataset, processed)

        logger.info(
            "Split — train : %d | val : %d", len(train_ds), len(val_ds)
        )

        val_records = [
            {
                "file": str(r["file"]),
                "label": r["transcript"],
                "type": r["type"],
                "speaker": r["speaker"],
                "session": r["session"],
                "micro": r["micro"],
            }
            for r in val_ds.to_list()
        ]

        return train_ds, val_ds, val_records

    # ── Privé ──────────────────────────────────────────────────────────────

    def _load_json(self) -> List[Dict]:
        with open(self._json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _filter_by_mode(self, rows: List[Dict]) -> List[Dict]:
        allowed = MODE_TO_TYPES[self._mode]
        return [r for r in rows if str(r.get("type", "")) in allowed]

    def _resolve_audio_path(self, audio_file: str) -> Path:
        p = Path(audio_file)
        return p if p.is_absolute() else self._audio_dir / audio_file

    def _normalize_transcript(self, label: object, label_type: object) -> str:
        """Normalise le transcript pour le stockage dans le dataset."""
        text = str(label).strip().lower()
        # Cas spécial : caractère espace pour les labels spéciaux vides
        if str(label_type) == "Special" and not text:
            return self._processor.tokenizer.word_delimiter_token or "|"
        return text

    def _preprocess_one(self, row: Dict, apply_augment: bool = True) -> Optional[Dict]:
        """Prétraite un enregistrement. Retourne None si échec."""
        import librosa

        audio_path = self._resolve_audio_path(str(row["file"]))
        if not audio_path.exists():
            logger.warning("Fichier manquant : %s", audio_path)
            return None

        try:
            waveform, _ = librosa.load(str(audio_path), sr=16_000, mono=True)
            waveform = waveform.astype(np.float32)

            # Augmentation (train seulement)
            if apply_augment and self._augmenter is not None:
                waveform = self._augmenter.augment(waveform)

            input_values = self._processor(
                waveform, sampling_rate=16_000
            ).input_values[0]

            transcript = self._normalize_transcript(row["label"], row.get("type", ""))

            with self._processor.as_target_processor():
                labels = self._processor(transcript).input_ids

            if not labels:
                labels = self._processor.tokenizer(
                    self._processor.tokenizer.word_delimiter_token
                ).input_ids

            return {
                "input_values": input_values,
                "labels": labels,
                "transcript": transcript,
                "file": str(audio_path),
                "type": str(row.get("type", "")),
                "session": str(row.get("session", "")),
                "micro": str(row.get("micro", "")),
                "speaker": str(row.get("speaker", "")),
            }
        except Exception as exc:
            logger.warning("Erreur prétraitement '%s' : %s", audio_path, exc)
            return None

    def _preprocess_all(self, rows: List[Dict]) -> Tuple[List[Dict], int]:
        processed, skipped = [], 0
        for row in rows:
            result = self._preprocess_one(row, apply_augment=True)
            if result is not None:
                processed.append(result)
            else:
                skipped += 1
        return processed, skipped

    def _stratified_split(self, dataset, processed: List[Dict]):
        """
        Split stratifié par session si plusieurs sessions existent.
        Sinon, split aléatoire simple.
        """
        sessions = [r["session"] for r in processed]
        unique_sessions = list(set(sessions))

        if len(unique_sessions) > 1:
            logger.info(
                "Split stratifié sur %d sessions : %s",
                len(unique_sessions),
                unique_sessions,
            )
            rng = np.random.default_rng(self._seed)
            train_idx, val_idx = [], []
            for sess in unique_sessions:
                s_idx = [i for i, s in enumerate(sessions) if s == sess]
                rng.shuffle(s_idx)
                n_val = max(1, int(len(s_idx) * self._val_split))
                val_idx.extend(s_idx[:n_val])
                train_idx.extend(s_idx[n_val:])
            return dataset.select(train_idx), dataset.select(val_idx)
        else:
            split = dataset.train_test_split(
                test_size=self._val_split, seed=self._seed, shuffle=True
            )
            return split["train"], split["test"]
