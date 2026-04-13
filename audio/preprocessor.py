"""
audio/preprocessor.py — Prétraitement audio.

Responsabilité unique : charger un fichier audio et le convertir
en waveform numpy 1-D à 16 kHz, sans aucune logique métier.
"""

from __future__ import annotations

import numpy as np

from config import AudioConfig


class AudioPreprocessor:
    """
    Charge et normalise un fichier audio pour Wav2Vec2.

    Ce composant ne sait rien du modèle, du mode, ni de l'augmentation.
    Il retourne toujours un np.ndarray float32 mono à `config.sample_rate`.
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        config.validate()

    def load(self, audio_path: str) -> np.ndarray:
        """
        Charge un fichier audio et retourne une waveform normalisée.

        Parameters
        ----------
        audio_path:
            Chemin vers le fichier audio (.wav, .flac, .mp3, …).

        Returns
        -------
        np.ndarray
            Waveform float32, shape (n_samples,), à `config.sample_rate` Hz.

        Raises
        ------
        FileNotFoundError
            Si le fichier n'existe pas.
        RuntimeError
            Si librosa échoue à charger le fichier.
        """
        import os
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

        try:
            import librosa  # import local pour ne pas polluer si non installé

            waveform, _ = librosa.load(
                audio_path,
                sr=self._config.sample_rate,
                mono=self._config.mono,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Erreur lors du chargement de '{audio_path}' : {exc}"
            ) from exc

        waveform = waveform.astype(np.float32)
        self._validate_waveform(waveform, audio_path)
        return waveform

    # ── Privé ──────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_waveform(waveform: np.ndarray, source: str) -> None:
        """Vérifie que la waveform est exploitable."""
        if waveform.ndim != 1:
            raise RuntimeError(
                f"Waveform non-mono après chargement ({waveform.shape}) pour : {source}"
            )
        if waveform.size == 0:
            raise RuntimeError(f"Fichier audio vide : {source}")
        if not np.isfinite(waveform).all():
            raise RuntimeError(f"Waveform contient des NaN/Inf : {source}")
