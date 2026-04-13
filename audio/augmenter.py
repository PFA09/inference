"""
audio/augmenter.py — Augmentation audio pour l'entraînement uniquement.

Responsabilité : appliquer des transformations stochastiques à une waveform
numpy pour améliorer la généralisation sur un petit corpus.

Ce module n'est JAMAIS appelé pendant l'inférence.
"""

from __future__ import annotations

import numpy as np

from config import AugmentConfig


class AudioAugmenter:
    """
    Augmenteur audio stochastique.

    Chaque transformation est appliquée indépendamment avec sa propre
    probabilité. L'ordre est fixe : shift → bruit → pitch, ce qui
    préserve le timing perçu avant d'ajouter des artefacts.

    Parameters
    ----------
    config:
        Paramètres d'augmentation (probabilités, amplitudes).
    sample_rate:
        Fréquence d'échantillonnage de la waveform d'entrée (Hz).
    seed:
        Graine aléatoire pour la reproductibilité (None = aléatoire).
    """

    def __init__(
        self,
        config: AugmentConfig,
        sample_rate: int = 16_000,
        seed: int | None = None,
    ) -> None:
        self._config = config
        self._sr = sample_rate
        self._rng = np.random.default_rng(seed)
        config.validate()

    # ── API publique ───────────────────────────────────────────────────────

    def augment(self, waveform: np.ndarray) -> np.ndarray:
        """
        Applique l'augmentation à une waveform.

        Si `config.enabled` est False, retourne la waveform inchangée.

        Parameters
        ----------
        waveform:
            Signal audio float32, shape (n_samples,).

        Returns
        -------
        np.ndarray
            Signal augmenté float32, même shape.
        """
        if not self._config.enabled:
            return waveform

        x = waveform.copy()
        x = self._apply_time_shift(x)
        x = self._apply_gaussian_noise(x)
        x = self._apply_pitch_shift(x)
        return x.astype(np.float32)

    # ── Transformations ────────────────────────────────────────────────────

    def _apply_time_shift(self, x: np.ndarray) -> np.ndarray:
        """Décalage circulaire du signal dans le temps."""
        if self._rng.random() > self._config.shift_prob:
            return x
        max_shift = int(len(x) * self._config.shift_max_fraction)
        if max_shift == 0:
            return x
        shift = int(self._rng.integers(-max_shift, max_shift + 1))
        return np.roll(x, shift)

    def _apply_gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        """
        Bruit blanc gaussien à SNR contrôlé.

        Le niveau de bruit est calculé à partir du SNR cible (dB) et
        de la puissance RMS du signal, ce qui préserve le rapport
        signal/bruit indépendamment de l'amplitude du signal.
        """
        if self._rng.random() > self._config.noise_prob:
            return x

        snr_db = self._rng.uniform(
            self._config.noise_min_snr_db,
            self._config.noise_max_snr_db,
        )
        signal_rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        noise_rms = signal_rms / (10 ** (snr_db / 20.0))
        noise = self._rng.normal(0.0, noise_rms, size=x.shape).astype(np.float32)
        return x + noise

    def _apply_pitch_shift(self, x: np.ndarray) -> np.ndarray:
        """
        Décalage de hauteur (pitch shift) via librosa.

        Conservé à ±1.5 demi-tons pour ne pas dénaturer les
        caractéristiques acoustiques de la parole dysarthrique.
        """
        if self._rng.random() > self._config.pitch_prob:
            return x
        try:
            import librosa

            n_steps = float(
                self._rng.uniform(
                    -self._config.pitch_semitones_range,
                    self._config.pitch_semitones_range,
                )
            )
            return librosa.effects.pitch_shift(
                x, sr=self._sr, n_steps=n_steps
            ).astype(np.float32)
        except Exception:
            # Dégradation gracieuse : si librosa échoue, on passe
            return x
