"""
model/decoder.py — Décodage CTC brut.

Responsabilité unique : convertir des logits en texte brut,
en gérant correctement le token `|` (word delimiter de Wav2Vec2).

Ce module ne connaît pas le mode (lettre/mot/phrase) et ne fait
aucun mapping sémantique. Il retourne du texte brut normalisé.
"""

from __future__ import annotations

import re

import torch


# Tokens spéciaux que Wav2Vec2Processor peut laisser passer
# même avec skip_special_tokens=True
_PIPE_RE = re.compile(r"\|+")
_WHITESPACE_RE = re.compile(r"\s+")


class CTCDecoder:
    """
    Décode les logits d'un modèle Wav2Vec2ForCTC en texte brut.

    Le texte retourné est :
    - en minuscules
    - sans token `|` (word delimiter)
    - sans espaces multiples ni espaces de début/fin

    Parameters
    ----------
    processor:
        Instance de `Wav2Vec2Processor` (chargée par `ModelLoader`).
    """

    def __init__(self, processor) -> None:  # type: ignore[annotation]
        self._processor = processor

    def decode(self, logits: torch.Tensor) -> str:
        """
        Décode un tenseur de logits en texte brut.

        Parameters
        ----------
        logits:
            Shape (1, T, vocab_size) ou (T, vocab_size).

        Returns
        -------
        str
            Texte décodé, nettoyé, en minuscules.
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        predicted_ids = torch.argmax(logits, dim=-1)
        raw: str = self._processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        return self._clean(raw)

    # ── Privé ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """
        Nettoie le texte brut issu de `batch_decode`.

        Étapes dans l'ordre :
        1. Supprime les tokens `|` (word delimiter Wav2Vec2).
        2. Supprime les tokens spéciaux résiduels <...>.
        3. Normalise les espaces multiples.
        4. Strip et minuscules.
        """
        # 1. Supprime les pipes (word delimiter de Wav2Vec2)
        text = _PIPE_RE.sub(" ", text)

        # 2. Supprime les tokens spéciaux résiduels ex: <unk>, <pad>
        text = re.sub(r"<[^>]+>", "", text)

        # 3. Normalise les espaces
        text = _WHITESPACE_RE.sub(" ", text)

        # 4. Strip et minuscules
        return text.strip().lower()
