"""
pipeline/inference.py — Façade publique pour l'inférence 

C'est le seul point d'entrée pour les utilisateurs du pipeline d'inférence.
Il orchestre : préprocesseur → modèle → décodeur → postprocesseur.

Le modèle peut être injecté (pour l'évaluation batch) ou chargé lazily
(pour l'inférence fichier par fichier).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import ASRConfig, AudioConfig, Mode
from audio.preprocessor import AudioPreprocessor
from model.decoder import CTCDecoder
from model.loader import ModelLoader
from model.postprocessor import TextPostprocessor

logger = logging.getLogger(__name__)


class SpeechInference:
    """
    Pipeline d'inférence ASR complet.

    Peut être utilisé de deux façons :

    1. Standalone (charge le modèle au premier appel) :
       >>> inf = SpeechInference.from_config(config)
       >>> result = inf.predict("audio.wav")

    2. Avec modèle injecté (pour l'évaluation batch — évite N rechargements) :
       >>> processor, model = ModelLoader(config.model).load_for_inference(path)
       >>> inf = SpeechInference.with_model(mode, processor, model)
       >>> result = inf.predict("audio.wav")
    """

    def __init__(
        self,
        mode: Mode,
        audio_config: AudioConfig,
        preprocessor: AudioPreprocessor,
        decoder: CTCDecoder,
        postprocessor: TextPostprocessor,
        model,
        device: str,
    ) -> None:
        self._mode = mode
        self._audio_config = audio_config
        self._preprocessor = preprocessor
        self._decoder = decoder
        self._postprocessor = postprocessor
        self._model = model
        self._device = device

    # ── Constructeurs alternatifs ──────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: ASRConfig,
        model_path: Optional[str | Path] = None,
    ) -> "SpeechInference":
        """
        Crée une instance en chargeant le modèle depuis le disque.

        Parameters
        ----------
        config:
            Configuration complète du pipeline.
        model_path:
            Chemin vers un modèle fine-tuné. Si None, utilise le modèle
            de base HuggingFace défini dans config.model.
        """
        loader = ModelLoader(config.model)
        processor, model = loader.load_for_inference(model_path)
        return cls.with_model(
            mode=config.mode,
            audio_config=config.audio,
            processor=processor,
            model=model,
            device=loader.device,
        )

    @classmethod
    def with_model(
        cls,
        mode: Mode,
        audio_config: AudioConfig,
        processor,
        model,
        device: str,
    ) -> "SpeechInference":
        """
        Crée une instance avec un modèle déjà chargé (injection de dépendances).

        Utilisé par ModelEvaluator pour éviter N rechargements.
        """
        preprocessor = AudioPreprocessor(audio_config)
        decoder = CTCDecoder(processor)
        postprocessor = TextPostprocessor(mode)

        return cls(
            mode=mode,
            audio_config=audio_config,
            preprocessor=preprocessor,
            decoder=decoder,
            postprocessor=postprocessor,
            model=model,
            device=device,
        )

    # ── API publique ───────────────────────────────────────────────────────

    def predict(self, audio_path: str) -> str:
        """
        Prédit le texte à partir d'un fichier audio.

        Parameters
        ----------
        audio_path:
            Chemin vers le fichier audio.

        Returns
        -------
        str
            Texte reconnu, post-traité selon le mode.
        """
        # 1. Prétraitement
        waveform = self._preprocessor.load(audio_path)

        # 2. Feature extraction
        from transformers import Wav2Vec2Processor
        input_values = self._decoder._processor(
            waveform,
            sampling_rate=self._audio_config.sample_rate,
            return_tensors="pt",
            padding=True,
        ).input_values.to(self._device)

        # 3. Inférence
        with torch.no_grad():
            logits = self._model(input_values).logits

        # 4. Décodage CTC → texte brut
        raw_text = self._decoder.decode(logits)

        # 5. Post-traitement sémantique
        return self._postprocessor.process(raw_text)
