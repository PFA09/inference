"""
model/loader.py — Chargement et gestion du cycle de vie du modèle.

Responsabilités :
- Charger le processor et le modèle Wav2Vec2 (base ou fine-tuné)
- Injecter les adapteurs LoRA pour l'entraînement
- Fusionner les adapteurs LoRA pour l'inférence (merge_and_unload)
- Placer le modèle sur le bon device
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from config import ModelConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Gère le chargement et la préparation du modèle Wav2Vec2.

    Deux cas d'usage :
    1. Inférence : `load_for_inference(model_path)` — modèle fusionné, eval mode.
    2. Entraînement : `load_for_training()` — modèle de base + LoRA injecté.

    Parameters
    ----------
    config:
        Configuration modèle (nom HF, SpecAugment, LoRA…).
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Device sélectionné : %s", self._device)

    @property
    def device(self) -> str:
        return self._device

    # ── Inférence ──────────────────────────────────────────────────────────

    def load_for_inference(self, model_path: Optional[str | Path] = None):
        """
        Charge processor + modèle pour l'inférence.

        Parameters
        ----------
        model_path:
            Chemin vers un modèle fine-tuné local. Si None, utilise
            le modèle de base HuggingFace.

        Returns
        -------
        tuple[Wav2Vec2Processor, Wav2Vec2ForCTC]
        """
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        source = str(model_path) if model_path else self._config.base_model_name
        logger.info("Chargement du modèle pour inférence depuis : %s", source)

        processor = Wav2Vec2Processor.from_pretrained(source)
        model = Wav2Vec2ForCTC.from_pretrained(source)
        model = model.to(self._device)
        model.eval()

        return processor, model

    # ── Entraînement ───────────────────────────────────────────────────────

    def load_for_training(self):
        """
        Charge le modèle de base avec SpecAugment + LoRA pour l'entraînement.

        Returns
        -------
        tuple[Wav2Vec2Processor, Wav2Vec2ForCTC (avec LoRA)]
        """
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model
        from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor

        logger.info(
            "Chargement du modèle de base pour entraînement : %s",
            self._config.base_model_name,
        )

        processor = Wav2Vec2Processor.from_pretrained(self._config.base_model_name)

        # ── Wav2Vec2Config avec SpecAugment et dropout ──────────────────
        # CORRECTIF : les paramètres de masquage doivent passer par la Config,
        # pas par from_pretrained() qui les ignore silencieusement.
        base_config = Wav2Vec2Config.from_pretrained(self._config.base_model_name)

        base_config.mask_time_prob = self._config.mask_time_prob
        base_config.mask_time_length = self._config.mask_time_length
        base_config.mask_feature_prob = self._config.mask_feature_prob
        base_config.mask_feature_length = self._config.mask_feature_length
        base_config.activation_dropout = self._config.activation_dropout
        base_config.hidden_dropout = self._config.hidden_dropout
        base_config.feat_proj_dropout = self._config.feat_proj_dropout
        base_config.attention_dropout = self._config.attention_dropout
        base_config.final_dropout = self._config.final_dropout

        model = Wav2Vec2ForCTC.from_pretrained(
            self._config.base_model_name,
            config=base_config,
            ignore_mismatched_sizes=True,
        )

        # Gel du feature extractor CNN (couches bas-niveau non utiles à fine-tuner)
        model.freeze_feature_encoder()

        # ── LoRA injection ───────────────────────────────────────────────
        lora_cfg = self._config.lora
        peft_config = PeftLoraConfig(
            # CORRECTIF : SEQ_2_SEQ_LM n'est pas adapté à CTC.
            # FEATURE_EXTRACTION est le seul TaskType qui ne modifie pas
            # le forward pass et laisse notre wrapper fonctionner correctement.
            # L'alternative propre est d'utiliser un custom forward.
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            bias=lora_cfg.bias,
        )

        peft_model = get_peft_model(model, peft_config)
        peft_model.print_trainable_parameters()

        return processor, peft_model

    # ── Fusion des adapteurs ───────────────────────────────────────────────

    @staticmethod
    def merge_and_save(peft_model, save_dir: str | Path, processor) -> Path:
        """
        Fusionne les adapteurs LoRA dans le modèle de base et sauvegarde.

        Parameters
        ----------
        peft_model:
            Modèle PEFT après entraînement.
        save_dir:
            Dossier de destination.
        processor:
            Processor à sauvegarder avec le modèle.

        Returns
        -------
        Path
            Chemin du dossier où le modèle fusionné est sauvegardé.
        """
        from peft import PeftModel

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info("Fusion des adapteurs LoRA → %s", save_path)

        if isinstance(peft_model, PeftModel):
            merged = peft_model.merge_and_unload()
        else:
            merged = peft_model

        merged.save_pretrained(str(save_path))
        processor.save_pretrained(str(save_path))

        logger.info("Modèle fusionné sauvegardé dans : %s", save_path)
        return save_path
