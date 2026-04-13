"""
pipeline/trainer.py — Fine-tuning Wav2Vec2 avec LoRA.

Orchestre : DatasetBuilder → ModelLoader → Trainer HuggingFace → merge.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from config import ASRConfig
from audio.augmenter import AudioAugmenter
from data.collator import DataCollatorCTCWithPadding
from data.dataset import DatasetBuilder
from model.loader import ModelLoader
from model.postprocessor import TextPostprocessor

logger = logging.getLogger(__name__)


class _Wav2Vec2CTCWrapper(torch.nn.Module):
    """
    Wrapper minimal autour du modèle PEFT pour le Trainer HuggingFace.

    Problème : PEFT injecte des kwargs non reconnus par Wav2Vec2ForCTC
    dans le forward pass. Ce wrapper les filtre proprement.
    """

    def __init__(self, peft_model: torch.nn.Module) -> None:
        super().__init__()
        self.peft_model = peft_model

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.peft_model, name)

    def gradient_checkpointing_enable(self, **kwargs):
        return self.peft_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        return self.peft_model.gradient_checkpointing_disable()

    def save_pretrained(self, *args, **kwargs):
        return self.peft_model.save_pretrained(*args, **kwargs)

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # On appelle base_model.model (Wav2Vec2ForCTC avec couches LoRA déjà
        # injectées) en contournant le forward() de PeftModelForFeatureExtraction
        # qui ajoute input_ids=None — argument inconnu de Wav2Vec2ForCTC.
        # Les gradients circulent bien via les couches LoRA car elles sont
        # physiquement dans le graphe de base_model.model.
        model = self.peft_model.base_model.model
        return model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
        )


class FineTuner:
    """
    Orchestre l'entraînement complet.

    Parameters
    ----------
    config:
        Configuration complète du pipeline.
    """

    def __init__(self, config: ASRConfig) -> None:
        config.validate()
        self._config = config
        self._postprocessor = TextPostprocessor(config.mode)

    # ── API publique ───────────────────────────────────────────────────────

    def train(self) -> Dict[str, float]:
        """
        Lance l'entraînement complet.

        Returns
        -------
        dict
            Métriques de validation finales (wer, cer, accuracy).
        """
        cfg = self._config
        paths = cfg.paths
        paths.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Chargement du modèle de base + LoRA
        loader = ModelLoader(cfg.model)
        processor, peft_model = loader.load_for_training()

        # 2. Augmentation
        augmenter = AudioAugmenter(
            config=cfg.augment,
            sample_rate=cfg.audio.sample_rate,
            seed=cfg.training.random_seed,
        )

        # 3. Construction des datasets
        builder = DatasetBuilder(
            json_path=paths.data_json,
            mode=cfg.mode,
            processor=processor,
            augmenter=augmenter,
            val_split=cfg.training.val_split,
            seed=cfg.training.random_seed,
        )
        train_ds, val_ds, val_records = builder.build()

        # Sauvegarde du split de validation
        paths.validation_json.parent.mkdir(parents=True, exist_ok=True)
        with open(paths.validation_json, "w", encoding="utf-8") as fh:
            json.dump(val_records, fh, ensure_ascii=False, indent=2)
        logger.info("Split validation sauvegardé : %s", paths.validation_json)

        # 4. Wrapper + collateur
        train_model = _Wav2Vec2CTCWrapper(peft_model)
        collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

        # 5. Arguments d'entraînement
        t = cfg.training
        training_args = TrainingArguments(
            output_dir=str(paths.output_dir),
            per_device_train_batch_size=t.per_device_train_batch_size,
            per_device_eval_batch_size=t.per_device_eval_batch_size,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            num_train_epochs=t.num_epochs,
            learning_rate=t.learning_rate,
            weight_decay=t.weight_decay,
            warmup_ratio=t.warmup_ratio,
            logging_strategy="steps",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=t.eval_steps,
            save_strategy="steps",
            save_steps=t.save_steps,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=False,  # désactivé : incompatible avec SpecAugment in-place + hook LoRA sur Wav2Vec2
            save_total_limit=t.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model=t.metric_for_best_model,
            greater_is_better=t.greater_is_better,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )

        # 6. compute_metrics (closure sur processor + postprocessor)
        def compute_metrics(pred) -> Dict[str, float]:
            return _compute_metrics(pred, processor, self._postprocessor, cfg.mode)

        # 7. Trainer
        trainer = Trainer(
            model=train_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=processor.feature_extractor,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=t.early_stopping_patience
                )
            ],
        )

        logger.info("Démarrage de l'entraînement…")
        trainer.train()

        # 8. Sauvegarde + fusion LoRA
        logger.info("Sauvegarde des artefacts…")
        trainer.save_model(str(paths.output_dir))
        processor.save_pretrained(str(paths.output_dir))

        merged_path = ModelLoader.merge_and_save(
            peft_model, paths.merged_model_dir, processor
        )
        logger.info("Modèle fusionné : %s", merged_path)

        # 9. Évaluation automatique
        from pipeline.evaluator import ModelEvaluator

        evaluator = ModelEvaluator(
            config=self._config,
            model_path=str(merged_path),
        )
        metrics = evaluator.run()

        # 10. Rapport PDF
        paths.reports_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = paths.reports_dir / f"evaluation_{cfg.mode}.pdf"
        evaluator.export_pdf(str(pdf_path))
        logger.info("Rapport PDF : %s", pdf_path)
        logger.info("Entraînement terminé. Métriques : %s", metrics)

        return metrics


# ── Fonction de métriques (top-level pour être picklable par le Trainer) ──────

def _compute_metrics(pred, processor, postprocessor: TextPostprocessor, mode: str) -> Dict[str, float]:
    """Calcule WER et CER pour le Trainer HuggingFace."""
    from sklearn.metrics import accuracy_score

    pred_ids = np.argmax(pred.predictions, axis=-1)
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, group_tokens=False, skip_special_tokens=True)

    # Post-traitement identique à l'inférence
    pred_str = [postprocessor.process(p) for p in pred_str]
    label_str = [postprocessor.normalize_label(l) for l in label_str]

    if mode == "letter":
        # WER calculé caractère par caractère (chaque char = un "mot")
        pred_for_wer = [" ".join(list(p)) if p else "<empty>" for p in pred_str]
        ref_for_wer = [" ".join(list(r)) if r else "<empty>" for r in label_str]
    else:
        pred_for_wer = pred_str
        ref_for_wer = label_str

    wer = float(jiwer_wer(ref_for_wer, pred_for_wer))
    cer = float(jiwer_cer(label_str, pred_str))

    metrics: Dict[str, float] = {"wer": wer, "cer": cer}
    if mode == "letter":
        metrics["accuracy"] = float(accuracy_score(label_str, pred_str))

    return metrics
