"""
data/collator.py — Collateur pour les batches CTC.

Padding dynamique : chaque batch est paddé au maximum de ses éléments,
pas à une longueur fixe globale. Les positions paddées des labels
sont remplacées par -100 pour que la CTC loss les ignore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union

import torch


@dataclass
class DataCollatorCTCWithPadding:
    """
    Collateur pour tâches CTC avec padding dynamique.

    Les inputs audio et les labels sont paddés séparément car
    ils ont des longueurs indépendantes.

    Parameters
    ----------
    processor:
        Instance Wav2Vec2Processor.
    padding:
        Stratégie de padding passée au processor.
    """

    processor: object  # Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        # Sépare inputs et labels avant le padding
        input_features = [
            {"input_values": f["input_values"]} for f in features
        ]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Padding des inputs audio
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Padding des labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Remplace les positions paddées par -100 (ignorées par CTC loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        # Ne garde que les clés attendues par Wav2Vec2ForCTC
        allowed = {"input_values", "attention_mask", "labels"}
        return {k: v for k, v in batch.items() if k in allowed}
