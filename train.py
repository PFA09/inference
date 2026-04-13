"""
train.py — Point d'entrée pour le fine-tuning.

Exemple d'utilisation :
    python train.py
"""

from __future__ import annotations

import logging

from config import ASRConfig, PathConfig
from pipeline.trainer import FineTuner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    cfg = ASRConfig(
        mode="letter",
        paths=PathConfig(
            data_json="/home/melissa/gdrive/Cours/ENSEIRB-MATMECA/2A/S8/PFA/data/labels.json",
            output_dir="/home/melissa/gdrive/Cours/ENSEIRB-MATMECA/2A/S8/PFA/code/inference/training_outputs",
        ),
        # Tous les hyperparamètres utilisent les valeurs EDA par défaut.
        # Pour les surcharger :
        # training=TrainingConfig(num_epochs=20, learning_rate=5e-5),
    )
    cfg.validate()

    fine_tuner = FineTuner(cfg)
    metrics = fine_tuner.train()
    print("Métriques finales :", metrics)


if __name__ == "__main__":
    main()
