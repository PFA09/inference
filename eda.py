"""
eda.py — Point d'entrée CLI pour l'analyse exploratoire du corpus.

Exemple :
    python eda.py
    python eda.py --output reports/eda.pdf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA du corpus ASR dysarthrie")
    parser.add_argument(
        "--output", type=str,
        default=None,
        help="Chemin du PDF généré (défaut : <output_dir>/reports/eda.pdf)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from config import ASRConfig, PathConfig
    from data.eda import DataAnalyzer

    cfg = ASRConfig(
        mode="letter",
        paths=PathConfig(
            data_json="/home/melissa/gdrive/Cours/ENSEIRB-MATMECA/2A/S8/PFA/data/labels.json",
            output_dir="/home/melissa/gdrive/Cours/ENSEIRB-MATMECA/2A/S8/PFA/code/inference/viz_outputs",
        ),
    )

    output_path = args.output or str(cfg.paths.reports_dir / "eda.pdf")

    analyzer = DataAnalyzer(cfg)
    analyzer.analyze()
    analyzer.export_pdf(output_path)
    print(f"Rapport EDA généré : {output_path}")


if __name__ == "__main__":
    main()
