"""
inference.py — Point d'entrée CLI pour l'inférence fichier par fichier.

Exemples :
    python inference.py audio.wav --mode letter
    python inference.py audio.wav --mode letter --model-path ./training_outputs/final_merged_model
"""

from __future__ import annotations

import argparse
import sys

from config import ASRConfig, PathConfig
from pipeline.inference import SpeechInference
from model.loader import ModelLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR inference (Wav2Vec2)")
    parser.add_argument("audio_path", type=str, help="Chemin vers le fichier audio")
    parser.add_argument(
        "--mode", default="letter",
        choices=["letter", "word", "sentences"],
        help="Mode d'inférence (défaut: letter)",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Chemin vers un modèle fine-tuné (optionnel)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Config minimale pour l'inférence (paths.data_json non utilisé ici)
    # On construit manuellement pour éviter de requérir un labels.json
    from config import AudioConfig, ModelConfig

    from audio.preprocessor import AudioPreprocessor
    from model.decoder import CTCDecoder
    from model.postprocessor import TextPostprocessor

    audio_config = AudioConfig()
    model_config = ModelConfig()
    loader = ModelLoader(model_config)
    processor, model = loader.load_for_inference(args.model_path)

    inference = SpeechInference.with_model(
        mode=args.mode,
        audio_config=audio_config,
        processor=processor,
        model=model,
        device=loader.device,
    )

    result = inference.predict(args.audio_path)
    print(f"Résultat : {result}")


if __name__ == "__main__":
    main()
