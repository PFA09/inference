import sys
import os
import argparse

_devnull = open(os.devnull, 'w')
_old_stderr = sys.stderr
sys.stderr = _devnull

import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import Optional, Dict

sys.stderr = _old_stderr

class SpeechInference:
    """
    Classe pour l'inférence ASR (Automatic Speech Recognition) basée sur Wav2Vec2.
    Supporte plusieurs modes : 'word', 'letter', 'sentences'.
    """

    # Paramètres par défaut des modèles
    # Modèles XLSR-53 français fine-tuné par Jonatas Grosman
    DEFAULT_MODELS = {
        "word": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        "letter": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        "sentences": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    }

    # Mapping pour le mode 'letter'
    LETTER_MAPPING: Dict[str, str] = {
        # Chiffres (texte -> symbole)
        "zéro": "0",
        "zero": "0",
        "un": "1",
        "deux": "2",
        "trois": "3",
        "quatre": "4",
        "cinq": "5",
        "six": "6",
        "sept": "7",
        "huit": "8",
        "neuf": "9",
        # Caractères spéciaux
        "tiret": "-",
        "trait": "-",
        "point": ".",
        "virgule": ",",
        "espace": " ",
        "slash": "/",
        "barre": "/",
        "parenthèse ouvrante": "(",
        "parenthèse fermante": ")",
        "arobase": "@",
        "hash": "#",
        "dièse": "#",
        "cedille": "ç",
        "cédille": "ç",
        # Variantes de prononciation pour les lettres (épellation)
        "bé": "b",
        "bee": "b",
        "cé": "c",
        "cec": "c",
        "dé": "d",
        "dee": "d",
        "è": "e",
        "eff": "f",
        "ef": "f",
        "gé": "g",
        "ge": "g",
        "ache": "h",
        "hache": "h",
        "i": "i",
        "ji": "j",
        "jé": "j",
        "ka": "k",
        "ké": "k",
        "elle": "l",
        "èle": "l",
        "em": "m",
        "ème": "m",
        "en": "n",
        "ène": "n",
        "o": "o",
        "eau": "o",
        "oh": "o",
        "pé": "p",
        "pe": "p",
        "qu": "q",
        "queue": "q",
        "qué": "q",
        "erre": "r",
        "ère": "r",
        "ès": "s",
        "esse": "s",
        "es": "s",
        "té": "t",
        "te": "t",
        "u": "u",
        "vu": "v",
        "ve": "v",
        "double v": "w",
        "w": "w",
        "iks": "x",
        "x": "x",
        "i grec": "y",
        "igrec": "y",
        "y grec": "y",
        "ygrec": "y",
        "zed": "z",
        "zède": "z",
        "zé": "z",
    }

    def __init__(
        self,
        audio_path: str,
        mode: str,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Initialisation de la classe SpeechInference.

        Args:
            audio_path: Chemin vers le fichier audio
            mode: Mode d'inférence ('word', 'letter', ou 'sentences')
            model_path: Chemin local vers le modèle (optionnel)
            model_name: Nom du modèle HuggingFace (optionnel)
        """
        if mode not in ["word", "letter", "sentences"]:
            raise ValueError(
                f"Mode '{mode}' invalide. Doit être 'word', 'letter', ou 'sentences'."
            )

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio introuvable: {audio_path}")

        self.audio_path: str = audio_path
        self.mode: str = mode
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path:
            self.model_path: str = model_path
            self.model_name: Optional[str] = None
        else:
            self.model_name: str = model_name or self.DEFAULT_MODELS[mode]
            self.model_path = None

        self.processor: Optional[Wav2Vec2Processor] = None
        self.model: Optional[Wav2Vec2ForCTC] = None
        self._audio_waveform: Optional[torch.Tensor] = None

    def _preprocess_audio(self) -> torch.Tensor:
        """
        Prétraitement de l'audio : chargement, rééchantillonnage à 16kHz et conversion en mono.

        Returns:
            Waveform audio (torch.Tensor) en mono à 16kHz
        """
        try:
            waveform, _ = librosa.load(
                self.audio_path,
                sr=16000,
                mono=True
            )

            waveform_tensor = torch.tensor(waveform, dtype=torch.float32)

            return waveform_tensor

        except Exception as e:
            raise RuntimeError(f"Erreur lors du prétraitement audio: {str(e)}")

    def _post_process_letter(self, text: str) -> str:
        """
        Post-traitement pour le mode 'letter':
        - Si le token est dans le mapping -> utilise la valeur mappée
        - Sinon, prend la première lettre du token (dans les cas Non-mapping)

        Args:
            text: Texte brut issu de l'inférence

        Returns:
            Texte post-traité avec symboles
        """
        token = str(text).strip().lower()

        if not token:
            return ""

        # Filter tokenizer special tokens (e.g. <unk>, <s>, </s>, <pad>)
        # so they never leak as '<' in letter mode outputs.
        if token in {"<unk>", "<s>", "</s>", "<pad>", "[pad]"}:
            return ""
        if token.startswith("<") and token.endswith(">"):
            return ""

        if token in self.LETTER_MAPPING:
            return self.LETTER_MAPPING[token]

        return token[0]

    def _load_models(self) -> None:
        """
        Charge le processeur et le modèle Wav2Vec2ForCTC depuis HuggingFace.
        """
        if self.processor is not None and self.model is not None:
            return  # Déjà chargés

        try:            
            if self.model_path:
                self.processor = Wav2Vec2Processor.from_pretrained(
                    self.model_path
                )
                self.model = Wav2Vec2ForCTC.from_pretrained(self.model_path)
            else:
                self.processor = Wav2Vec2Processor.from_pretrained(
                    self.model_name
                )
                self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)

            self.model = self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {str(e)}")

    def predict(self) -> str:
        """
        Effectue l'inférence sur l'audio fourni.

        Process:
            1. Prétraitement de l'audio
            2. Chargement des modèles
            3. Extraction des features avec le processeur
            4. Inférence avec torch.no_grad()
            5. Post-traitement selon le mode

        Returns:
            Texte reconnu (lettres/mots/phrases selon le mode)
        """
        try:
            # Prétraitement de l'audio
            if self._audio_waveform is None:
                self._audio_waveform = self._preprocess_audio()

            # Chargement des modèles
            self._load_models()
            
            # Extraction des features avec le processeur
            input_values = self.processor(
                self._audio_waveform.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).input_values

            # Placement des inputs sur le device
            input_values = input_values.to(self.device)

            # Inférence sans calcul de gradient
            with torch.no_grad():
                logits = self.model(input_values).logits

            # Récupération des prédictions
            predicted_ids = torch.argmax(logits, dim=-1)

            # Décodage du texte
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # Post-traitement selon le mode
            if self.mode == "letter":
                result = self._post_process_letter(transcription)
            else:
                result = transcription

            return result

        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'inférence: {str(e)}")



# ---------------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASR inference on a single audio file (Wav2Vec2)."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Chemin vers le fichier audio à tester (.wav).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="letter",
        choices=["word", "letter", "sentences"],
        help="Mode d'inférence (défaut: letter).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Chemin local vers un modèle fine-tuné (optionnel).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Nom HuggingFace du modèle (optionnel).",
    )

    args = parser.parse_args()
    if args.model_path and args.model_name:
        parser.error("Choisir soit --model-path soit --model-name, pas les deux.")
    return args


def main() -> None:
    args = _parse_args()

    inference = SpeechInference(
        audio_path=args.audio_path,
        mode=args.mode,
        model_path=args.model_path,
        model_name=args.model_name,
    )
    result = inference.predict()

    print(f"Résultat de l'inférence: {result}")


if __name__ == "__main__":
    main()