"""
model/postprocessor.py — Post-traitement sémantique du texte décodé.

Responsabilité unique : convertir le texte brut (sorti du CTCDecoder)
vers la représentation finale selon le mode.

C'est L'UNIQUE endroit dans le projet qui connaît le mapping
français → symbole. Il remplace toutes les versions éparpillées dans
l'ancien code (_post_process_letter, _normalize_letter_sequence_from_inference,
_normalize_text).

Architecture du mode lettre
---------------------------
Un enregistrement = UNE seule lettre/chiffre/symbole prononcé.
Le modèle peut retourner une transcription multi-mots (ex: "i grec", "double v"),
mais elle correspond toujours à UN seul symbole cible.

On cherche donc la correspondance la PLUS LONGUE dans le mapping
(longest-match) pour gérer "i grec" > "i" et "double v" > "v".
Si aucun match, on prend la première lettre de la transcription.
"""

from __future__ import annotations

from config import Mode


# ─── Mapping français → symbole ───────────────────────────────────────────────
#
# Clés en minuscules (le CTCDecoder garantit des sorties en minuscules).
# Ordonnées du plus long au plus court pour le longest-match.

_LETTER_MAPPING: dict[str, str] = {
    # Multi-mots en premier (longest match)
    "parenthèse ouvrante": "(",
    "parenthèse fermante": ")",
    "double v": "w",
    "i grec": "y",
    "y grec": "y",
    "i grecque": "y",
    # Chiffres
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
    "arobase": "@",
    "hash": "#",
    "dièse": "#",
    "cedille": "ç",
    "cédille": "ç",
    # Lettres — variantes de prononciation françaises
    "bé": "b",
    "bee": "b",
    "cé": "c",
    "dé": "d",
    "dee": "d",
    "eff": "f",
    "ef": "f",
    "gé": "g",
    "ge": "g",
    "ache": "h",
    "hache": "h",
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
    "vu": "v",
    "ve": "v",
    "igrec": "y",
    "ygrec": "y",
    "zed": "z",
    "zède": "z",
    "zé": "z",
    # Lettres isolées (mono-caractère) — en dernier car shortest
    "a": "a",
    "b": "b",
    "c": "c",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "g",
    "h": "h",
    "i": "i",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "p": "p",
    "q": "q",
    "r": "r",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "x": "x",
    "y": "y",
    "z": "z",
}

# Pré-calculé une fois : liste triée du plus long au plus court
_SORTED_KEYS: list[str] = sorted(_LETTER_MAPPING, key=len, reverse=True)


class TextPostprocessor:
    """
    Post-traite le texte brut selon le mode du pipeline.

    Parameters
    ----------
    mode:
        "letter" | "word" | "sentences"
    """

    def __init__(self, mode: Mode) -> None:
        self._mode = mode

    def process(self, raw_text: str) -> str:
        """
        Convertit le texte brut décodé vers la représentation finale.

        Parameters
        ----------
        raw_text:
            Texte retourné par `CTCDecoder.decode()` — minuscules, sans pipe.

        Returns
        -------
        str
            Représentation finale selon le mode.
        """
        if self._mode == "letter":
            return self._process_letter(raw_text)
        # Modes word et sentences : on retourne le texte tel quel
        # (les majuscules sont laissées au soin de l'appelant si besoin)
        return raw_text

    # ── Mode lettre ────────────────────────────────────────────────────────

    @staticmethod
    def _process_letter(text: str) -> str:
        """
        Mappe une transcription mono-lettre vers son symbole cible.

        Algorithme longest-match :
        On parcourt les clés du plus long au plus court et on retourne
        la première qui est trouvée comme sous-chaîne exacte du texte.

        Si aucune clé ne matche, on prend la première lettre de `text`.
        Si `text` est vide, on retourne "".

        Note : un enregistrement = une seule lettre.
        On ne fait donc PAS de split multi-tokens ici.
        """
        text = text.strip()
        if not text:
            return ""

        # Longest-match : parcours des clés triées du plus long au plus court
        for key in _SORTED_KEYS:
            if key in text:
                return _LETTER_MAPPING[key]

        # Fallback : première lettre alphanumérique du texte brut
        for char in text:
            if char.isalnum() or char in "-.,/@#()ç":
                return char

        return ""

    # ── Utilitaires pour l'évaluation ─────────────────────────────────────

    def normalize_label(self, label: str) -> str:
        """
        Normalise un label de vérité terrain pour la comparaison.

        En mode lettre, on s'assure que le label est un symbole unique
        en minuscule. En mode word/sentences, on normalise les espaces.
        """
        if self._mode == "letter":
            label = label.strip().lower()
            return label[0] if label else ""
        return " ".join(label.strip().lower().split())
