"""
pipeline/evaluator.py — Évaluation du modèle sur le split de validation.

CORRECTIF MAJEUR : le modèle est chargé UNE SEULE FOIS et injecté dans
SpeechInference via `with_model()`. L'ancienne version rechargeait le
modèle pour chaque fichier audio — O(N) I/O disque inutiles.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer
from sklearn.metrics import accuracy_score

from config import ASRConfig
from model.loader import ModelLoader
from model.postprocessor import TextPostprocessor
from pipeline.inference import SpeechInference

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Évalue le modèle sur le split de validation et agrège les métriques.

    Parameters
    ----------
    config:
        Configuration complète du pipeline (pour mode, audio, model).
    model_path:
        Chemin vers le modèle fine-tuné fusionné. Si None, utilise le
        modèle de base.
    validation_json:
        Chemin vers le JSON de validation. Si None, utilise
        config.paths.validation_json.
    """

    def __init__(
        self,
        config: ASRConfig,
        model_path: Optional[str] = None,
        validation_json: Optional[str] = None,
    ) -> None:
        self._config = config
        self._model_path = model_path
        self._val_json = Path(
            validation_json or config.paths.validation_json
        )
        self._postprocessor = TextPostprocessor(config.mode)

        self.results_df: Optional[pd.DataFrame] = None
        self.global_metrics: Dict[str, float] = {}

    # ── API publique ───────────────────────────────────────────────────────

    def run(self, show_progress: bool = True) -> Dict[str, float]:
        """
        Lance l'évaluation complète.

        Returns
        -------
        dict
            {"wer": float, "cer": float, "accuracy": float, "samples": int}
        """
        from tqdm.auto import tqdm

        val_records = self._load_val_json()
        logger.info("Évaluation sur %d fichiers…", len(val_records))

        # ── Chargement du modèle UNE SEULE FOIS ──────────────────────────
        loader = ModelLoader(self._config.model)
        processor, model = loader.load_for_inference(self._model_path)

        # Injection du modèle dans SpeechInference
        inference = SpeechInference.with_model(
            mode=self._config.mode,
            audio_config=self._config.audio,
            processor=processor,
            model=model,
            device=loader.device,
        )

        rows: List[Dict] = []
        iterator = val_records
        if show_progress:
            iterator = tqdm(val_records, desc="Évaluation", unit="fichier")

        for record in iterator:
            audio_path = self._resolve_audio(str(record["file"]))
            truth = self._postprocessor.normalize_label(str(record["label"]))

            try:
                pred = inference.predict(audio_path)
                error_msg = ""
            except Exception as exc:
                pred = ""
                error_msg = str(exc)
                logger.warning("Échec inférence '%s' : %s", audio_path, exc)

            # Métriques par échantillon
            sample_cer = jiwer_cer(truth, pred) if truth else 1.0
            sample_wer = self._sample_wer(truth, pred)

            rows.append({
                "file": record["file"],
                "session": record.get("session", ""),
                "micro": record.get("micro", ""),
                "speaker": record.get("speaker", ""),
                "type": record.get("type", ""),
                "truth": truth,
                "prediction": pred,
                "cer": sample_cer,
                "wer": sample_wer,
                "is_correct": int(pred == truth),
                "error": error_msg,
            })

        self.results_df = pd.DataFrame(rows)
        self.global_metrics = self._aggregate_metrics()
        return self.global_metrics

    def export_pdf(self, output_path: str) -> None:
        """Délègue la génération PDF au ReportBuilder."""
        if self.results_df is None:
            raise RuntimeError("Appeler run() avant export_pdf().")
        from reporting.pdf_report import ReportBuilder

        builder = ReportBuilder(
            mode=self._config.mode,
            results_df=self.results_df,
            global_metrics=self.global_metrics,
        )
        builder.export(output_path)

    # ── Privé ──────────────────────────────────────────────────────────────

    def _load_val_json(self) -> List[Dict]:
        if not self._val_json.exists():
            raise FileNotFoundError(
                f"JSON de validation introuvable : {self._val_json}"
            )
        with open(self._val_json, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _resolve_audio(self, audio_file: str) -> str:
        p = Path(audio_file)
        if p.is_absolute():
            return str(p)
        return str(self._config.paths.audio_dir / audio_file)

    def _sample_wer(self, truth: str, pred: str) -> float:
        if not truth:
            return 1.0
        if self._config.mode == "letter":
            t = " ".join(list(truth)) if truth else "<empty>"
            p = " ".join(list(pred)) if pred else "<empty>"
            return float(jiwer_wer(t, p))
        return float(jiwer_wer(truth, pred))

    def _aggregate_metrics(self) -> Dict[str, float]:
        df = self.results_df
        refs = df["truth"].fillna("").tolist()
        hyps = df["prediction"].fillna("").tolist()

        if self._config.mode == "letter":
            refs_wer = [" ".join(list(r)) if r else "<empty>" for r in refs]
            hyps_wer = [" ".join(list(h)) if h else "<empty>" for h in hyps]
        else:
            refs_wer, hyps_wer = refs, hyps

        metrics: Dict[str, float] = {
            "wer": float(jiwer_wer(refs_wer, hyps_wer)),
            "cer": float(jiwer_cer(refs, hyps)),
            "samples": float(len(df)),
        }
        if self._config.mode == "letter":
            metrics["accuracy"] = float(accuracy_score(refs, hyps))
        return metrics
