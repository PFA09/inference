"""Model evaluation utilities for ASR inference results.

This module evaluates SpeechInference predictions against labeled JSON data and
exports a professional PDF report with metrics, error analysis, and charts.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

from inference import SpeechInference

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ModelEvaluator:
    """Evaluate ASR predictions for a given mode and export results to PDF."""

    MODE_TO_TYPES: Dict[str, List[str]] = {
        "letter": ["Letter", "Digit", "Special"],
        "word": ["Word"],
        "sentences": ["Sentences"],
    }

    def __init__(self, json_path: str, model_path: Optional[str], mode: str) -> None:
        if mode not in {"letter", "word", "sentences"}:
            raise ValueError("mode must be one of: 'letter', 'word', 'sentences'")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        self.json_path: str = json_path
        self.model_path: Optional[str] = model_path
        self.mode: str = mode
        self.audio_dir: str = os.path.dirname(json_path)

        self.df_raw: pd.DataFrame = self._load_json()
        self.df_filtered: pd.DataFrame = self._filter_data()
        self.results_df: Optional[pd.DataFrame] = None
        self.global_metrics: Dict[str, float] = {}

    def _load_json(self) -> pd.DataFrame:
        with open(self.json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        df = pd.DataFrame(data)

        required_cols = {"file", "label", "type", "speaker", "session", "micro"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in JSON: {sorted(missing)}")

        return df

    def _filter_data(self) -> pd.DataFrame:
        """Filter JSON records depending on evaluation mode."""
        target_types = self.MODE_TO_TYPES[self.mode]
        filtered = self.df_raw[self.df_raw["type"].isin(target_types)].copy()
        filtered.reset_index(drop=True, inplace=True)

        if filtered.empty:
            raise ValueError(
                f"No data left after filtering for mode '{self.mode}' with types {target_types}."
            )

        return filtered

    def _resolve_audio_path(self, audio_file: str) -> str:
        if os.path.isabs(audio_file):
            return audio_file
        return os.path.join(self.audio_dir, audio_file)

    def _normalize_text(self, value: str) -> str:
        raw = str(value)

        if self.mode == "letter":
            # Keep explicit empty prediction as-is (model failure case).
            if raw == "":
                return ""
            # Preserve the special space symbol used in labels.
            if raw.isspace():
                return " "

            # In letter mode, force a single symbol prediction/label.
            compact = "".join(raw.upper().split())
            return compact[0] if compact else ""

        return " ".join(raw.strip().upper().split())

    def _prepare_for_wer(self, value: str) -> str:
        text = self._normalize_text(value)
        if self.mode == "letter":
            return " ".join(list(text))
        return text

    def run_evaluation(self, show_progress: bool = True) -> Dict[str, float]:
        """Run inference on filtered data and compute evaluation metrics.

        Args:
            show_progress: If True, display a progress bar over processed files.
        """
        rows: List[Dict[str, object]] = []

        iterator = self.df_filtered.iterrows()
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(self.df_filtered),
                desc="Evaluation en cours",
                unit="fichier",
            )

        for _, row in iterator:
            audio_path = self._resolve_audio_path(str(row["file"]))
            truth = self._normalize_text(str(row["label"]))

            try:
                infer = SpeechInference(
                    audio_path=audio_path,
                    mode=self.mode,
                    model_path=self.model_path,
                )
                pred = self._normalize_text(infer.predict())
                error_message = ""
            except Exception as exc:  # pragma: no cover
                pred = ""
                error_message = str(exc)

            sample_cer = jiwer_cer(truth, pred) if truth else 1.0
            sample_wer = jiwer_wer(self._prepare_for_wer(truth), self._prepare_for_wer(pred)) if truth else 1.0

            rows.append(
                {
                    "file": row["file"],
                    "session": row["session"],
                    "micro": row["micro"],
                    "speaker": row["speaker"],
                    "type": row["type"],
                    "truth": truth,
                    "prediction": pred,
                    "cer": sample_cer,
                    "wer": sample_wer,
                    "is_correct": int(pred == truth),
                    "error": error_message,
                }
            )

        self.results_df = pd.DataFrame(rows)

        refs = self.results_df["truth"].fillna("").tolist()
        hyps = self.results_df["prediction"].fillna("").tolist()

        self.global_metrics = {
            "wer": float(jiwer_wer([self._prepare_for_wer(x) for x in refs], [self._prepare_for_wer(x) for x in hyps])),
            "cer": float(jiwer_cer(refs, hyps)),
            "samples": float(len(self.results_df)),
        }

        if self.mode == "letter":
            self.global_metrics["accuracy"] = float(accuracy_score(refs, hyps))

        return self.global_metrics

    def _build_confusion_matrix_image(self) -> Optional[BytesIO]:
        if self.mode != "letter" or self.results_df is None or self.results_df.empty:
            return None

        labels = sorted(set(self.results_df["truth"]).union(set(self.results_df["prediction"])))
        cm = confusion_matrix(self.results_df["truth"], self.results_df["prediction"], labels=labels)

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm,
            annot=True,
            fmt="g",
            cmap="Blues",
            cbar=True,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title("Confusion Matrix (Mode Letter)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer

    def _build_error_table(self, top_k: int = 12) -> List[List[str]]:
        if self.results_df is None:
            return [["No results"]]

        worst = self.results_df.sort_values(["cer", "wer"], ascending=False).head(top_k)
        table_data = [["file", "truth", "prediction", "CER", "WER"]]
        for _, row in worst.iterrows():
            table_data.append(
                [
                    os.path.basename(str(row["file"])),
                    str(row["truth"]),
                    str(row["prediction"]),
                    f"{float(row['cer']):.3f}",
                    f"{float(row['wer']):.3f}",
                ]
            )
        return table_data

    def _group_metrics(self, by_col: str) -> List[List[str]]:
        if self.results_df is None:
            return [[by_col, "count", "WER", "CER"]]

        grouped = (
            self.results_df.groupby(by_col)
            .agg(count=("file", "count"), wer=("wer", "mean"), cer=("cer", "mean"), accuracy=("is_correct", "mean"))
            .reset_index()
        )

        headers = [by_col, "count", "WER", "CER"]
        if self.mode == "letter":
            headers.append("Accuracy")

        rows: List[List[str]] = [headers]
        for _, row in grouped.iterrows():
            values = [
                str(row[by_col]),
                str(int(row["count"])),
                f"{float(row['wer']):.3f}",
                f"{float(row['cer']):.3f}",
            ]
            if self.mode == "letter":
                values.append(f"{float(row['accuracy']):.3f}")
            rows.append(values)

        return rows

    def _build_classification_report_table(self) -> List[List[str]]:
        """Build per-character precision/recall/F1 table for letter mode."""
        if self.mode != "letter" or self.results_df is None or self.results_df.empty:
            return [["Caractère", "Précision", "Rappel", "F1-Score", "Support"]]

        truth = self.results_df["truth"].fillna("").astype(str).tolist()
        prediction = self.results_df["prediction"].fillna("").astype(str).tolist()

        report = classification_report(
            truth,
            prediction,
            output_dict=True,
            zero_division=0,
        )

        rows: List[List[str]] = [["Caractère", "Précision", "Rappel", "F1-Score", "Support"]]

        class_rows = []
        for label, metrics in report.items():
            if label in {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}:
                continue
            if not isinstance(metrics, dict):
                continue

            f1_value = float(metrics.get("f1-score", 0.0))
            class_rows.append(
                (
                    f1_value,
                    [
                        str(label),
                        f"{float(metrics.get('precision', 0.0)):.3f}",
                        f"{float(metrics.get('recall', 0.0)):.3f}",
                        f"{f1_value:.3f}",
                        str(int(metrics.get("support", 0))),
                    ],
                )
            )

        class_rows.sort(key=lambda x: x[0])
        rows.extend([row for _, row in class_rows])
        return rows

    def _build_error_distribution_chart(self) -> Optional[BytesIO]:
        """Build histogram of CER distribution across evaluated samples."""
        if self.results_df is None or self.results_df.empty:
            return None

        cer_values = pd.to_numeric(self.results_df["cer"], errors="coerce").dropna()
        if cer_values.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 4.8))
        sns.histplot(cer_values, bins=20, kde=True, color="#2E86C1", ax=ax)
        ax.set_title("Distribution du CER", fontsize=12)
        ax.set_xlabel("CER")
        ax.set_ylabel("Nombre d'échantillons")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer

    def _build_worst_classes_chart(self) -> Optional[BytesIO]:
        """Build horizontal bar chart of top 10 classes with highest mean CER."""
        if self.results_df is None or self.results_df.empty:
            return None

        work_df = self.results_df.copy()
        work_df["truth"] = work_df["truth"].fillna("").astype(str)
        work_df["cer"] = pd.to_numeric(work_df["cer"], errors="coerce")
        work_df = work_df.dropna(subset=["cer"])
        work_df = work_df[work_df["truth"] != ""]

        if work_df.empty:
            return None

        worst = (
            work_df.groupby("truth", as_index=False)
            .agg(mean_cer=("cer", "mean"), count=("cer", "size"))
            .sort_values("mean_cer", ascending=False)
            .head(10)
            .sort_values("mean_cer", ascending=True)
        )

        if worst.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.barplot(data=worst, x="mean_cer", y="truth", color="#E67E22", ax=ax)
        ax.set_title("Top 10 classes les plus difficiles (CER moyen)", fontsize=12)
        ax.set_xlabel("CER moyen")
        ax.set_ylabel("Classe (truth)")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer

    @staticmethod
    def _style_table(table: Table, header_color: colors.Color = colors.HexColor("#2E5F8A")) -> None:
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), header_color),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )

    def export_results_to_pdf(self, output_pdf_path: str) -> None:
        """Export full evaluation report to a professional PDF layout."""
        if self.results_df is None:
            raise RuntimeError("run_evaluation() must be executed before export_results_to_pdf().")

        os.makedirs(os.path.dirname(output_pdf_path) or ".", exist_ok=True)

        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=A4,
            rightMargin=0.8 * inch,
            leftMargin=0.8 * inch,
            topMargin=0.8 * inch,
            bottomMargin=0.8 * inch,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleCustom",
            parent=styles["Heading1"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1F5A96"),
            fontSize=22,
            spaceAfter=12,
        )
        h2_style = ParagraphStyle(
            "H2Custom",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#2E5F8A"),
            fontSize=13,
            spaceAfter=8,
        )

        story = []
        story.append(Paragraph("ASR Model Evaluation Report", title_style))
        story.append(Paragraph(f"Mode: {self.mode.upper()} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

        metrics_table_data = [["Metric", "Value"], ["WER", f"{self.global_metrics.get('wer', 0.0):.4f}"], ["CER", f"{self.global_metrics.get('cer', 0.0):.4f}"]]
        if self.mode == "letter":
            metrics_table_data.append(["Accuracy", f"{self.global_metrics.get('accuracy', 0.0):.4f}"])
        metrics_table_data.append(["Samples", f"{int(self.global_metrics.get('samples', 0))}"])

        story.append(Paragraph("1. Global Metrics", h2_style))
        metrics_table = Table(metrics_table_data, colWidths=[2.5 * inch, 2.0 * inch])
        self._style_table(metrics_table)
        story.append(metrics_table)
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("2. Error Analysis - Worst Predictions", h2_style))
        error_table = Table(self._build_error_table(), colWidths=[1.8 * inch, 1.0 * inch, 1.3 * inch, 0.8 * inch, 0.8 * inch])
        self._style_table(error_table)
        story.append(error_table)

        cm_image = self._build_confusion_matrix_image()
        if cm_image is not None:
            story.append(PageBreak())
            story.append(Paragraph("3. Confusion Matrix", h2_style))
            story.append(Spacer(1, 0.1 * inch))
            story.append(Image(cm_image, width=6.3 * inch, height=4.8 * inch))

        story.append(PageBreak())
        story.append(Paragraph("4. Metrics by Session", h2_style))
        session_table = Table(self._group_metrics("session"), colWidths=[1.2 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch] if self.mode == "letter" else [1.6 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])
        self._style_table(session_table)
        story.append(session_table)
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("5. Metrics by Micro", h2_style))
        micro_table = Table(self._group_metrics("micro"), colWidths=[1.4 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch] if self.mode == "letter" else [1.8 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])
        self._style_table(micro_table)
        story.append(micro_table)

        if self.mode == "letter":
            story.append(PageBreak())
            story.append(Paragraph("6. Detailed Character Metrics", h2_style))
            char_table = Table(
                self._build_classification_report_table(),
                colWidths=[1.3 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch],
            )
            self._style_table(char_table)
            story.append(char_table)

        error_dist_image = self._build_error_distribution_chart()
        worst_classes_image = self._build_worst_classes_chart()

        if error_dist_image is not None or worst_classes_image is not None:
            story.append(PageBreak())
            story.append(Paragraph("7. Error Distribution and Hardest Classes", h2_style))
            story.append(Spacer(1, 0.1 * inch))

            if error_dist_image is not None:
                story.append(Paragraph("7.1 CER Distribution", styles["Heading3"]))
                story.append(Image(error_dist_image, width=6.3 * inch, height=2.8 * inch))
                story.append(Spacer(1, 0.15 * inch))

            if worst_classes_image is not None:
                story.append(Paragraph("7.2 Worst Classes by Mean CER", styles["Heading3"]))
                story.append(Image(worst_classes_image, width=6.3 * inch, height=3.2 * inch))

        doc.build(story)


def main() -> None:
    json_path = "/home/melissa/gdrive/Cours/ENSEIRB-MATMECA/2A/S8/PFA/data/labels.json"
    model_path = "/home/melissa/gdrive/Cours/ENSEIRB-MATMECA/2A/S8/PFA/code/inference/training_outputs/model.safetensors"
    mode = "letter"
    output_pdf = "/home/melissa/gdrive/Cours/ENSEIRB-MATMECA/2A/S8/PFA/code/inference/reports/evaluation_report.pdf"

    evaluator = ModelEvaluator(json_path=json_path, model_path=model_path, mode=mode)
    metrics = evaluator.run_evaluation()
    evaluator.export_results_to_pdf(output_pdf)
    print("Evaluation complete.")
    print(metrics)
    print(f"PDF generated: {output_pdf}")


if __name__ == "__main__":
    main()
