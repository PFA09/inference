"""
reporting/pdf_report.py — Génération du rapport PDF d'évaluation.

Ce module est totalement découplé du pipeline : il reçoit un DataFrame
et un dict de métriques, et produit un PDF. Aucune dépendance vers
inference, trainer ou evaluator.

Sections générées :
1. Métriques globales
2. Pires prédictions
3. Matrice de confusion (mode lettre)
4. F1-score par caractère (mode lettre)
5. Métriques par session
6. Métriques par micro
7. Distribution du CER
8. Classes les plus difficiles (CER moyen)
"""

from __future__ import annotations

import logging
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class ReportBuilder:
    """
    Construit et exporte un rapport PDF d'évaluation 

    Parameters
    ----------
    mode:
        "letter" | "word" | "sentences"
    results_df:
        DataFrame avec colonnes : file, session, micro, speaker, type,
        truth, prediction, cer, wer, is_correct, error.
    global_metrics:
        Dict {"wer", "cer", "accuracy"?, "samples"}.
    """

    def __init__(
        self,
        mode: str,
        results_df: pd.DataFrame,
        global_metrics: Dict[str, float],
    ) -> None:
        self._mode = mode
        self._df = results_df.copy()
        self._metrics = global_metrics

    # ── API publique ───────────────────────────────────────────────────────

    def export(self, output_path: str) -> None:
        """Génère et sauvegarde le rapport PDF."""
        import os
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.8 * inch,
            leftMargin=0.8 * inch,
            topMargin=0.8 * inch,
            bottomMargin=0.8 * inch,
        )

        styles = getSampleStyleSheet()
        title_s = ParagraphStyle(
            "T", parent=styles["Heading1"], alignment=TA_CENTER,
            textColor=colors.HexColor("#1F5A96"), fontSize=22, spaceAfter=12,
        )
        h2_s = ParagraphStyle(
            "H2", parent=styles["Heading2"],
            textColor=colors.HexColor("#2E5F8A"), fontSize=13, spaceAfter=8,
        )
        h3_s = styles["Heading3"]

        story = []
        story.append(Paragraph("ASR Evaluation Report", title_s))
        story.append(Paragraph(
            f"Mode: {self._mode.upper()} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.2 * inch))

        # 1. Métriques globales
        story.append(Paragraph("1. Global Metrics", h2_s))
        story.append(self._make_table(self._global_metrics_data()))
        story.append(Spacer(1, 0.2 * inch))

        # 2. Pires prédictions
        story.append(Paragraph("2. Worst Predictions", h2_s))
        story.append(self._make_table(self._worst_predictions_data(), col_widths=[
            1.8 * inch, 1.0 * inch, 1.3 * inch, 0.8 * inch, 0.8 * inch
        ]))

        # 3. Matrice de confusion (lettre seulement)
        cm_img = self._confusion_matrix_image()
        if cm_img:
            story.append(PageBreak())
            story.append(Paragraph("3. Confusion Matrix", h2_s))
            story.append(Image(cm_img, width=6.3 * inch, height=4.8 * inch))

        # 4. F1 par caractère (lettre seulement)
        f1_data = self._f1_per_char_data()
        if f1_data:
            story.append(PageBreak())
            story.append(Paragraph("4. Per-Character F1 Score", h2_s))
            story.append(self._make_table(f1_data, col_widths=[
                1.3 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch
            ]))

        # 5. Par session
        story.append(PageBreak())
        story.append(Paragraph("5. Metrics by Session", h2_s))
        story.append(self._make_table(self._group_metrics("session")))
        story.append(Spacer(1, 0.2 * inch))

        # 6. Par micro
        story.append(Paragraph("6. Metrics by Micro", h2_s))
        story.append(self._make_table(self._group_metrics("micro")))

        # 7. Distribution CER + classes difficiles
        cer_img = self._cer_distribution_image()
        worst_img = self._worst_classes_image()
        if cer_img or worst_img:
            story.append(PageBreak())
            story.append(Paragraph("7. Error Distribution", h2_s))
            if cer_img:
                story.append(Paragraph("CER Distribution", h3_s))
                story.append(Image(cer_img, width=6.3 * inch, height=2.8 * inch))
                story.append(Spacer(1, 0.15 * inch))
            if worst_img:
                story.append(Paragraph("Top 10 Hardest Classes (mean CER)", h3_s))
                story.append(Image(worst_img, width=6.3 * inch, height=3.2 * inch))

        doc.build(story)
        logger.info("Rapport PDF généré : %s", output_path)

    # ── Données tabulaires ─────────────────────────────────────────────────

    def _global_metrics_data(self) -> List[List[str]]:
        rows = [
            ["Metric", "Value"],
            ["WER", f"{self._metrics.get('wer', 0):.4f}"],
            ["CER", f"{self._metrics.get('cer', 0):.4f}"],
        ]
        if "accuracy" in self._metrics:
            rows.append(["Accuracy", f"{self._metrics['accuracy']:.4f}"])
        rows.append(["Samples", str(int(self._metrics.get("samples", 0)))])
        return rows

    def _worst_predictions_data(self, top_k: int = 12) -> List[List[str]]:
        worst = self._df.sort_values(["cer", "wer"], ascending=False).head(top_k)
        header = ["File", "Truth", "Prediction", "CER", "WER"]
        rows = [header]
        for _, r in worst.iterrows():
            import os
            rows.append([
                os.path.basename(str(r["file"])),
                str(r["truth"]),
                str(r["prediction"]),
                f"{float(r['cer']):.3f}",
                f"{float(r['wer']):.3f}",
            ])
        return rows

    def _f1_per_char_data(self) -> Optional[List[List[str]]]:
        """F1-score par caractère via sklearn, trié du pire au meilleur."""
        if self._mode != "letter" or self._df.empty:
            return None
        from sklearn.metrics import classification_report

        truth = self._df["truth"].fillna("").tolist()
        pred = self._df["prediction"].fillna("").tolist()

        report = classification_report(truth, pred, output_dict=True, zero_division=0)
        rows = [["Character", "Precision", "Recall", "F1-Score", "Support"]]

        char_rows = []
        for label, m in report.items():
            if label in {"accuracy", "macro avg", "weighted avg", "micro avg"}:
                continue
            if not isinstance(m, dict):
                continue
            char_rows.append((
                float(m.get("f1-score", 0)),
                [
                    str(label),
                    f"{float(m.get('precision', 0)):.3f}",
                    f"{float(m.get('recall', 0)):.3f}",
                    f"{float(m.get('f1-score', 0)):.3f}",
                    str(int(m.get("support", 0))),
                ],
            ))
        char_rows.sort(key=lambda x: x[0])  # pire F1 en premier
        rows.extend([r for _, r in char_rows])
        return rows

    def _group_metrics(self, by_col: str) -> List[List[str]]:
        headers = [by_col.capitalize(), "Count", "WER", "CER"]
        if self._mode == "letter":
            headers.append("Accuracy")
        rows = [headers]

        if by_col not in self._df.columns:
            return rows

        for group_val, sub in self._df.groupby(by_col):
            row = [
                str(group_val),
                str(len(sub)),
                f"{float(sub['wer'].mean()):.3f}",
                f"{float(sub['cer'].mean()):.3f}",
            ]
            if self._mode == "letter":
                row.append(f"{float(sub['is_correct'].mean()):.3f}")
            rows.append(row)
        return rows

    # ── Graphiques ─────────────────────────────────────────────────────────

    def _confusion_matrix_image(self) -> Optional[BytesIO]:
        """
        Matrice de confusion robuste.

        CORRECTIF : on vérifie que la matrice n'est pas mono-classe
        avant d'appeler seaborn, et on gère les DataFrames vides.
        """
        if self._mode != "letter" or self._df.empty:
            return None

        truth = self._df["truth"].fillna("").tolist()
        pred = self._df["prediction"].fillna("").tolist()
        labels = sorted(set(truth) | set(pred))

        if len(labels) < 2:
            logger.warning("Matrice de confusion ignorée : moins de 2 classes uniques.")
            return None

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(truth, pred, labels=labels)

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm, annot=True, fmt="g", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_title("Confusion Matrix (letter mode)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf

    def _cer_distribution_image(self) -> Optional[BytesIO]:
        if self._df.empty:
            return None
        cer_vals = pd.to_numeric(self._df["cer"], errors="coerce").dropna()
        if cer_vals.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 4.8))
        sns.histplot(cer_vals, bins=20, kde=True, color="#2E86C1", ax=ax)
        ax.set_title("CER Distribution")
        ax.set_xlabel("CER")
        ax.set_ylabel("Samples")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf

    def _worst_classes_image(self) -> Optional[BytesIO]:
        if self._df.empty:
            return None
        work = self._df.copy()
        work["cer"] = pd.to_numeric(work["cer"], errors="coerce")
        work = work.dropna(subset=["cer"])
        work = work[work["truth"].fillna("") != ""]
        if work.empty:
            return None

        worst = (
            work.groupby("truth", as_index=False)
            .agg(mean_cer=("cer", "mean"), count=("cer", "size"))
            .sort_values("mean_cer", ascending=False)
            .head(10)
            .sort_values("mean_cer", ascending=True)
        )

        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.barplot(data=worst, x="mean_cer", y="truth", color="#E67E22", ax=ax)
        ax.set_title("Top 10 Hardest Classes (mean CER)")
        ax.set_xlabel("Mean CER")
        ax.set_ylabel("Truth class")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf

    # ── Utilitaire table ───────────────────────────────────────────────────

    @staticmethod
    def _make_table(data: List[List[str]], col_widths=None):
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import Table, TableStyle

        if col_widths is None:
            n_cols = len(data[0]) if data else 1
            col_widths = [6.0 * inch / n_cols] * n_cols

        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5F8A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table
