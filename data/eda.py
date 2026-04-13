"""
data/eda.py — Analyse exploratoire du corpus audio (EDA).

Usage :
    analyzer = DataAnalyzer(config)
    analyzer.analyze()           # extrait les features audio → self.df
    analyzer.export_pdf(path)    # génère le rapport PDF
"""

from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ASRConfig

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Analyse le corpus audio défini dans un ASRConfig et génère un rapport PDF.

    Parameters
    ----------
    config:
        Configuration complète du pipeline. Seuls config.paths est utilisé
        pour localiser labels.json et le dossier audio.
    """

    def __init__(self, config: ASRConfig) -> None:
        self._config = config
        self._cache: Dict[str, dict] = {}  # chemin → features extraites
        self.df: Optional[pd.DataFrame] = None

    # ── API publique ───────────────────────────────────────────────────────

    def analyze(self) -> pd.DataFrame:
        """
        Charge labels.json, extrait les features audio de chaque fichier
        et stocke le résultat dans self.df.

        Returns
        -------
        pd.DataFrame
            Une ligne par enregistrement avec colonnes :
            file, label, type, speaker, session, micro,
            duration_s, speech_ratio, snr_db, zcr_mean.
        """
        records = self._load_json()
        rows = []
        for rec in records:
            audio_path = self._resolve_path(str(rec["file"]))
            features = self._extract_features(audio_path)
            rows.append({
                "file": str(rec.get("file", "")),
                "label": str(rec.get("label", "")),
                "type": str(rec.get("type", "")),
                "speaker": str(rec.get("speaker", "")),
                "session": str(rec.get("session", "")),
                "micro": str(rec.get("micro", "")),
                **features,
            })

        self.df = pd.DataFrame(rows)
        logger.info("EDA terminée : %d enregistrements analysés.", len(self.df))
        return self.df

    def export_pdf(self, output_path: str) -> None:
        """
        Génère le rapport PDF EDA.

        Parameters
        ----------
        output_path:
            Chemin du fichier PDF à créer.
        """
        if self.df is None:
            raise RuntimeError("Appeler analyze() avant export_pdf().")

        import os
        from datetime import datetime

        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
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
            "EDA_T", parent=styles["Heading1"], alignment=TA_CENTER,
            textColor=colors.HexColor("#1F5A96"), fontSize=22, spaceAfter=12,
        )
        h2_s = ParagraphStyle(
            "EDA_H2", parent=styles["Heading2"],
            textColor=colors.HexColor("#2E5F8A"), fontSize=13, spaceAfter=8,
        )

        story = []
        story.append(Paragraph("ASR Corpus — Exploratory Data Analysis", title_s))
        story.append(Paragraph(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"| {len(self.df)} enregistrements",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.2 * inch))

        # 1. Statistiques globales
        story.append(Paragraph("1. Statistiques globales du dataset", h2_s))
        story.append(_make_table(self._global_stats_data()))
        story.append(Spacer(1, 0.2 * inch))

        # 2. Résumé acoustique
        story.append(Paragraph("2. Résumé acoustique", h2_s))
        story.append(_make_table(self._acoustic_summary_data()))
        story.append(Spacer(1, 0.2 * inch))

        # 3. Répartition par type
        story.append(PageBreak())
        story.append(Paragraph("3. Répartition par type", h2_s))
        story.append(_make_table(self._by_group_data("type")))
        story.append(Spacer(1, 0.15 * inch))
        img = self._boxplot_by_type()
        if img:
            story.append(Image(img, width=6.3 * inch, height=3.2 * inch))

        # 4. Répartition par session et par micro
        story.append(PageBreak())
        story.append(Paragraph("4. Répartition par session", h2_s))
        story.append(_make_table(self._by_group_data("session")))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("Répartition par micro", h2_s))
        story.append(_make_table(self._by_group_data("micro")))

        # 5. Top 20 labels
        story.append(PageBreak())
        story.append(Paragraph("5. Top 20 labels les plus fréquents", h2_s))
        img = self._top_labels_chart()
        if img:
            story.append(Image(img, width=6.3 * inch, height=4.0 * inch))

        # 6. Variabilité inter-locuteurs
        story.append(PageBreak())
        story.append(Paragraph("6. Variabilité inter-locuteurs", h2_s))
        img = self._speaker_scatter()
        if img:
            story.append(Image(img, width=6.3 * inch, height=4.0 * inch))

        # 7. Distributions acoustiques globales
        story.append(PageBreak())
        story.append(Paragraph("7. Distributions acoustiques globales", h2_s))
        img = self._acoustic_distributions()
        if img:
            story.append(Image(img, width=6.3 * inch, height=3.2 * inch))

        doc.build(story)
        logger.info("Rapport EDA généré : %s", output_path)

    # ── Extraction des features ────────────────────────────────────────────

    def _extract_features(self, audio_path: str) -> dict:
        """Extrait les features d'un fichier audio. Utilise le cache."""
        if audio_path in self._cache:
            return self._cache[audio_path]

        null = {"duration_s": float("nan"), "speech_ratio": float("nan"),
                "snr_db": float("nan"), "zcr_mean": float("nan")}

        if not Path(audio_path).exists():
            logger.warning("Fichier manquant : %s", audio_path)
            self._cache[audio_path] = null
            return null

        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=16_000, mono=True)
            duration_s = float(len(y) / sr)

            # ── Speech ratio ──────────────────────────────────────────────
            intervals = librosa.effects.split(y, top_db=30)
            speech_samples = sum(end - start for start, end in intervals)
            speech_ratio = float(speech_samples / max(len(y), 1))

            # ── SNR estimé ────────────────────────────────────────────────
            speech_mask = np.zeros(len(y), dtype=bool)
            for start, end in intervals:
                speech_mask[start:end] = True
            silence_mask = ~speech_mask

            speech_rms = float(np.sqrt(np.mean(y[speech_mask] ** 2))) if speech_mask.any() else 0.0
            silence_rms = float(np.sqrt(np.mean(y[silence_mask] ** 2))) if silence_mask.any() else 0.0

            if silence_rms > 1e-9 and speech_rms > 1e-9:
                snr_db = float(20.0 * np.log10(speech_rms / silence_rms))
            else:
                snr_db = float("nan")

            # ── Zero-crossing rate ────────────────────────────────────────
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = float(np.mean(zcr))

            result = {
                "duration_s": duration_s,
                "speech_ratio": speech_ratio,
                "snr_db": snr_db,
                "zcr_mean": zcr_mean,
            }
            self._cache[audio_path] = result
            return result

        except Exception as exc:
            logger.warning("Erreur extraction '%s' : %s", audio_path, exc)
            self._cache[audio_path] = null
            return null

    # ── Données tabulaires ─────────────────────────────────────────────────

    def _load_json(self) -> List[dict]:
        path = self._config.paths.data_json
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _resolve_path(self, audio_file: str) -> str:
        p = Path(audio_file)
        if p.is_absolute():
            return str(p)
        return str(self._config.paths.audio_dir / audio_file)

    def _global_stats_data(self) -> List[List[str]]:
        df = self.df
        valid = df["duration_s"].notna().sum()
        rows = [
            ["Métrique", "Valeur"],
            ["Fichiers totaux", str(len(df))],
            ["Fichiers avec features extraites", str(valid)],
            ["Labels uniques", str(df["label"].nunique())],
            ["Types distincts", ", ".join(sorted(df["type"].dropna().unique()))],
            ["Sessions", ", ".join(sorted(df["session"].dropna().unique()))],
            ["Locuteurs", ", ".join(sorted(df["speaker"].dropna().unique()))],
            ["Micros", ", ".join(sorted(df["micro"].dropna().unique()))],
            ["Durée totale (min)", f"{df['duration_s'].sum() / 60:.1f}"],
        ]
        return rows

    def _acoustic_summary_data(self) -> List[List[str]]:
        rows = [["Feature", "Moyenne", "Médiane", "Min", "Max"]]
        features = [
            ("Durée (s)", "duration_s"),
            ("Speech ratio", "speech_ratio"),
            ("SNR (dB)", "snr_db"),
            ("ZCR moyen", "zcr_mean"),
        ]
        for label, col in features:
            s = self.df[col].dropna()
            if s.empty:
                rows.append([label, "—", "—", "—", "—"])
            else:
                rows.append([
                    label,
                    f"{s.mean():.3f}",
                    f"{s.median():.3f}",
                    f"{s.min():.3f}",
                    f"{s.max():.3f}",
                ])
        return rows

    def _by_group_data(self, col: str) -> List[List[str]]:
        headers = [col.capitalize(), "Count", "Durée moy. (s)", "Speech ratio moy.", "SNR moy. (dB)"]
        rows = [headers]
        if col not in self.df.columns:
            return rows
        for group_val, sub in self.df.groupby(col):
            rows.append([
                str(group_val),
                str(len(sub)),
                f"{sub['duration_s'].mean():.3f}" if sub['duration_s'].notna().any() else "—",
                f"{sub['speech_ratio'].mean():.3f}" if sub['speech_ratio'].notna().any() else "—",
                f"{sub['snr_db'].mean():.1f}" if sub['snr_db'].notna().any() else "—",
            ])
        return rows

    # ── Graphiques ─────────────────────────────────────────────────────────

    def _boxplot_by_type(self) -> Optional[BytesIO]:
        df = self.df.dropna(subset=["duration_s", "speech_ratio"])
        if df.empty or "type" not in df.columns:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        sns.boxplot(data=df, x="type", y="duration_s", hue="type",
                    palette="Blues", legend=False, ax=axes[0])
        axes[0].set_title("Durée par type (s)")
        axes[0].set_xlabel("Type")
        axes[0].set_ylabel("Durée (s)")
        axes[0].tick_params(axis="x", rotation=20)

        sns.boxplot(data=df, x="type", y="speech_ratio", hue="type",
                    palette="Greens", legend=False, ax=axes[1])
        axes[1].set_title("Speech ratio par type")
        axes[1].set_xlabel("Type")
        axes[1].set_ylabel("Speech ratio")
        axes[1].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        return _fig_to_buf(fig)

    def _top_labels_chart(self) -> Optional[BytesIO]:
        if self.df.empty:
            return None

        counts = (
            self.df["label"]
            .value_counts()
            .head(20)
            .sort_values()
        )
        if counts.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        counts.plot(kind="barh", color="#2E86C1", ax=ax)
        ax.set_title("Top 20 labels les plus fréquents")
        ax.set_xlabel("Nombre d'occurrences")
        ax.set_ylabel("Label")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        return _fig_to_buf(fig)

    def _speaker_scatter(self) -> Optional[BytesIO]:
        df = self.df.dropna(subset=["duration_s", "speech_ratio"])
        if df.empty or "speaker" not in df.columns:
            return None

        grouped = (
            df.groupby("speaker")
            .agg(
                mean_duration=("duration_s", "mean"),
                mean_speech_ratio=("speech_ratio", "mean"),
                count=("duration_s", "size"),
            )
            .reset_index()
        )
        if grouped.empty:
            return None

        fig, ax = plt.subplots(figsize=(9, 5))
        scatter = ax.scatter(
            grouped["mean_duration"],
            grouped["mean_speech_ratio"],
            s=grouped["count"] * 10,
            c=range(len(grouped)),
            cmap="tab10",
            alpha=0.8,
            edgecolors="grey",
            linewidths=0.5,
        )
        for _, row in grouped.iterrows():
            ax.annotate(
                row["speaker"],
                (row["mean_duration"], row["mean_speech_ratio"]),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=8,
            )

        ax.set_title("Variabilité inter-locuteurs\n(taille des points ∝ nb samples)")
        ax.set_xlabel("Durée moyenne (s)")
        ax.set_ylabel("Speech ratio moyen")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        return _fig_to_buf(fig)

    def _acoustic_distributions(self) -> Optional[BytesIO]:
        df = self.df.dropna(subset=["snr_db", "zcr_mean"])
        if df.empty:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        sns.histplot(df["snr_db"].dropna(), bins=25, kde=True,
                     color="#E67E22", ax=axes[0])
        axes[0].set_title("Distribution du SNR estimé")
        axes[0].set_xlabel("SNR (dB)")
        axes[0].set_ylabel("Échantillons")
        axes[0].grid(axis="y", alpha=0.25)

        sns.histplot(df["zcr_mean"].dropna(), bins=25, kde=True,
                     color="#2E86C1", ax=axes[1])
        axes[1].set_title("Distribution du ZCR moyen")
        axes[1].set_xlabel("ZCR moyen")
        axes[1].set_ylabel("Échantillons")
        axes[1].grid(axis="y", alpha=0.25)

        plt.tight_layout()
        return _fig_to_buf(fig)


# ── Utilitaires module-level ───────────────────────────────────────────────────

def _fig_to_buf(fig: plt.Figure) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def _make_table(data: List[List[str]], col_widths=None):
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle

    if not data:
        return Table([["(vide)"]])

    if col_widths is None:
        n_cols = len(data[0])
        col_widths = [6.0 * inch / n_cols] * n_cols

    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5F8A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F4F7FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table
