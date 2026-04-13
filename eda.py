"""
Module d'Analyse Exploratoire des Données (EDA) pour des corpus audio de dysarthrie.
Extraction de features acoustiques, analyses par type et par locuteur, puis export PDF.
"""

import io
import json
import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Suppression des warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration de matplotlib
matplotlib.use("Agg")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Import pour lire les fichiers audio
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


class DataAnalyzer:
    """Analyse exploratoire audio avec extraction de features acoustiques."""

    FEATURE_COLUMNS = ["duration", "speech_ratio", "snr_estimate", "mean_zcr"]

    def __init__(self, json_path: str, audio_directory: Optional[str] = None, top_db: float = 35.0, max_speakers_plot: int = 12) -> None:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Fichier JSON introuvable: {json_path}")

        self.json_path = json_path
        self.audio_directory = audio_directory or os.path.dirname(json_path)
        self.top_db = top_db
        self.max_speakers_plot = max_speakers_plot
        self.df: Optional[pd.DataFrame] = None
        self._feature_cache: Dict[str, Dict[str, float]] = {}
        self.feature_extraction_failures: List[Dict[str, str]] = []

        sns.set_theme(style="whitegrid", context="talk")
        self._load_and_parse()

    def _blank_feature_row(self) -> Dict[str, float]:
        return {"duration": np.nan, "speech_ratio": np.nan, "snr_estimate": np.nan, "mean_zcr": np.nan}

    def _resolve_audio_path(self, audio_file: Any) -> Optional[str]:
        if pd.isna(audio_file):
            return None

        audio_name = str(audio_file).strip()
        if not audio_name or audio_name.lower() == "nan":
            return None

        return os.path.join(self.audio_directory, audio_name)

    def _string_series(self, column: str) -> pd.Series:
        if self.df is None or column not in self.df.columns:
            return pd.Series(dtype=str)

        return self.df[column].fillna("").astype(str).str.strip()

    def _numeric_series(self, column: str) -> pd.Series:
        if self.df is None or column not in self.df.columns:
            return pd.Series(dtype=float)

        return pd.to_numeric(self.df[column], errors="coerce")

    def _non_empty_unique_count(self, column: str) -> int:
        series = self._string_series(column)
        if series.empty:
            return 0
        return int(series[series != ""].nunique())

    def _load_audio(self, full_path: str) -> tuple[np.ndarray, int]:
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa n'est pas disponible")

        y, sr = librosa.load(full_path, sr=None, mono=True)
        if y is None or len(y) == 0:
            raise ValueError("Fichier audio vide")
        return y, sr

    def _extract_audio_features(self, audio_file: Any) -> Dict[str, float]:
        cache_key = str(audio_file)
        if cache_key in self._feature_cache:
            return dict(self._feature_cache[cache_key])

        blank = self._blank_feature_row()
        full_path = self._resolve_audio_path(audio_file)
        if full_path is None or not os.path.exists(full_path):
            self.feature_extraction_failures.append({"file": cache_key, "reason": "missing"})
            self._feature_cache[cache_key] = blank
            return dict(blank)

        try:
            if LIBROSA_AVAILABLE:
                y, sr = self._load_audio(full_path)
                total_samples = int(len(y))
                duration = float(librosa.get_duration(y=y, sr=sr))

                intervals = librosa.effects.split(y, top_db=self.top_db)
                speech_samples = 0
                speech_chunks: List[np.ndarray] = []
                silence_chunks: List[np.ndarray] = []
                cursor = 0

                for start, end in intervals:
                    if end <= start:
                        continue
                    speech_samples += int(end - start)
                    speech_chunks.append(y[start:end])
                    if start > cursor:
                        silence_chunks.append(y[cursor:start])
                    cursor = end

                if cursor < total_samples:
                    silence_chunks.append(y[cursor:total_samples])

                speech_ratio = float(speech_samples / total_samples) if total_samples > 0 else np.nan

                if speech_chunks and silence_chunks:
                    speech_signal = np.concatenate(speech_chunks)
                    silence_signal = np.concatenate(silence_chunks)
                    speech_rms = float(np.sqrt(np.mean(np.square(speech_signal)) + 1e-12))
                    silence_rms = float(np.sqrt(np.mean(np.square(silence_signal)) + 1e-12))
                    snr_estimate = float(20.0 * np.log10((speech_rms + 1e-12) / (silence_rms + 1e-12)))
                else:
                    snr_estimate = np.nan

                zcr = librosa.feature.zero_crossing_rate(y)
                mean_zcr = float(np.nanmean(zcr)) if zcr.size else np.nan
            elif SOUNDFILE_AVAILABLE:
                data, sr = sf.read(full_path, always_2d=False)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                duration = float(len(data) / sr) if sr else np.nan
                speech_ratio = np.nan
                snr_estimate = np.nan
                mean_zcr = np.nan
            else:
                raise ImportError("librosa et soundfile sont indisponibles")

            result = {
                "duration": duration,
                "speech_ratio": speech_ratio,
                "snr_estimate": snr_estimate,
                "mean_zcr": mean_zcr,
            }
            self._feature_cache[cache_key] = result
            return dict(result)

        except Exception as exc:
            self.feature_extraction_failures.append({"file": cache_key, "reason": str(exc)})
            self._feature_cache[cache_key] = blank
            return dict(blank)

    def _load_and_parse(self) -> None:
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.df = pd.DataFrame(data)
            if self.df.empty:
                self.df = pd.DataFrame(columns=["file", "label", "type", "speaker", "micro", "session"])

            for column in ["file", "label", "type", "speaker", "micro", "session"]:
                if column not in self.df.columns:
                    self.df[column] = ""
                self.df[column] = self.df[column].fillna("").astype(str).str.strip()

            print("📊 Extraction des features acoustiques (durée, speech_ratio, SNR, ZCR)...")
            feature_rows = [self._extract_audio_features(audio_file) for audio_file in self.df["file"].tolist()]
            feature_df = pd.DataFrame(feature_rows)
            self.df = pd.concat([self.df.reset_index(drop=True), feature_df], axis=1)

            for column in self.FEATURE_COLUMNS:
                self.df[column] = pd.to_numeric(self.df[column], errors="coerce")

            valid_ratio = 0.0 if len(self.df) == 0 else float(100.0 * (1.0 - self.df["duration"].isna().mean()))
            print(f"✅ {len(self.df)} fichiers chargés. Couverture durée valide: {valid_ratio:.1f}%")
            if self.feature_extraction_failures:
                print(f"⚠️ {len(self.feature_extraction_failures)} fichiers n'ont pas pu être analysés proprement.")

        except json.JSONDecodeError as e:
            raise ValueError(f"Erreur lors du décodage du JSON: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement des données: {str(e)}")

    def _type_order(self) -> List[str]:
        if self.df is None or "type" not in self.df.columns:
            return []

        types = [value for value in self._string_series("type").unique().tolist() if value]
        priority = {"letter": 0, "word": 1, "sentence": 2, "sentences": 2}
        return sorted(types, key=lambda value: (priority.get(value.lower(), 99), value.lower()))

    def _speaker_order(self) -> List[str]:
        if self.df is None or "speaker" not in self.df.columns:
            return []

        counts = self._string_series("speaker")
        counts = counts[counts != ""].value_counts()
        return counts.head(self.max_speakers_plot).index.tolist()

    def _safe_stat(self, value: float, precision: int = 2) -> str:
        if pd.isna(value):
            return "N/A"
        return f"{value:.{precision}f}"

    def _style_table(self, table: Table, header_color: str = "#2e5f8a", align: str = "LEFT") -> Table:
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#7f8c8d")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
                    ("ALIGN", (0, 0), (-1, -1), align),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        return table

    def _build_dataset_stats_table(self) -> List[List[str]]:
        duration = self._numeric_series("duration")
        speech_ratio = self._numeric_series("speech_ratio")
        snr = self._numeric_series("snr_estimate")
        zcr = self._numeric_series("mean_zcr")

        stats = [
            ["Métrique", "Valeur"],
            ["Nombre total de fichiers", str(len(self.df))],
            ["Labels uniques", str(self._non_empty_unique_count("label"))],
            ["Types uniques", str(self._non_empty_unique_count("type"))],
            ["Locuteurs uniques", str(self._non_empty_unique_count("speaker"))],
            ["Sessions uniques", str(self._non_empty_unique_count("session"))],
            ["Fichiers avec durée valide", str(int(duration.notna().sum()))],
            ["Fichiers avec speech_ratio valide", str(int(speech_ratio.notna().sum()))],
            ["Fichiers avec SNR valide", str(int(snr.notna().sum()))],
            ["Fichiers avec ZCR valide", str(int(zcr.notna().sum()))],
            ["Durée moyenne (s)", self._safe_stat(float(duration.mean()), 2)],
            ["Durée médiane (s)", self._safe_stat(float(duration.median()), 2)],
            ["Speech ratio moyen", self._safe_stat(float(speech_ratio.mean()), 3)],
            ["SNR moyen (dB)", self._safe_stat(float(snr.mean()), 2)],
            ["ZCR moyen", self._safe_stat(float(zcr.mean()), 4)],
        ]
        return stats

    def _build_acoustic_summary_table(self) -> List[List[str]]:
        summary = [["Feature", "Moyenne", "Médiane", "Min", "Max"]]
        for column in ["snr_estimate", "mean_zcr"]:
            series = self._numeric_series(column)
            summary.append(
                [
                    column,
                    self._safe_stat(float(series.mean()), 3 if column == "mean_zcr" else 2),
                    self._safe_stat(float(series.median()), 3 if column == "mean_zcr" else 2),
                    self._safe_stat(float(series.min()), 3 if column == "mean_zcr" else 2),
                    self._safe_stat(float(series.max()), 3 if column == "mean_zcr" else 2),
                ]
            )
        return summary

    def _build_type_summary_table(self) -> List[List[str]]:
        stats = [["Type", "Nombre", "% du total", "Durée moyenne (s)", "Speech ratio moyen"]]
        type_series = self._string_series("type")
        type_order = self._type_order()

        for label_type in type_order:
            subset = self.df[type_series == label_type]
            count = len(subset)
            percentage = (count / len(self.df)) * 100 if len(self.df) else np.nan
            stats.append(
                [
                    label_type,
                    str(count),
                    self._safe_stat(float(percentage), 1) + "%" if pd.notna(percentage) else "N/A",
                    self._safe_stat(float(pd.to_numeric(subset["duration"], errors="coerce").mean()), 2),
                    self._safe_stat(float(pd.to_numeric(subset["speech_ratio"], errors="coerce").mean()), 3),
                ]
            )

        return stats

    def _build_speaker_summary_table(self) -> List[List[str]]:
        if self.df is None or "speaker" not in self.df.columns:
            return [["Speaker", "Nombre", "Durée moyenne (s)", "Speech ratio moyen"]]

        speaker_series = self._string_series("speaker")
        valid = self.df[speaker_series != ""].copy()
        if valid.empty:
            return [["Speaker", "Nombre", "Durée moyenne (s)", "Speech ratio moyen"]]

        grouped = (
            valid.groupby("speaker", dropna=False)
            .agg(count=("speaker", "size"), duration_mean=("duration", "mean"), speech_ratio_mean=("speech_ratio", "mean"))
            .reset_index()
            .sort_values(["count", "speaker"], ascending=[False, True])
        )

        data = [["Speaker", "Nombre", "Durée moyenne (s)", "Speech ratio moyen"]]
        for _, row in grouped.head(self.max_speakers_plot).iterrows():
            data.append([
                str(row["speaker"]),
                str(int(row["count"])),
                self._safe_stat(float(row["duration_mean"]), 2),
                self._safe_stat(float(row["speech_ratio_mean"]), 3),
            ])
        return data

    def _create_histograms_chart(self) -> io.BytesIO:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.histplot(self._numeric_series("snr_estimate"), bins=30, kde=True, color="#1f77b4", ax=axes[0])
        axes[0].set_title("Distribution du SNR estimé", fontweight="bold")
        axes[0].set_xlabel("SNR estimé (dB)")
        axes[0].set_ylabel("Nombre d'échantillons")

        sns.histplot(self._numeric_series("mean_zcr"), bins=30, kde=True, color="#d95f02", ax=axes[1])
        axes[1].set_title("Distribution du Zero-Crossing Rate moyen", fontweight="bold")
        axes[1].set_xlabel("Mean ZCR")
        axes[1].set_ylabel("Nombre d'échantillons")

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer

    def _create_type_boxplots_chart(self) -> io.BytesIO:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        type_order = self._type_order()
        plot_df = self.df.copy()

        sns.boxplot(data=plot_df, x="type", y="duration", order=type_order, ax=axes[0], palette="Blues", showfliers=False)
        axes[0].set_title("Durée par type", fontweight="bold")
        axes[0].set_xlabel("Type")
        axes[0].set_ylabel("Durée (s)")
        axes[0].tick_params(axis="x", rotation=20)

        sns.boxplot(data=plot_df, x="type", y="speech_ratio", order=type_order, ax=axes[1], palette="Greens", showfliers=False)
        axes[1].set_title("Speech ratio par type", fontweight="bold")
        axes[1].set_xlabel("Type")
        axes[1].set_ylabel("Speech ratio")
        axes[1].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer

    def _create_speaker_scatter_chart(self) -> io.BytesIO:
        fig, ax = plt.subplots(figsize=(12, 7))

        if self.df is None or "speaker" not in self.df.columns:
            ax.text(0.5, 0.5, "Aucune colonne speaker disponible.", ha="center", va="center")
        else:
            speaker_series = self._string_series("speaker")
            valid = self.df[speaker_series != ""].copy()

            if valid.empty:
                ax.text(0.5, 0.5, "Aucun speaker exploitable.", ha="center", va="center")
            else:
                summary = (
                    valid.groupby("speaker", dropna=False)
                    .agg(duration_mean=("duration", "mean"), speech_ratio_mean=("speech_ratio", "mean"), samples=("speaker", "size"))
                    .reset_index()
                    .sort_values(["samples", "speaker"], ascending=[False, True])
                )

                sns.scatterplot(
                    data=summary,
                    x="duration_mean",
                    y="speech_ratio_mean",
                    size="samples",
                    sizes=(80, 500),
                    color="#1f77b4",
                    edgecolor="white",
                    linewidth=0.8,
                    legend=False,
                    ax=ax,
                )

                for _, row in summary.head(self.max_speakers_plot).iterrows():
                    ax.annotate(
                        str(row["speaker"]),
                        (row["duration_mean"], row["speech_ratio_mean"]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                        color="#2c3e50",
                    )

        ax.set_title("Variabilité inter-locuteurs: durée moyenne vs speech ratio moyen", fontweight="bold")
        ax.set_xlabel("Durée moyenne par speaker (s)")
        ax.set_ylabel("Speech ratio moyen par speaker")
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer

    def _create_speaker_violin_chart(self) -> io.BytesIO:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        if self.df is None or "speaker" not in self.df.columns:
            for axis in axes:
                axis.text(0.5, 0.5, "Aucune colonne speaker disponible.", ha="center", va="center")
        else:
            speaker_order = self._speaker_order()
            valid = self.df[self._string_series("speaker") != ""].copy()
            if valid.empty:
                for axis in axes:
                    axis.text(0.5, 0.5, "Aucun speaker exploitable.", ha="center", va="center")
            else:
                if speaker_order:
                    valid = valid[valid["speaker"].isin(speaker_order)]

                sns.violinplot(data=valid, x="speaker", y="duration", order=speaker_order, inner="quartile", cut=0, ax=axes[0], palette="Blues")
                axes[0].set_title("Distribution de la durée par speaker", fontweight="bold")
                axes[0].set_xlabel("Speaker")
                axes[0].set_ylabel("Durée (s)")
                axes[0].tick_params(axis="x", rotation=35)

                sns.violinplot(data=valid, x="speaker", y="speech_ratio", order=speaker_order, inner="quartile", cut=0, ax=axes[1], palette="Greens")
                axes[1].set_title("Distribution du speech ratio par speaker", fontweight="bold")
                axes[1].set_xlabel("Speaker")
                axes[1].set_ylabel("Speech ratio")
                axes[1].tick_params(axis="x", rotation=35)

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer

    def _add_page_header_footer(self, canvas_obj, doc) -> None:
        canvas_obj.saveState()
        width, height = doc.pagesize
        canvas_obj.setStrokeColor(colors.HexColor("#d0d7de"))
        canvas_obj.setLineWidth(0.5)
        canvas_obj.line(doc.leftMargin, height - 0.55 * inch, width - doc.rightMargin, height - 0.55 * inch)
        canvas_obj.line(doc.leftMargin, 0.55 * inch, width - doc.rightMargin, 0.55 * inch)
        canvas_obj.setFillColor(colors.HexColor("#4b5563"))
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.drawString(doc.leftMargin, height - 0.45 * inch, "EDA audio dysarthrie")
        canvas_obj.drawRightString(width - doc.rightMargin, 0.38 * inch, f"Page {canvas_obj.getPageNumber()}")
        canvas_obj.restoreState()

    def generate_eda_pdf(self, output_pdf_path: str) -> None:
        if self.df is None or len(self.df) == 0:
            raise ValueError("Aucune donnée à analyser")

        print("📄 Génération du rapport PDF...")

        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=A4,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.95 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#123b66"),
            spaceAfter=10,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        )
        subtitle_style = ParagraphStyle(
            "CustomSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceAfter=8,
        )
        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=15,
            textColor=colors.HexColor("#1f4e79"),
            spaceAfter=10,
            fontName="Helvetica-Bold",
        )
        section_style = ParagraphStyle(
            "CustomSection",
            parent=styles["Heading3"],
            fontSize=11,
            textColor=colors.HexColor("#334155"),
            spaceAfter=8,
            fontName="Helvetica-Bold",
        )

        story: List[Any] = []
        story.append(Paragraph("Rapport d'analyse exploratoire des données", title_style))
        story.append(
            Paragraph(
                f"Corpus de voix pathologiques | extraction top_db={self.top_db:.0f} | généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}",
                subtitle_style,
            )
        )
        story.append(Spacer(1, 0.15 * inch))

        story.append(Paragraph("1. Statistiques du dataset", heading_style))
        story.append(Paragraph("Vue d'ensemble des échantillons et qualité d'extraction.", styles["Normal"]))
        story.append(Spacer(1, 0.12 * inch))
        table_stats = self._style_table(Table(self._build_dataset_stats_table(), colWidths=[3.35 * inch, 2.0 * inch]))
        story.append(table_stats)
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("Qualité et résumé des features acoustiques.", section_style))
        table_quality = self._style_table(Table(self._build_acoustic_summary_table(), colWidths=[2.2 * inch, 1.4 * inch, 1.4 * inch, 1.4 * inch, 1.4 * inch]), align="CENTER")
        story.append(table_quality)
        story.append(PageBreak())

        story.append(Paragraph("2. Analyse acoustique globale", heading_style))
        story.append(Paragraph("Distribution globale des variables acoustiques extraites.", styles["Normal"]))
        story.append(Spacer(1, 0.12 * inch))
        histograms = self._create_histograms_chart()
        story.append(Image(histograms, width=6.9 * inch, height=3.0 * inch))
        story.append(Spacer(1, 0.12 * inch))
        story.append(Paragraph("Statistiques synthétiques des indicateurs acoustiques.", section_style))
        acoustic_table = self._style_table(Table(self._build_acoustic_summary_table(), colWidths=[2.2 * inch, 1.4 * inch, 1.4 * inch, 1.4 * inch, 1.4 * inch]), align="CENTER")
        story.append(acoustic_table)
        story.append(PageBreak())

        story.append(Paragraph("3. Analyse temporelle par type", heading_style))
        story.append(Paragraph("Comparaison des distributions de durée et de speech ratio selon le mode d'annotation.", styles["Normal"]))
        story.append(Spacer(1, 0.12 * inch))
        type_boxplots = self._create_type_boxplots_chart()
        story.append(Image(type_boxplots, width=6.9 * inch, height=3.0 * inch))
        story.append(Spacer(1, 0.12 * inch))
        story.append(Paragraph("Synthèse par type.", section_style))
        type_table = self._style_table(Table(self._build_type_summary_table(), colWidths=[1.7 * inch, 1.0 * inch, 1.0 * inch, 1.6 * inch, 1.5 * inch]), align="CENTER")
        story.append(type_table)
        story.append(PageBreak())

        story.append(Paragraph("4. Variabilité inter-locuteurs", heading_style))
        story.append(Paragraph("Les figures ci-dessous mettent en évidence les écarts de débit et de structure temporelle entre speakers.", styles["Normal"]))
        story.append(Spacer(1, 0.12 * inch))
        scatter = self._create_speaker_scatter_chart()
        story.append(Image(scatter, width=6.9 * inch, height=3.7 * inch))
        story.append(Spacer(1, 0.12 * inch))
        story.append(Paragraph("Distribution intra-locuteur sur les speakers les plus représentés.", section_style))
        violin = self._create_speaker_violin_chart()
        story.append(Image(violin, width=6.9 * inch, height=3.4 * inch))
        story.append(Spacer(1, 0.12 * inch))
        story.append(Paragraph("Top locuteurs par nombre d'échantillons.", section_style))
        speaker_table = self._style_table(Table(self._build_speaker_summary_table(), colWidths=[2.0 * inch, 0.9 * inch, 1.55 * inch, 1.55 * inch]), align="CENTER")
        story.append(speaker_table)

        doc.build(story, onFirstPage=self._add_page_header_footer, onLaterPages=self._add_page_header_footer)
        print(f"✅ Rapport PDF généré avec succès: {output_pdf_path}")


def main() -> None:
    json_path = "/home/melissa/gdrive/Cours/ENSEIRB-MATMECA/2A/S8/PFA/data/labels.json"
    output_pdf = "./reports/eda_report.pdf"

    try:
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
        analyzer = DataAnalyzer(json_path, audio_directory="/home/melissa/gdrive/Cours/ENSEIRB-MATMECA/2A/S8/PFA/data")
        analyzer.generate_eda_pdf(output_pdf)
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")


if __name__ == "__main__":
    main()
