from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import librosa
import numpy as np
import torch
from datasets import Dataset
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments, Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import EarlyStoppingCallback

from evaluate import ModelEvaluator
from inference import SpeechInference


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────────────────────────
# EDA-driven constants
# ──────────────────────────────────────────────────────────────────────────────
# Dataset breakdown (from EDA):
#   Letter  : 921 samples (75 %) — dominant class, used for "letter" mode
#   Special : 168 samples (13.7 %)
#   Digit   :  69 samples  (5.6 %)
#   Sentence:  69 samples  (5.6 %)
#
# Single speaker ("client"), 2 sessions → no speaker-level split needed,
# but session-stratified split is important to avoid session leakage.
#
# Audio quality is high (SNR ~32.7 dB, speech ratio ~0.929) so aggressive
# noise-based augmentation is not required.  However, the small corpus size
# (~1 228 files total, ~783 letter-mode train samples after 15 % val split)
# makes regularisation critical.
#
# Chosen adjustments vs. original code:
#   1. LoRA rank r=16  (r=8 was under-parameterised for a 300 M model on ~800 samples)
#   2. lora_alpha=32   (keeps alpha/r=2, standard practice)
#   3. lora_dropout=0.15  (slightly stronger dropout for small corpus)
#   4. weight_decay=0.01  (L2 regularisation, missing from original)
#   5. warmup_ratio=0.15  (longer warmup — short files → fast steps)
#   6. learning_rate=1e-4 (lower than 3e-4; avoids overshooting on small set)
#   7. epochs=15          (more epochs + early stopping patience=5)
#   8. batch_size=4       (1 228 short files; smaller batch → more gradient steps)
#   9. gradient_accumulation=4  (effective batch = 16, same as original 8×2)
#  10. val split = 20 %   (small dataset — give more signal to validation)
#  11. test_size seed preserved per session when possible
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DataCollatorCTCWithPadding:
	"""Dynamic padding collator for CTC tasks.

	Inputs and labels are padded separately.
	Label padding positions are replaced by -100 so CTC loss ignores them.
	"""

	processor: Wav2Vec2Processor
	padding: Union[bool, str] = True

	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		input_features = [{"input_values": feature["input_values"]} for feature in features]
		label_features = [{"input_ids": feature["labels"]} for feature in features]

		batch = self.processor.pad(
			input_features,
			padding=self.padding,
			return_tensors="pt",
		)

		with self.processor.as_target_processor():
			labels_batch = self.processor.pad(
				label_features,
				padding=self.padding,
				return_tensors="pt",
			)

		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
		batch["labels"] = labels

		allowed_keys = {"input_values", "attention_mask", "labels"}
		return {key: value for key, value in batch.items() if key in allowed_keys}


class Wav2Vec2CTCForwardWrapper(torch.nn.Module):
	"""Forward wrapper that strips PEFT-injected args unsupported by Wav2Vec2ForCTC."""

	def __init__(self, peft_model: torch.nn.Module) -> None:
		super().__init__()
		self.peft_model = peft_model

	def __getattr__(self, name: str):
		try:
			return super().__getattr__(name)
		except AttributeError:
			return getattr(self.peft_model, name)

	def gradient_checkpointing_enable(self, **kwargs):
		return self.peft_model.gradient_checkpointing_enable(**kwargs)

	def gradient_checkpointing_disable(self):
		return self.peft_model.gradient_checkpointing_disable()

	def save_pretrained(self, *args, **kwargs):
		return self.peft_model.save_pretrained(*args, **kwargs)

	def forward(
		self,
		input_values: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		labels: Optional[torch.Tensor] = None,
		**kwargs,
	):
		kwargs.pop("input_ids", None)
		base_model = getattr(getattr(self.peft_model, "base_model", None), "model", None)
		if base_model is None:
			return self.peft_model(
				input_values=input_values,
				attention_mask=attention_mask,
				labels=labels,
				**kwargs,
			)
		return base_model(
			input_values=input_values,
			attention_mask=attention_mask,
			labels=labels,
		)


class FineTuner:
	"""Fine-tune Wav2Vec2 for ASR with LoRA, adapted for a small single-speaker dysarthria corpus."""

	BASE_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
	MODE_TO_TYPES: Dict[str, List[str]] = {
		"letter": ["Letter", "Digit", "Special"],
		"word": ["Word"],
		"sentences": ["Sentences"],
	}

	# ── EDA-driven hyperparameters ───────────────────────────────────────────
	# Small corpus (~800 train samples after split) with a single speaker
	# → lower LR, stronger regularisation, more epochs with early stopping.
	DEFAULT_EPOCHS       = 15       # was 10 — more room for early stopping
	DEFAULT_BATCH_SIZE   = 4        # was 8  — more gradient steps on short files
	DEFAULT_LR           = 1e-4     # was 3e-4 — conservative for small dataset
	DEFAULT_VAL_SPLIT    = 0.20     # was 0.15 — larger val set on small corpus
	GRAD_ACCUM_STEPS     = 4        # effective batch = 4×4 = 16
	WARMUP_RATIO         = 0.15     # was 0.10 — longer warmup, fast steps
	WEIGHT_DECAY         = 0.01     # was 0   — L2 regularisation
	EARLY_STOP_PATIENCE  = 5        # was 3   — more patience, small noisy val
	LORA_R               = 16       # was 8   — more capacity for 300 M model
	LORA_ALPHA           = 32       # was 16  — keep alpha/r = 2
	LORA_DROPOUT         = 0.15     # was 0.10 — stronger dropout, small corpus
	# ─────────────────────────────────────────────────────────────────────────

	def __init__(
		self,
		mode: str,
		train_json_path: str,
		output_dir: str,
		epochs: int = DEFAULT_EPOCHS,
		batch_size: int = DEFAULT_BATCH_SIZE,
		learning_rate: float = DEFAULT_LR,
	) -> None:
		if mode not in {"letter", "word", "sentences"}:
			raise ValueError("mode must be one of: 'letter', 'word', 'sentences'")
		if not os.path.exists(train_json_path):
			raise FileNotFoundError(f"Training JSON not found: {train_json_path}")

		self.mode = mode
		self.train_json_path = train_json_path
		self.output_dir = output_dir
		self.epochs = epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate

		self.train_audio_dir = os.path.dirname(train_json_path)

		self.processor: Optional[Wav2Vec2Processor] = None
		self.model: Optional[Wav2Vec2ForCTC] = None
		self.train_model: Optional[torch.nn.Module] = None
		self.peft_model: Optional[torch.nn.Module] = None

		self.train_dataset: Optional[Dataset] = None
		self.val_dataset: Optional[Dataset] = None
		self.validation_json_path: Optional[str] = None
		self._inference_letter_helper = object.__new__(SpeechInference)

	def _normalize_letter_sequence_from_inference(self, text: str) -> str:
		raw = str(text)
		if raw == "":
			return ""
		if raw.isspace():
			return " "
		tokens = [tok for tok in raw.lower().split() if tok]
		if not tokens:
			return SpeechInference._post_process_letter(self._inference_letter_helper, raw)
		mapped_tokens: List[str] = [
			SpeechInference._post_process_letter(self._inference_letter_helper, token)
			for token in tokens
		]
		return "".join(mapped_tokens)

	def _filter_rows_by_mode(self, rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
		allowed_types = self.MODE_TO_TYPES[self.mode]
		return [row for row in rows if str(row.get("type", "")) in allowed_types]

	def _normalize_transcript(self, value: object) -> str:
		return str(value).strip().lower()

	def _training_transcript(self, label: object, label_type: object) -> str:
		label_text = str(label)
		if str(label_type) == "Special" and label_text.strip() == "":
			return self.processor.tokenizer.word_delimiter_token
		return self._normalize_transcript(label_text)

	def _prepare_dataset(self) -> None:
		"""Load base model/processor and prepare HF datasets from JSON.

		Val split raised to 20 % (EDA: small corpus, single speaker).
		Session metadata is preserved in the split for post-hoc analysis.
		"""
		logger.info("Loading base processor/model: %s", self.BASE_MODEL_NAME)
		self.processor = Wav2Vec2Processor.from_pretrained(self.BASE_MODEL_NAME)
		base_model = Wav2Vec2ForCTC.from_pretrained(
			self.BASE_MODEL_NAME,
            mask_time_prob=0.1,        # Masque 10% du temps
            mask_time_length=10,       # Longueur du masque temporel (adapté aux audios courts)
            mask_feature_prob=0.05,    # Masque 5% des fréquences
            activation_dropout=0.1,    # Dropout standard
            hidden_dropout=0.1,
											  )

		with open(self.train_json_path, "r", encoding="utf-8") as handle:
			raw_rows = json.load(handle)

		filtered_rows = self._filter_rows_by_mode(raw_rows)
		if not filtered_rows:
			raise ValueError(f"No rows found for mode '{self.mode}' in {self.train_json_path}")

		logger.info("Rows after mode filtering: %d", len(filtered_rows))

		records: List[Dict[str, object]] = []
		for row in filtered_rows:
			audio_file = str(row["file"])
			audio_path = audio_file if os.path.isabs(audio_file) else os.path.join(self.train_audio_dir, audio_file)
			records.append(
				{
					"audio": audio_path,
					"transcript": str(self._training_transcript(row["label"], row.get("type", ""))),
					"file": str(audio_path),
					"type": str(row.get("type", "")),
					"session": str(row.get("session", "")),
					"micro": str(row.get("micro", "")),
					"speaker": str(row.get("speaker", "")),
				}
			)

		processed_records: List[Dict[str, object]] = []
		skipped_files = 0

		for record in records:
			try:
				waveform, _ = librosa.load(str(record["audio"]), sr=16000, mono=True)
				input_values = self.processor(waveform, sampling_rate=16000).input_values[0]
				with self.processor.as_target_processor():
					labels = self.processor(str(record["transcript"])).input_ids
				if len(labels) == 0:
					labels = self.processor.tokenizer(self.processor.tokenizer.word_delimiter_token).input_ids

				processed_records.append(
					{
						"input_values": input_values,
						"labels": labels,
						"transcript": str(record["transcript"]),
						"file": str(record["file"]),
						"type": str(record["type"]),
						"session": str(record["session"]),
						"micro": str(record["micro"]),
						"speaker": str(record["speaker"]),
					}
				)
			except Exception as exc:
				skipped_files += 1
				logger.warning("Skipping %s due to preprocessing error: %s", record["audio"], exc)

		if not processed_records:
			raise ValueError("No audio examples could be preprocessed successfully.")

		dataset = Dataset.from_list(processed_records)

		# EDA: 2 sessions → stratify on session to ensure both appear in train+val
		# HuggingFace train_test_split does not support stratify directly,
		# so we do it manually when session info is available.
		sessions = [r["session"] for r in processed_records]
		unique_sessions = list(set(sessions))

		if len(unique_sessions) > 1:
			logger.info("Stratifying split across %d sessions: %s", len(unique_sessions), unique_sessions)
			train_indices, val_indices = [], []
			rng = np.random.default_rng(seed=42)
			for sess in unique_sessions:
				sess_indices = [i for i, s in enumerate(sessions) if s == sess]
				rng.shuffle(sess_indices)
				n_val = max(1, int(len(sess_indices) * self.DEFAULT_VAL_SPLIT))
				val_indices.extend(sess_indices[:n_val])
				train_indices.extend(sess_indices[n_val:])
			self.train_dataset = dataset.select(train_indices)
			self.val_dataset   = dataset.select(val_indices)
		else:
			# Single session (or no session tag) — plain random split
			split = dataset.train_test_split(
				test_size=self.DEFAULT_VAL_SPLIT,
				seed=42,
				shuffle=True,
			)
			self.train_dataset = split["train"]
			self.val_dataset   = split["test"]

		validation_records = self.val_dataset.to_list()

		logger.info(
			"Train samples: %d | Validation samples: %d | Skipped files: %d",
			len(self.train_dataset),
			len(self.val_dataset),
			skipped_files,
		)

		# ── LoRA configuration (EDA-adjusted) ───────────────────────────────
		# r=16 / alpha=32: more representational capacity for a 300 M-param model
		# fine-tuned on ~800 samples.  dropout=0.15 adds regularisation.
		lora_cfg = LoraConfig(
			task_type=TaskType.FEATURE_EXTRACTION,
			inference_mode=False,
			r=self.LORA_R,
			lora_alpha=self.LORA_ALPHA,
			lora_dropout=self.LORA_DROPOUT,
			target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
			bias="none",
		)
		self.peft_model = get_peft_model(base_model, lora_cfg)
		self.peft_model.print_trainable_parameters()
		self.train_model = Wav2Vec2CTCForwardWrapper(self.peft_model)
		self.model = self.peft_model

		os.makedirs(self.output_dir, exist_ok=True)
		self.validation_json_path = os.path.join(self.output_dir, "validation_split.json")
		with open(self.validation_json_path, "w", encoding="utf-8") as handle:
			json.dump(
				[
					{
						"file": r["file"],
						"label": r["transcript"],
						"type": r["type"],
						"speaker": r["speaker"],
						"session": r["session"],
						"micro": r["micro"],
					}
					for r in validation_records
				],
				handle,
				ensure_ascii=False,
				indent=2,
			)

	def _compute_metrics(self, pred) -> Dict[str, float]:
		pred_logits = pred.predictions
		pred_ids = np.argmax(pred_logits, axis=-1)

		label_ids = pred.label_ids
		label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

		pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
		label_str = self.processor.batch_decode(label_ids, group_tokens=False, skip_special_tokens=True)

		if self.mode == "letter":
			pred_str  = [self._normalize_letter_sequence_from_inference(x) for x in pred_str]
			label_str = [self._normalize_letter_sequence_from_inference(x) for x in label_str]

			def _chars_for_wer(text: str) -> str:
				chars = ["<sp>" if ch == " " else ch for ch in text.upper()]
				return " ".join(chars)

			ref_for_wer = [_chars_for_wer(x) for x in label_str]
			hyp_for_wer = [_chars_for_wer(x) for x in pred_str]
			wer = jiwer_wer(ref_for_wer, hyp_for_wer)
		else:
			wer = jiwer_wer(label_str, pred_str)

		cer = jiwer_cer(label_str, pred_str)
		return {"wer": float(wer), "cer": float(cer)}

	def train(self) -> None:
		"""Run fine-tuning, save model, and automatically evaluate on validation split."""
		logger.info("Preparing datasets and LoRA model...")
		self._prepare_dataset()

		assert self.processor   is not None
		assert self.model       is not None
		assert self.train_model is not None
		assert self.train_dataset is not None
		assert self.val_dataset   is not None

		data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

		training_args = TrainingArguments(
			output_dir=self.output_dir,
			# ── batch / accumulation ──────────────────────────────────────
			# EDA: ~800 letter-mode train samples, short files (~0.7 s median).
			# batch=4 + accum=4 → effective batch=16 (same as original 8×2)
			# but gives 2× more gradient-update steps per epoch, which helps
			# convergence on a small dataset.
			per_device_train_batch_size=self.batch_size,        # 4
			per_device_eval_batch_size=self.batch_size,         # 4
			gradient_accumulation_steps=self.GRAD_ACCUM_STEPS,  # 4
			# ── optimiser ────────────────────────────────────────────────
			# Lower LR (1e-4 vs 3e-4) reduces risk of overshooting on the
			# small corpus; weight_decay=0.01 adds L2 regularisation.
			num_train_epochs=self.epochs,                        # 15
			learning_rate=self.learning_rate,                    # 1e-4
			weight_decay=self.WEIGHT_DECAY,                      # 0.01
			warmup_ratio=self.WARMUP_RATIO,                      # 0.15
			# ── logging / checkpointing ───────────────────────────────────
			logging_strategy="steps",
			logging_steps=10,
            evaluation_strategy="steps", # MODIFIÉ : On évalue par pas, pas par époque
            eval_steps=20,               # MODIFIÉ : Évaluation tous les 20 pas d'entraînement
            save_strategy="steps",       # MODIFIÉ : Sauvegarde alignée sur l'évaluation
            save_steps=20,
			fp16=torch.cuda.is_available(),
			gradient_checkpointing=True,
			save_total_limit=3,              # keep best + last 2 for analysis
			load_best_model_at_end=True,
			metric_for_best_model="wer",
			greater_is_better=False,
			report_to=["tensorboard"],
			remove_unused_columns=False,
			dataloader_num_workers=0,
		)

		trainer = Trainer(
			model=self.train_model,
			args=training_args,
			train_dataset=self.train_dataset,
			eval_dataset=self.val_dataset,
			tokenizer=self.processor.feature_extractor,
			data_collator=data_collator,
			compute_metrics=self._compute_metrics,
			# Patience=5: small val set is noisy — avoid stopping too early.
			callbacks=[EarlyStoppingCallback(early_stopping_patience=self.EARLY_STOP_PATIENCE)],
		)

		logger.info("Starting training...")
		trainer.train()

		logger.info("Saving trainer model artifacts to: %s", self.output_dir)
		trainer.save_model(self.output_dir)
		self.processor.save_pretrained(self.output_dir)

		merged_model_dir = os.path.join(self.output_dir, "final_merged_model")
		os.makedirs(merged_model_dir, exist_ok=True)

		logger.info("Merging LoRA adapters into the base model for inference...")
		if isinstance(self.model, PeftModel):
			merged_model = self.model.merge_and_unload()
			merged_model.save_pretrained(merged_model_dir)
		else:
			self.model.save_pretrained(merged_model_dir)
		self.processor.save_pretrained(merged_model_dir)

		if self.validation_json_path is None:
			raise RuntimeError("Validation JSON path is missing; dataset preparation failed.")

		logger.info("Running automatic evaluation on validation set...")
		evaluator = ModelEvaluator(
			json_path=self.validation_json_path,
			model_path=merged_model_dir,
			mode=self.mode,
		)
		metrics = evaluator.run_evaluation()

		pdf_path = os.path.join(self.output_dir, f"evaluation_report_{self.mode}.pdf")
		evaluator.export_results_to_pdf(pdf_path)

		logger.info("Training completed successfully.")
		logger.info("Validation metrics: %s", metrics)
		logger.info("Evaluation PDF generated: %s", pdf_path)


def main() -> None:
	fine_tuner = FineTuner(
		mode="letter",
		train_json_path="/home/melissa/gdrive/Cours/ENSEIRB-MATMECA/2A/S8/PFA/data/labels.json",
		output_dir="/home/melissa/gdrive/Cours/ENSEIRB-MATMECA/2A/S8/PFA/code/inference/training_outputs",
		# All hyperparameters below now use EDA-driven class defaults.
		# Override here only if you want to experiment:
		# epochs=15, batch_size=4, learning_rate=1e-4
	)
	fine_tuner.train()


if __name__ == "__main__":
	main()