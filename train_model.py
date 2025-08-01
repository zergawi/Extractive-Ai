import os
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from datasets import load_from_disk
from evaluate import load
import torch

# إعداد نظام التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """تكوين النموذج والتدريب"""
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "bert-base-multilingual-cased"))
    dataset_path: str = field(default_factory=lambda: os.getenv("DATASET_PATH", "multilang_dataset"))
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "./qa_model"))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "16")))
    learning_rate: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "2e-5")))
    num_epochs: int = field(default_factory=lambda: int(os.getenv("NUM_EPOCHS", "3")))
    max_length: int = field(default_factory=lambda: int(os.getenv("MAX_LENGTH", "384")))
    doc_stride: int = field(default_factory=lambda: int(os.getenv("DOC_STRIDE", "128")))
    warmup_steps: int = field(default_factory=lambda: int(os.getenv("WARMUP_STEPS", "500")))
    weight_decay: float = field(default_factory=lambda: float(os.getenv("WEIGHT_DECAY", "0.01")))
    gradient_accumulation_steps: int = field(default_factory=lambda: int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1")))

    def save_config(self, path: str):
        """حفظ التكوين في ملف JSON"""
        with open(os.path.join(path, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)


class QADataProcessor:
    """معالج البيانات للأسئلة والأجوبة"""

    def __init__(self, tokenizer, config: ModelConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.errors_count = 0

    def validate_example(self, example: Dict) -> bool:
        """التحقق من صحة المثال"""
        required_keys = ["question", "context", "answers"]

        # التحقق من وجود المفاتيح المطلوبة
        if not all(key in example for key in required_keys):
            logger.warning(f"مثال ناقص: {example.get('id', 'unknown')}")
            return False

        # التحقق من أن الإجابة ليست فارغة
        if not example["answers"].get("text") or not example["answers"].get("answer_start"):
            logger.warning(f"إجابة فارغة في المثال: {example.get('id', 'unknown')}")
            return False

        # التحقق من أن الإجابة موجودة في السياق
        answer_text = example["answers"]["text"][0] if isinstance(example["answers"]["text"], list) else \
        example["answers"]["text"]
        if answer_text not in example["context"]:
            logger.warning(f"الإجابة غير موجودة في السياق: {example.get('id', 'unknown')}")
            return False

        return True

    def preprocess_example(self, example: Dict) -> Dict:
        """معالجة مثال واحد من البيانات"""
        try:
            # التحقق من صحة البيانات
            if not self.validate_example(example):
                self.errors_count += 1
                # إرجاع مثال فارغ بدلاً من None
                return {
                    "input_ids": [0] * self.config.max_length,
                    "attention_mask": [0] * self.config.max_length,
                    "start_positions": 0,
                    "end_positions": 0,
                }

            # استخراج البيانات
            question = example["question"]
            context = example["context"]
            answers = example["answers"]

            # معالجة الإجابة (قد تكون قائمة أو نص مفرد)
            if isinstance(answers["text"], list):
                answer_text = answers["text"][0]
                answer_start = answers["answer_start"][0] if isinstance(answers["answer_start"], list) else answers[
                    "answer_start"]
            else:
                answer_text = answers["text"]
                answer_start = answers["answer_start"]

            # تطبيق التوكنايزر
            inputs = self.tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=self.config.max_length,
                stride=self.config.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            # إزالة offset_mapping من المدخلات
            offset_mapping = inputs.pop("offset_mapping")

            # معالجة كل جزء من النص (في حالة التقسيم)
            sample_map = inputs.pop("overflow_to_sample_mapping")

            # البحث عن موقع الإجابة في التوكنات
            start_positions = []
            end_positions = []

            for i, offsets in enumerate(offset_mapping):
                input_ids = inputs["input_ids"][i]
                cls_index = input_ids.index(self.tokenizer.cls_token_id)
                sequence_ids = inputs.sequence_ids(i)

                # البحث عن بداية ونهاية السياق
                context_start = cls_index + 1
                context_end = len(sequence_ids) - 1
                while context_end >= context_start and sequence_ids[context_end] != 1:
                    context_end -= 1

                # التحقق من وجود الإجابة في هذا الجزء
                answer_end = answer_start + len(answer_text)

                # البحث عن موقع التوكنات
                token_start_index = 0
                token_end_index = 0
                found_answer = False

                for idx in range(context_start, context_end + 1):
                    if idx < len(offsets) and offsets[idx] is not None:
                        if offsets[idx][0] <= answer_start <= offsets[idx][1]:
                            token_start_index = idx
                            found_answer = True
                        if offsets[idx][0] <= answer_end <= offsets[idx][1]:
                            token_end_index = idx

                if found_answer:
                    start_positions.append(token_start_index)
                    end_positions.append(token_end_index)
                else:
                    # إذا لم نجد الإجابة، نضع 0
                    start_positions.append(0)
                    end_positions.append(0)

            # إرجاع أول جزء فقط (يمكن تحسين هذا لمعالجة أجزاء متعددة)
            return {
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "start_positions": start_positions[0] if start_positions else 0,
                "end_positions": end_positions[0] if end_positions else 0,
            }

        except Exception as e:
            logger.error(f"خطأ في معالجة المثال: {e}")
            self.errors_count += 1
            # إرجاع مثال فارغ بدلاً من None
            return {
                "input_ids": [0] * self.config.max_length,
                "attention_mask": [0] * self.config.max_length,
                "start_positions": 0,
                "end_positions": 0,
            }


class QATrainer:
    """مدرب نموذج الأسئلة والأجوبة"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.metric = load("squad")

    def load_model_and_tokenizer(self):
        """تحميل النموذج والتوكنايزر"""
        try:
            logger.info(f"تحميل النموذج: {self.config.model_name}")
            self.tokenizer = BertTokenizerFast.from_pretrained(self.config.model_name)
            self.model = BertForQuestionAnswering.from_pretrained(self.config.model_name)

            # التحقق من وجود GPU
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info(f"استخدام GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("لا يوجد GPU متاح، سيتم استخدام CPU")

        except Exception as e:
            logger.error(f"خطأ في تحميل النموذج: {e}")
            raise

    def load_dataset(self):
        """تحميل مجموعة البيانات"""
        try:
            logger.info(f"تحميل البيانات من: {self.config.dataset_path}")
            self.dataset = load_from_disk(self.config.dataset_path)

            logger.info(f"عدد عينات التدريب: {len(self.dataset['train'])}")
            logger.info(f"عدد عينات التحقق: {len(self.dataset['validation'])}")

        except Exception as e:
            logger.error(f"خطأ في تحميل البيانات: {e}")
            raise

    def preprocess_dataset(self):
        """معالجة مجموعة البيانات"""
        processor = QADataProcessor(self.tokenizer, self.config)

        logger.info("بدء معالجة البيانات...")

        # معالجة البيانات
        self.tokenized_datasets = self.dataset.map(
            processor.preprocess_example,
            remove_columns=self.dataset["train"].column_names,
            load_from_cache_file=False,
            desc="معالجة البيانات"
        )

        # إزالة الأمثلة الفارغة (التي لها start_positions = 0 و end_positions = 0)
        def is_valid_example(example):
            # نعتبر المثال صالحاً إذا كان له موقع بداية ونهاية غير صفر
            # أو إذا كان الموقع صفر لكن المثال ليس فارغاً تماماً
            return not all(token == 0 for token in example["input_ids"])

        self.tokenized_datasets = self.tokenized_datasets.filter(is_valid_example)

        logger.info(f"تمت معالجة البيانات. عدد الأخطاء: {processor.errors_count}")
        logger.info(f"عدد العينات بعد المعالجة - التدريب: {len(self.tokenized_datasets['train'])}")
        logger.info(f"عدد العينات بعد المعالجة - التحقق: {len(self.tokenized_datasets['validation'])}")

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """حساب معايير الأداء"""
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # استخراج مواقع البداية والنهاية
        start_preds = np.argmax(predictions[0], axis=1)
        end_preds = np.argmax(predictions[1], axis=1)

        # حساب الدقة البسيطة
        start_accuracy = np.mean(start_preds == label_ids[0])
        end_accuracy = np.mean(end_preds == label_ids[1])
        exact_match = np.mean((start_preds == label_ids[0]) & (end_preds == label_ids[1]))

        return {
            "start_accuracy": start_accuracy,
            "end_accuracy": end_accuracy,
            "exact_match": exact_match,
        }

    def get_training_args(self) -> TrainingArguments:
        """إنشاء معاملات التدريب"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            evaluation_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            logging_steps=100,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="exact_match",
            greater_is_better=True,
            push_to_hub=False,
            report_to=["tensorboard"],
            fp16=torch.cuda.is_available(),  # استخدام fp16 إذا كان GPU متاح
            dataloader_num_workers=4,
            remove_unused_columns=True,
        )

    def train(self):
        """تنفيذ التدريب"""
        try:
            # تحميل المكونات
            self.load_model_and_tokenizer()
            self.load_dataset()
            self.preprocess_dataset()

            # إنشاء المدرب
            training_args = self.get_training_args()

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.tokenized_datasets["train"],
                eval_dataset=self.tokenized_datasets["validation"],
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
            )

            # بدء التدريب
            logger.info("بدء التدريب...")
            train_result = trainer.train()

            # حفظ النموذج النهائي
            logger.info("حفظ النموذج...")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)

            # حفظ التكوين
            self.config.save_config(self.config.output_dir)

            # حفظ نتائج التدريب
            with open(os.path.join(self.config.output_dir, "train_results.json"), "w") as f:
                json.dump(train_result.metrics, f, indent=2)

            # التقييم النهائي
            logger.info("التقييم النهائي...")
            eval_result = trainer.evaluate()

            with open(os.path.join(self.config.output_dir, "eval_results.json"), "w") as f:
                json.dump(eval_result, f, indent=2)

            logger.info(f"اكتمل التدريب! النتائج: {eval_result}")

        except Exception as e:
            logger.error(f"خطأ أثناء التدريب: {e}")
            raise


def main():
    """الدالة الرئيسية"""
    # إنشاء التكوين
    config = ModelConfig()

    # طباعة التكوين
    logger.info("تكوين التدريب:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")

    # إنشاء المدرب وبدء التدريب
    trainer = QATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()