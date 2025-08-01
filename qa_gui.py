import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QLineEdit, QTextEdit, QPushButton, QMessageBox, QProgressBar
)
from PyQt5.QtCore import QThread, pyqtSignal
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


class ModelLoader(QThread):
    """خيط منفصل لتحميل النموذج"""
    model_loaded = pyqtSignal(object, object)  # tokenizer, model
    error_occurred = pyqtSignal(str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForQuestionAnswering.from_pretrained(self.model_path)
            self.model_loaded.emit(tokenizer, model)
        except Exception as e:
            self.error_occurred.emit(f"خطأ في تحميل النموذج: {str(e)}")


class QAGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("اختبار نموذج الإجابة على الأسئلة")
        self.setGeometry(100, 100, 700, 500)

        # متغيرات النموذج
        self.tokenizer = None
        self.model = None
        self.model_path = "qa_model/checkpoint-24075"

        # إعداد واجهة المستخدم
        self.setup_ui()

        # تحميل النموذج
        self.load_model()

    def setup_ui(self):
        """إعداد واجهة المستخدم"""
        layout = QVBoxLayout()

        # شريط التقدم
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # حالة النموذج
        self.status_label = QLabel("جاري تحميل النموذج...")
        layout.addWidget(self.status_label)

        # إدخال الفقرة
        layout.addWidget(QLabel("الفقرة:"))
        self.context_input = QTextEdit()
        self.context_input.setPlaceholderText("أدخل الفقرة هنا...")
        self.context_input.setMaximumHeight(150)
        layout.addWidget(self.context_input)

        # إدخال السؤال
        layout.addWidget(QLabel("السؤال:"))
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("أدخل السؤال هنا...")
        self.question_input.returnPressed.connect(self.answer_question)
        layout.addWidget(self.question_input)

        # زر الإجابة
        self.answer_button = QPushButton("تشغيل النموذج")
        self.answer_button.clicked.connect(self.answer_question)
        self.answer_button.setEnabled(False)
        layout.addWidget(self.answer_button)

        # عرض الإجابة
        layout.addWidget(QLabel("الإجابة:"))
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        self.answer_output.setMaximumHeight(100)
        layout.addWidget(self.answer_output)

        # معلومات إضافية
        layout.addWidget(QLabel("تفاصيل:"))
        self.details_output = QTextEdit()
        self.details_output.setReadOnly(True)
        self.details_output.setMaximumHeight(80)
        layout.addWidget(self.details_output)

        self.setLayout(layout)

    def load_model(self):
        """تحميل النموذج في خيط منفصل"""
        if not os.path.exists(self.model_path):
            self.status_label.setText(f"خطأ: المسار غير موجود: {self.model_path}")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # شريط تقدم غير محدد

        self.model_loader = ModelLoader(self.model_path)
        self.model_loader.model_loaded.connect(self.on_model_loaded)
        self.model_loader.error_occurred.connect(self.on_model_error)
        self.model_loader.start()

    def on_model_loaded(self, tokenizer, model):
        """عند تحميل النموذج بنجاح"""
        self.tokenizer = tokenizer
        self.model = model
        self.status_label.setText("النموذج جاهز للاستخدام")
        self.progress_bar.setVisible(False)
        self.answer_button.setEnabled(True)

    def on_model_error(self, error_msg):
        """عند حدوث خطأ في تحميل النموذج"""
        self.status_label.setText(error_msg)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "خطأ", error_msg)

    def answer_question(self):
        """معالجة السؤال والحصول على الإجابة"""
        if not self.tokenizer or not self.model:
            QMessageBox.warning(self, "خطأ", "النموذج غير محمل بعد.")
            return

        context = self.context_input.toPlainText().strip()
        question = self.question_input.text().strip()

        if not context or not question:
            QMessageBox.warning(self, "خطأ", "الرجاء إدخال الفقرة والسؤال.")
            return

        try:
            # تعطيل الزر أثناء المعالجة
            self.answer_button.setEnabled(False)
            self.answer_button.setText("جاري المعالجة...")

            # ترميز المدخلات
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # التنبؤ
            with torch.no_grad():
                outputs = self.model(**inputs)

            # استخراج الإجابة
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1

            # التحقق من صحة المؤشرات
            if answer_end <= answer_start:
                answer = "لم يتم العثور على إجابة واضحة"
                confidence = 0.0
            else:
                # استخراج الإجابة
                answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

                # حساب الثقة
                start_confidence = torch.softmax(outputs.start_logits, dim=1)[0][answer_start].item()
                end_confidence = torch.softmax(outputs.end_logits, dim=1)[0][answer_end - 1].item()
                confidence = (start_confidence + end_confidence) / 2

            # تنظيف الإجابة
            if answer.strip():
                self.answer_output.setPlainText(answer.strip())
            else:
                self.answer_output.setPlainText("لم يتم العثور على إجابة")

            # عرض التفاصيل
            details = f"موضع البداية: {answer_start.item()}\n"
            details += f"موضع النهاية: {answer_end.item() - 1}\n"
            details += f"مستوى الثقة: {confidence:.2%}"
            self.details_output.setPlainText(details)

        except Exception as e:
            QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء المعالجة:\n{str(e)}")
            self.answer_output.setPlainText("حدث خطأ أثناء المعالجة")

        finally:
            # إعادة تفعيل الزر
            self.answer_button.setEnabled(True)
            self.answer_button.setText("تشغيل النموذج")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setLayoutDirection(2)  # لدعم الكتابة من اليمين لليسار
    window = QAGUI()
    window.show()
    sys.exit(app.exec_())