# Machine Learning Project | مشروع التعلم الآلي

[العربية](#العربية) | [English](#english)

---

## العربية

هذا المشروع هو تطبيق للتعلم الآلي مبني بـ Python، يستخدم أطر عمل التعلم العميق مثل PyTorch و TensorFlow لتدريب وتطوير النماذج.

### المتطلبات الأساسية

قبل تشغيل هذا المشروع، تأكد من أن لديك ما يلي مُثبت:

- **Python 3.8.10**: حمل من [python.org](https://www.python.org/downloads/release/python-3810/)
- **PyCharm IDE** (مُوصى به): حمل من [JetBrains PyCharm](https://www.jetbrains.com/pycharm/download/?section=windows)

### التثبيت

#### 1. استنساخ المستودع

```bash
git clone <رابط-المستودع-الخاص-بك>
cd <مجلد-المشروع>
```

#### 2. إنشاء بيئة افتراضية (مُوصى به)

```bash
python -m venv venv
source venv/bin/activate  # في Windows: venv\Scripts\activate
```

#### 3. ترقية pip

```bash
pip install --upgrade pip
```

#### 4. تثبيت التبعيات

```bash
pip install -r requirements.txt
```

### التبعيات

يستخدم هذا المشروع المكتبات الرئيسية التالية:

- **PyTorch**: إطار عمل التعلم العميق مع دعم CUDA
- **Transformers**: مكتبة Hugging Face لنماذج معالجة اللغة الطبيعية
- **TensorFlow**: إطار عمل بديل للتعلم العميق
- **Datasets & Evaluate**: أدوات Hugging Face للبيانات والتقييم
- **PyQt5**: إطار عمل للواجهة الرسومية
- **TensorBoard**: أداة مراقبة التدريب

### تشغيل المشروع

#### مراقبة التدريب مع TensorBoard

لمراقبة تقدم تدريب النموذج:

```bash
tensorboard --logdir=./qa_model/logs
```

ثم افتح المتصفح وانتقل إلى `http://localhost:6006`

#### تشغيل التطبيق

```bash
python main.py  # استبدل باسم الملف الرئيسي
```

### هيكل المشروع

```
مشروعك/
├── README.md
├── requirements.txt
├── qa_model/
│   └── logs/          # مجلد سجلات TensorBoard
├── main.py            # ملف التطبيق الرئيسي
└── ...
```

### الميزات

- تدريب نماذج التعلم العميق مع PyTorch/TensorFlow
- قدرات معالجة اللغة الطبيعية مع Transformers
- واجهة رسومية مبنية بـ PyQt5
- مراقبة التدريب مع TensorBoard
- تقييم النماذج ومعالجة البيانات

### دعم GPU

هذا المشروع مُكوَّن لاستخدام CUDA 11.8 لتسريع GPU. تأكد من أن لديك:

- بطاقة NVIDIA GPU مع دعم CUDA 11.8
- تعريفات NVIDIA المناسبة مُثبتة

### حل المشكلات

#### المشكلات الشائعة

1. **توافق CUDA**: تأكد من أن تعريفات GPU تدعم CUDA 11.8
2. **مشكلات الذاكرة**: قلل حجم الـ batch إذا واجهت خطأ نفاد الذاكرة
3. **تعارض التبعيات**: استخدم البيئة الافتراضية لعزل التبعيات

#### توافق الإصدارات

- TensorFlow: 2.10.x
- TensorFlow-text: 2.10.0
- Protobuf: 3.20.x

### المساهمة

1. اعمل Fork للمستودع
2. أنشئ فرع للميزة الجديدة
3. اعمل التغييرات المطلوبة
4. أرسل Pull Request

### الترخيص

[أضف معلومات الترخيص هنا]

### الدعم

للمشكلات والأسئلة، يرجى فتح issue في مستودع GitHub.

---

## English

This project is a machine learning application built with Python, utilizing deep learning frameworks like PyTorch and TensorFlow for model training and development.

### Prerequisites

Before running this project, make sure you have the following installed:

- **Python 3.8.10**: Download from [python.org](https://www.python.org/downloads/release/python-3810/)
- **PyCharm IDE** (recommended): Download from [JetBrains PyCharm](https://www.jetbrains.com/pycharm/download/?section=windows)

### Installation

#### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-directory>
```

#### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Upgrade pip

```bash
pip install --upgrade pip
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

This project uses the following main libraries:

- **PyTorch**: Deep learning framework with CUDA support
- **Transformers**: Hugging Face transformers library for NLP models
- **TensorFlow**: Alternative deep learning framework
- **Datasets & Evaluate**: Hugging Face datasets and evaluation tools
- **PyQt5**: GUI framework for desktop applications
- **TensorBoard**: Visualization tool for training metrics

### Running the Project

#### Training Monitoring with TensorBoard

To monitor your model training progress:

```bash
tensorboard --logdir=./qa_model/logs
```

Then open your browser and navigate to `http://localhost:6006`

#### Running the Application

```bash
python main.py  # Replace with your main script name
```

### Project Structure

```
your-project/
├── README.md
├── requirements.txt
├── qa_model/
│   └── logs/          # TensorBoard logs directory
├── main.py            # Main application file
└── ...
```

### Features

- Deep learning model training with PyTorch/TensorFlow
- Natural Language Processing capabilities with Transformers
- GUI interface built with PyQt5
- Training visualization with TensorBoard
- Model evaluation and datasets handling

### GPU Support

This project is configured to use CUDA 11.8 for GPU acceleration. Make sure you have:

- NVIDIA GPU with CUDA 11.8 support
- Appropriate NVIDIA drivers installed

### Troubleshooting

#### Common Issues

1. **CUDA compatibility**: Ensure your GPU drivers support CUDA 11.8
2. **Memory issues**: Reduce batch size if you encounter out-of-memory errors
3. **Dependencies conflicts**: Use virtual environment to isolate dependencies

#### Version Compatibility

- TensorFlow: 2.10.x
- TensorFlow-text: 2.10.0
- Protobuf: 3.20.x

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### License

[Add your license information here]

### Support

For issues and questions, please open an issue in the GitHub repository.
