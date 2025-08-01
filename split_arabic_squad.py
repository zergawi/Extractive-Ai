import json
import random

# اسم ملف Arabic-SQuAD الأصلي
INPUT_FILE = "Arabic-SQuAD.json"

# أسماء الملفات الناتجة
TRAIN_OUTPUT = "train-v2.0-ar.json"
DEV_OUTPUT = "dev-v2.0-ar.json"

# نسبة التقسيم
train_ratio = 0.9

# تحميل البيانات
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# التأكد من وجود المفتاح "data"
all_data = data["data"]

# خلط البيانات عشوائيًا
random.shuffle(all_data)

# تقسيم
split_index = int(len(all_data) * train_ratio)
train_data = all_data[:split_index]
dev_data = all_data[split_index:]

# بناء الشكل المطلوب للملفين الناتجين
train_json = {
    "version": "2.0",
    "data": train_data
}

dev_json = {
    "version": "2.0",
    "data": dev_data
}

# حفظ الملفات
with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(train_json, f, ensure_ascii=False, indent=2)

with open(DEV_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(dev_json, f, ensure_ascii=False, indent=2)

print("✅ تم إنشاء:")
print(f"- {TRAIN_OUTPUT} ← ({len(train_data)} مثال)")
print(f"- {DEV_OUTPUT} ← ({len(dev_data)} مثال)")
