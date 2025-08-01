import json
import pandas as pd
from datasets import Dataset, DatasetDict

def load_squad(path):
    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)["data"]

    contexts, questions, answers = [], [], []
    for article in raw_data:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                for answer in qa["answers"]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append({
                        "text": answer["text"],
                        "answer_start": answer["answer_start"]
                    })

    return pd.DataFrame({
        "context": contexts,
        "question": questions,
        "answers": answers
    })

# تحميل ملفاتك
train_df = load_squad("train-multilang.json")
dev_df = load_squad("dev-multilang.json")

# تحويلها إلى DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(dev_df)
})

# حفظها بصيغة HuggingFace
dataset.save_to_disk("multilang_dataset")

print("✅ تم تجهيز البيانات وتخزينها في multilang_dataset")
