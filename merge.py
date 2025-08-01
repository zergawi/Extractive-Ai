import json

def merge_files(file1, file2, output_file):
    with open(file1, 'r', encoding='utf8') as f1, open(file2, 'r', encoding='utf8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    merged = {
        "version": "multi-lang",
        "data": data1["data"] + data2["data"]
    }

    with open(output_file, 'w', encoding='utf8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ تم إنشاء {output_file} ← بعد دمج {len(data1['data'])} + {len(data2['data'])} مثال")

# دمج التدريب
merge_files("train-v1.1.json", "train-v2.0-ar.json", "train-multilang.json")

# دمج التحقق
merge_files("dev-v1.1.json", "dev-v2.0-ar.json", "dev-multilang.json")
