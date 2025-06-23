import json

# TEST
# truth = ["sénat", "loi", "1931", "740", "8", "9", "10", "p. 8", "p.9"]
# predicted = ["semat", "l0i", "1931", "74O", "B", "p.9"]

def extract_all_values(json_obj):
    all_values = []

    def recurse(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                recurse(value)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
        else:
            all_values.append(str(obj))
    recurse(json_obj)
    return all_values

def make_columns_unique(columns):
    counts = {}
    new_columns = []
    for col in columns:
        if col in counts:
            counts[col] += 1
            new_columns.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 1
            new_columns.append(col)
    return new_columns

json_obj = []
all_values = ["", ""]

# VT
try:
    with open('json/truth.json', 'r', encoding='utf-8') as file1:
        json_obj.append(json.load(file1))
    all_values[0] = extract_all_values(json_obj[0])
except Exception as e:
    print(f"Error loading file1: {e}")

# PREDICTED
try:
    with open('json/predicted.json', 'r', encoding='utf-8') as file2:
        json_obj.append(json.load(file2))
    all_values[1] = extract_all_values(json_obj[1])
except Exception as e:
    print(f"Error loading file2: {e}")

truth= all_values[0]
predicted = all_values[1]