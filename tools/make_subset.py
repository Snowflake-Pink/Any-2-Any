import json

def equidistant_sample_and_rename(filepath, sample_count):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sorted_keys = sorted(data.keys())
    total_entries = len(sorted_keys)

    if sample_count > total_entries or sample_count <= 0:
        raise ValueError()

    step = (total_entries - 1) / (sample_count - 1) if sample_count > 1 else 0
    indices = [int(round(i * step)) for i in range(sample_count)]

    indices = sorted(set(indices))
    print(f"select task: {indices}")
    sampled_keys = [sorted_keys[i] for i in indices]

    new_data = {}
    start_index = 1001
    for idx, key in enumerate(sampled_keys):
        new_key = f"{start_index + idx:05d}"
        new_data[new_key] = data[key]
    
    return new_data

sampled_data = equidistant_sample_and_rename('meta.json', 20)

with open('subset20.json', 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=4)
