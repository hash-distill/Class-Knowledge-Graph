import os
import math

dataset_path = '/mnt/Data4/24zhs/Class-Knowledge-Graph/SCB_yolo_dataset_13cls/labels'
subsets = ['train', 'val']
errors = []
total_count = 0

for subset in subsets:
    folder = os.path.join(dataset_path, subset)
    if not os.path.exists(folder):
        continue
    for filename in os.listdir(folder):
        if not filename.endswith('.txt'): continue
        with open(os.path.join(folder, filename), 'r', encoding='utf-8', errors='ignore') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                total_count += 1
                p = line.split()
                if len(p) != 5:
                    errors.append(f"{filename}:{idx+1}: Not 5 cols: {line}")
                    continue
                try:
                    c_f = float(p[0])
                    if not c_f.is_integer() or not (0 <= int(c_f) <= 12):
                        errors.append(f"{filename}:{idx+1}: Class error: {line}")
                    coords = [float(x) for x in p[1:]]
                    if any(math.isnan(x) or math.isinf(x) for x in coords):
                        errors.append(f"{filename}:{idx+1}: NaN/Inf: {line}")
                    elif any(x < 0 or x > 1 for x in coords[:2]) or any(x < 0 or x > 1 for x in coords[2:]) or any(x <= 0 for x in coords[2:]):
                        errors.append(f"{filename}:{idx+1}: Bounds/Size error: {line}")
                except:
                    errors.append(f"{filename}:{idx+1}: Parse error: {line}")

if not errors: print("标签格式检查通过")
else:
    print(f"Errors: {len(errors)}/{total_count}")
    for e in errors[:20]: print(e)
