import json
import csv
from pathlib import Path

def process(file_path, type):
    new = []

    with open(file_path, 'r') as f:
        data = json.load(f)

        for row in data:
            new.append({
                'A': row['concept_A'],  
                'B': row['concept_B']
            })
            # print(row)

    write_csv(new, type)

def write_csv(obj, type):
    with open(type + '.csv', 'w') as f:
        writer = csv.writer(f, delimiter = '\t')

        for row in obj:
            writer.writerow([row['A'], row['B']])

# path = Path('~/Downloads/prereq/datasets/AL-CPL/data_mining_full.json').expanduser()
# path = Path('~/Downloads/prereq/datasets/AL-CPL/geometry_full.json').expanduser()
# path = Path('~/Downloads/prereq/datasets/AL-CPL/physics_full.json').expanduser()
# path = Path('~/Downloads/prereq/datasets/AL-CPL/precalculus_full.json').expanduser()

# path = Path('~/Downloads/prereq/datasets/AL-CPL-LV/data_mining_full_lv.json').expanduser()
path1 = Path('~/Downloads/prereq/datasets/AL-CPL-LV/geometry_full_lv.json').expanduser()
path2 = Path('~/Downloads/prereq/datasets/AL-CPL-LV/physics_full_lv.json').expanduser()
path3 = Path('~/Downloads/prereq/datasets/AL-CPL-LV/precalculus_full_lv.json').expanduser()
paths = [path1, path2, path3]

# ds = 'data_mining'
ds1 = 'geometry'
ds2 = 'physics'
ds3 = 'precalculus'
dss = [ds1, ds2, ds3]

for path, ds in zip(paths, dss):
    process(path, ds)

