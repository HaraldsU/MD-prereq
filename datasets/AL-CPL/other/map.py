import json
import sys

def load_mapping(mapping_file):
    mapping = {}
    with open(mapping_file) as f:
        for line in f:
            line = line.strip()
            if "; " in line:
                en, lv = line.split("; ", 1)
                if lv != "None":
                    mapping[en] = lv
    return mapping

def translate(json_file, mapping_file, output_file=None):
    mapping = load_mapping(mapping_file)

    with open(json_file) as f:
        data = json.load(f)

    for obj in data:
        obj["concept_A"] = mapping.get(obj["concept_A"], obj["concept_A"])
        obj["concept_B"] = mapping.get(obj["concept_B"], obj["concept_B"])

    out = output_file or json_file
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Translated {len(data)} entries -> {out}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python translate.py data.json mapping.txt [output.json]")
        sys.exit(1)
    json_file = sys.argv[1]
    mapping_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    translate(json_file, mapping_file, output_file)
