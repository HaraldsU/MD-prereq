import json
import sys

def get_unique_concepts(filepath):
    with open(filepath) as f:
        data = json.load(f)
    concepts = set()
    for obj in data:
        concepts.add(obj["concept_A"])
        concepts.add(obj["concept_B"])
    return sorted(concepts)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data.json"
    for c in get_unique_concepts(path):
        print(c)
