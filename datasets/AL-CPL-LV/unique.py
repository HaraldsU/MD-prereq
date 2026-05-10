import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

concepts = set()
for item in data:
    concepts.add(item["concept_A"])
    concepts.add(item["concept_B"])

print(f"Unique concepts: {len(concepts)}")


