from collections import defaultdict, Counter
from graphviz import Digraph
import os


# https://github.com/harrylclc/RefD-dataset/tree/master/Course
DATASET_PATH = os.path.expanduser("~/Downloads/prereq/datasets/Course/CS_LV.edges")
# Only keep nodes with at least N edges (in + out)
MIN_EDGES = 6


def print_def_dict(d: defaultdict[str, list[str]]):
    for key, values in d.items():
        print(f"{key}: {values}")


def load_edges(path: str) -> defaultdict[str, list[str]]:
    d = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            words = line.strip().split('\t')
            key = words[1]
            value = words[0]
            d[key].append(value)
    return d


def count_edges(d: defaultdict[str, list[str]]) -> Counter:
    # Count connections per node
    edge_count = Counter()
    for key, values in d.items():
        edge_count[key] += len(values)
        for v in values:
            edge_count[v] += 1
    return edge_count


def filter_nodes(edge_count: Counter, min_edges: int) -> set[str]:
    return {node for node, count in edge_count.items() if count >= min_edges}


def build_graph(d: defaultdict[str, list[str]], keep: set[str]) -> Digraph:
    # Build filtered graph
    dot = Digraph()
    dot.attr(ranksep='3')
    dot.attr('node', fontsize='30')
    dot.attr('node', height='1.5')
    for key, values in d.items():
        for value in values:
            if key in keep and value in keep:
                dot.edge(key, value)
    return dot


def main():
    d = load_edges(DATASET_PATH)
    edge_count = count_edges(d)
    keep = filter_nodes(edge_count, MIN_EDGES)
    dot = build_graph(d, keep)
    dot.render('pr_graph', format='png', view=True)


if __name__ == '__main__':
    main()

