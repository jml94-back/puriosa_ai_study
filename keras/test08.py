import itertools

node_candidates = [16, 32, 64]

node_combinations = list(itertools.product(node_candidates, repeat=5))

print(node_combinations)