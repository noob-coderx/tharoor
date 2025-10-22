import json
import matplotlib.pyplot as plt
import numpy as np

input_file = "data/outputs_task1_IS.jsonl"   
all_weights = []
with open(input_file, "r") as f:
    for line in f:
        row = json.loads(line)
        for cont in row["continuations"]:
            all_weights.extend(cont["normalized_weights"])

bins = np.linspace(0, 1, 11) 
plt.hist(all_weights, bins=bins, edgecolor='black')
plt.xlabel("Normalized Importance Weights")
plt.ylabel("Frequency")
plt.title("Histogram of Normalized Importance Weights (IS)")
plt.xticks(bins)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

