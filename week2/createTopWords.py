import fasttext
import csv

THRESHOLD = 0.8

# Loads the fastText model you created in the previous step (and probably stored in workspace/datasets/fasttext/title_model.bin).
model = fasttext.load_model("/workspace/datasets/fasttext/title_model.bin")

#Iterates through each line of /workspace/datasets/fasttext/top_words.txt (or wherever you stored the top 
# 1,000 title words).
synonyms = []
with open("/workspace/datasets/fasttext/top_words.txt") as file:
    for line in file:
        word = line.strip()
        neighbours = [neighbour[1] for neighbour in  model.get_nearest_neighbors(word) if neighbour[0] >= THRESHOLD]
        neighbours = [word] + neighbours
        synonyms.append(neighbours)

with open("/workspace/datasets/fasttext/synonyms.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(synonyms)
        

