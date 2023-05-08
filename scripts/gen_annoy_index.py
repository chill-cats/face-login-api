from pathlib import Path
import torch
from annoy import AnnoyIndex
import pickle

embedding_path = Path("../embeddings").glob("*")
median_embeddings = []
person = []
for file in embedding_path:
    if not file.name.endswith(".pt"):
        continue
    median_embeddings.append(torch.load(file))
    person_name = file.with_suffix("")
    person.append(person_name.name)

annoy = AnnoyIndex(512, "euclidean")
for index, embedding in enumerate(median_embeddings):
    annoy.add_item(index, embedding)

annoy.build(10, n_jobs=4)
annoy.save("index.ann")

with open("person_name", "wb") as file:
    pickle.dump(person, file=file)
