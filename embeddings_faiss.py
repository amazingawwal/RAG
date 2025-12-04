import sys, os, json
import numpy as np
import faiss

"""
Usage:
  python embeddings_faiss.py index <payload.json> <indexdir>
  python embeddings_faiss.py query <query_json> <indexdir>
"""

mode = sys.argv[1]
index_dir = sys.argv[-1]

index_file = os.path.join(index_dir, "faiss.index")
meta_file = os.path.join(index_dir, "meta.json")

if mode == "index":
    payload = sys.argv[2]
    data = json.load(open(payload))

    chunks = data["chunks"]
    embeddings = np.array([c["embedding"] for c in chunks]).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_file)

    # store metadata + text
    meta = [
        {"text": c["text"], "metadata": c["metadata"], "context": c["context"]}
        for c in chunks
    ]
    json.dump(meta, open(meta_file, "w"))

    print(f"Indexed {len(chunks)} chunks.")

elif mode == "query":
    query_json = json.loads(sys.argv[2])
    q_emb = np.array([query_json["embedding"]]).astype("float32")
    k = query_json["k"]
    context = query_json["context"]

    index = faiss.read_index(index_file)
    D, I = index.search(q_emb, k)

    meta = json.load(open(meta_file))

    results = [meta[i] for i in I[0]]
    print(json.dumps(results))
