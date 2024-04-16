import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
)

angulos_txt = ""
axiomas_definiciones_teoremas_txt = ""
axiomas_txt = ""
congruencia_similitud_txt = ""
definicion_txt = ""

with open("data/angulos.txt", "r", encoding="utf-8") as file:
    angulos_txt = file.read()

with open("data/axiomas-definiciones-teoremas.txt", "r", encoding="utf-8") as file:
    axiomas_definiciones_teoremas_txt = file.read()

with open("data/axiomas.txt", "r", encoding="utf-8") as file:
    axiomas_txt = file.read()

with open("data/congruencia-similitud.txt", "r", encoding="utf-8") as file:
    congruencia_similitud_txt = file.read()

with open("data/definicion.txt", "r", encoding="utf-8") as file:
    definicion_txt = file.read()

documents = [
    angulos_txt,
    axiomas_definiciones_teoremas_txt,
    axiomas_txt,
    congruencia_similitud_txt,
    definicion_txt,
]

collection.add(
    documents=documents,
    ids=[f"id{i}" for i in range(len(documents))],
    metadatas=[{"genre": "geometría"} for _ in range(len(documents))],
)
