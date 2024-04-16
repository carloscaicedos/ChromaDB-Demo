import json
import argparse
import chromadb
from chromadb.utils import embedding_functions


CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"

parser = argparse.ArgumentParser(description="Ejecutar consulta en ChromaDb.")
parser.add_argument(
    "-c",
    "--collection",
    dest="collection",
    type=str,
    help="Nombre de la colección.",
    required=True,
)
parser.add_argument(
    "-t", "--text", dest="text", type=str, help="Texto a buscar.", required=True
)
parser.add_argument(
    "-w",
    "--where",
    dest="where",
    type=json.loads,
    help="Condición usada para filtrar los resultados.",
)
parser.add_argument("-n", dest="n_results", type=int, help="Número de resultados.")

args = parser.parse_args()


if __name__ == "__main__":
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

    collection = client.get_collection(
        name=args.collection, embedding_function=embedding_func
    )

    query_results = collection.query(
        query_texts=[args.text],
        where=args.where,
        n_results=args.n_results,
        include=["embeddings"],
    )

    print(query_results)
