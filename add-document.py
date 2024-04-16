import argparse
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"

parser = argparse.ArgumentParser(description="Agregar una colección a ChromaDB.")
parser.add_argument(
    "-c", "--collection", dest="collection", type=str, help="Nombre de la colección."
)
parser.add_argument(
    "-d",
    "--document",
    dest="document",
    type=str,
    help="Documento que se agregará a la colección.",
)
parser.add_argument(
    "-m",
    "--metadata",
    dest="metadata",
    type=str,
    help="Metada del documento a agregar.",
)

args = parser.parse_args()


def get_next_id(ids_list: list[str]) -> str:
    sorted_list = sorted(ids_list)
    last_id = int(sorted_list[-1].replace("id", ""))
    next_id = last_id + 1

    return f"id{next_id}"


if __name__ == "__main__":
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

    collection = client.get_collection(
        name=args.collection, embedding_function=embedding_func
    )

    results = collection.get(include=[])

    next_id = get_next_id(results["ids"])

    collection.add(
        ids=[next_id], documents=[args.document], metadatas=[{"genre": args.metadata}]
    )

    print('Documento agregado exitosamente.')
