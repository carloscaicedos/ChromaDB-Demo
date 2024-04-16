import json
import argparse
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"

parser = argparse.ArgumentParser(description="Actualiza una colección de ChromaDB.")
parser.add_argument(
    "-c",
    "--collection",
    dest="collection",
    type=str,
    help="Nombre de la colección.",
    required=True,
)
parser.add_argument(
    "-i",
    "--id",
    dest="id",
    type=str,
    help="Id del documento a actualizar.",
    required=True,
)
parser.add_argument(
    "-d",
    "--document",
    dest="document",
    type=str,
    help="Documento que se agregará a la colección.",
    required=True,
)
parser.add_argument(
    "-m",
    "--metadata",
    dest="metadata",
    type=json.loads,
    help="Metada del documento a agregar.",
)

args = parser.parse_args()


if __name__ == "__main__":
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

    collection = client.get_collection(
        name=args.collection, embedding_function=embedding_func
    )

    collection.update(
        ids=[args.id],
        documents=[args.document] if args.document else args.document,
        metadatas=[args.metadata] if args.metadata else args.metadata,
    )

    print("Documento actualizado exitosamente.")
