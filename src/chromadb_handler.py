# src/chromadb_handler.py

import chromadb
from chromadb.utils import embedding_functions
import os
import json
import uuid
import numpy as np

def initialize_chromadb(persist_directory: str = "chroma_db", collection_name: str = "game_logs_collection"):
    """
    Menginisialisasi ChromaDB client dan collection.

    Args:
        persist_directory (str): Direktori untuk menyimpan data ChromaDB.
        collection_name (str): Nama koleksi ChromaDB.

    Returns:
        chromadb.Collection: Objek koleksi ChromaDB.
    """
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        
        try:
            client.delete_collection(name=collection_name)
            print(f"Koleksi '{collection_name}' yang sudah ada dihapus.")
        except Exception:
            pass
            
        collection = client.get_or_create_collection(name=collection_name)
        print(f"Koleksi ChromaDB '{collection_name}' berhasil diinisialisasi.")
        return collection
    except Exception as e:
        print(f"Kesalahan saat menginisialisasi ChromaDB: {e}")
        raise

def store_embeddings_in_chromadb(collection: chromadb.Collection, documents_with_embeddings: list[dict]):
    """
    Menyimpan dokumen dan embedding-nya ke ChromaDB.

    Args:
        collection (chromadb.Collection): Objek koleksi ChromaDB.
        documents_with_embeddings (list[dict]): Daftar dokumen, masing-masing dengan kunci 'embedding_vector'.
    """
    ids = []
    embeddings = []
    metadatas = []
    documents_text = []

    try:
        for doc in documents_with_embeddings:
            if "embedding_vector" not in doc or doc["embedding_vector"] is None or np.all(doc["embedding_vector"] == 0):
                print(f"Peringatan: Dokumen {doc.get('chunk_id', 'unknown')} tidak memiliki 'embedding_vector' yang valid (kosong atau semua nol). Melewatkan.")
                continue
            
            # current_embedding sudah dijamin numpy array dari generate_embedding_for_chunk,
            # dan di main.py sudah diubah ke list sebelum dimasukkan ke documents_with_embeddings
            current_embedding = doc["embedding_vector"] 
            
            ids.append(doc["chunk_id"])
            embeddings.append(current_embedding)

            metadata = {k: v for k, v in doc.items() if k not in ["embedding_vector", "features_sequence"]}
            
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    metadata[key] = str(value)
            
            metadatas.append(metadata)
            
            documents_text.append(json.dumps(doc.get("features_sequence",), indent=2))

        if ids:
            collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text,
                ids=ids
            )
            print(f"{len(ids)} dokumen berhasil disimpan ke ChromaDB.")
        else:
            print("Tidak ada dokumen dengan embedding yang valid untuk disimpan.")
    except Exception as e:
        print(f"Kesalahan saat menyimpan embedding ke ChromaDB: {e}")
        raise