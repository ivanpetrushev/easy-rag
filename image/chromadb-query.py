import os
from langchain_chroma import Chroma
from langchain_community.embeddings import BedrockEmbeddings

db = None


def get_embeddings():
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
    )
    return embeddings


def query_chromadb(query: str) -> str:
    """
    Query the ChromaDB with a given query string. 
    The function returns a list of results from the database.
    """
    result = db.similarity_search(query)
    # print(f"Function: query_chromadb({query}) = {result}")
    return result


def load_chromadb():
    global db
    if db is not None:
        return db
    # load ChromaDB only if /chromadb is empty
    if os.listdir("/chromadb"):
        # use ChromaDB from disk
        db = Chroma(persist_directory="/chromadb",
                    embedding_function=get_embeddings())
    else:
        print(f"ChromaDB data not found in /chromadb")


if __name__ == "__main__":
    load_chromadb()
    while True:
        query = input("\n\nEnter query:\n")
        if query == "exit":
            break
        results = query_chromadb(query)
        print("\n\n")
        print("Number of results: ", len(results))
        for result in results:
            # turn all \n in result.page_content to real newline for readability
            result.page_content = result.page_content.replace("\\n", "\n")
            print(f"===> {result}")
