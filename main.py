import os
from utils.gdrive_access import GD2MilvusManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize global variables
gdm = None
mvc = None

def initialize_manager():
    global gdm, mvc
    if gdm is None or mvc is None:
        folder_id = os.getenv("GD_FOLDER_ID")
        milvus_uri = os.getenv("MILVUS_URI")

        local_folder = "./data"
        collection_name = "candidate_colpali"

        gdm = GD2MilvusManager(
            milvus_uri=milvus_uri,
            collection_name=collection_name,
            login=False,  # True if not wanna auto login
            # device='mps'
        )

        mvc = gdm.milvus

def main():
    initialize_manager()
    topk = int(input("Enter the number of top results to return: "))
    while True:
        query = input("Enter your search query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        result = mvc.search_query(query, topk)

if __name__ == "__main__":
    main()