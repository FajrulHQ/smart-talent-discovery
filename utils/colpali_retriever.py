from transformers import AutoProcessor, AutoModelForPreTraining
from pymilvus import DataType
from PIL import Image
import numpy as np
import concurrent.futures
from tqdm import tqdm
import fitz

import os
import torch
from torch.utils.data import DataLoader

class MilvusColbertRetriever:
    def __init__(self, milvus_client, collection_name, dim=128):
        # Initialize the retriever with a Milvus client, collection name, and dimensionality of the vector embeddings.
        # If the collection exists, load it.
        self.collection_name = collection_name
        self.client = milvus_client
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.load_collection(collection_name)
        self.dim = dim

    def create_collection(self):
        # Create a new collection in Milvus for storing embeddings.
        # Drop the existing collection if it already exists and define the schema for the collection.
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )
        schema.add_field(field_name="sequence_id", datatype=DataType.INT16)
        schema.add_field(field_name="document_id", datatype=DataType.INT64)
        schema.add_field(field_name="page_number", datatype=DataType.INT64)
        schema.add_field(field_name="document_path", datatype=DataType.VARCHAR, max_length=65535)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )

    def create_index(self):
        # Create an index on the vector field to enable fast similarity search.
        # Releases and drops any existing index before creating a new one with specified parameters.
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(
            collection_name=self.collection_name, index_name="vector"
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="HNSW",  # or any other index type you want
            metric_type="IP",  # or the appropriate metric type
            params={
                "M": 16,
                "efConstruction": 500,
            },  # adjust these parameters as needed
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def create_scalar_index(self):
        # Create a scalar index for the "document_id" field to enable fast lookups by document ID.
        self.client.release_collection(collection_name=self.collection_name)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="document_id",
            index_name="int32_index",
            index_type="INVERTED",  # or any other index type you want
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def search(self, data, topk):
        # Perform a vector search on the collection to find the top-k most similar documents.
        search_params = {"metric_type": "IP", "params": {}}
        results = self.client.search(
            self.collection_name,
            data,
            limit=int(topk),
            output_fields=["vector", "sequence_id", "document_id"],
            search_params=search_params,
        )
        doc_ids = set()
        for r_id in range(len(results)):
            for r in range(len(results[r_id])):
                doc_ids.add(results[r_id][r]["entity"]["document_id"])

        scores = []

        def rerank_single_doc(doc_id, data, client, collection_name):
            # Rerank a single document by retrieving its embeddings and calculating the similarity with the query.
            doc_colbert_vecs = client.query(
                collection_name=collection_name,
                filter=f"doc_id in [{doc_id}]",
                output_fields=["sequence_id", "vector", "document_path"],
                limit=200,
            )
            doc_vecs = np.vstack(
                [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
            )
            score = np.dot(data, doc_vecs.T).max(1).sum()
            return (score, doc_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc_id, data, client, self.collection_name
                ): doc_id
                for doc_id in doc_ids
            }
            for future in concurrent.futures.as_completed(futures):
                score, doc_id = future.result()
                scores.append((score, doc_id))

        scores.sort(key=lambda x: x[0], reverse=True)
        if len(scores) >= topk:
            return scores[:topk]
        else:
            return scores

    def insert(self, data):
        # Insert ColBERT embeddings and metadata for a document into the collection.
        colbert_vecs = [vec for vec in data["colbert_vecs"]]
        seq_length = len(colbert_vecs)

        # Insert the data as multiple vectors (one for each sequence) along with the corresponding metadata.
        self.client.insert(
            self.collection_name,
            [
                {
                    "vector": colbert_vecs[i],
                    "sequence_id": i,
                    "document_id": data["document_id"],
                    "document_path": data["document_path"],
                    "page_number": data["page_number"],
                }
                for i in range(seq_length)
            ],
        )

class MilvusPDFImageRetriever(MilvusColbertRetriever):
    def __init__(self, milvus_client, collection_name, dim=128):
        # Initialize with the parent class constructor
        super().__init__(milvus_client, collection_name, dim)
        self.processor = AutoProcessor.from_pretrained("./models/vidore/colpali-v1.3-hf")
        self.model = AutoModelForPreTraining.from_pretrained("./models/vidore/colpali-v1.3-hf")

    def extract_images_from_pdf(self, pdf_path):
        """Extracts images from each page of a PDF file."""
        docs = fitz.open(pdf_path)
        images, page_nums = [], []
        
        for page_num, page in enumerate(docs, start=1):
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
            page_nums.append(page_num)
        return images, page_nums

    def store_pdf_images_in_milvus(self, pdf_path, doc_id):
        """Extracts images, generates embeddings, and stores them in Milvus."""
        images, page_nums = self.extract_images_from_pdf(pdf_path)
        dataloader = DataLoader(
            dataset=images,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
        )
        ds: list[torch.Tensor] = []
        for batch_query in tqdm(dataloader, "Process embedding image"):
            with torch.no_grad():
                batch_query = {k: v.to(self.model.device) for k,v in batch_query.items()}
                embeddings_query = self.model(**batch_query).embeddings
                ds.extend(list(torch.unbind(embeddings_query)))
        
        for page_num, vector in enumerate(tqdm(ds), start=1):
            data = dict(
                colbert_vecs=vector.float().numpy(),
                document_id=doc_id,
                document_path=os.path.abspath(pdf_path),
                page_number=page_num
            )
            self.insert(data)
            
        print(f"Stored images from {pdf_path} in Milvus.")

    def search_pdf_images(self, query, top_k=5):
        """Searches for the most relevant images based on a query."""
        query_embedding = self.model.encode(query).tolist()
        return self.search([query_embedding], top_k)
