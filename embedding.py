import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler
from langfuse.decorators import observe
import pandas as pd
import Stemmer
from llama_index.core import Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore


from settings import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, RETRIEVER_PATH
import utils

langfuse_callback_handler = LlamaIndexCallbackHandler(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host="https://cloud.langfuse.com",
)
Settings.callback_manager = CallbackManager([langfuse_callback_handler])


def create_embedding_model():
    # Check if the embedding model already exists
    if os.path.exists(RETRIEVER_PATH):
        print(f"BM25 retriever already exists at {RETRIEVER_PATH}. Skipping creation.")
        return
    df = pd.read_csv("../datasets/merged_emails.csv")
    # Initialize HuggingFace embedding model and configure settings
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Configure global settings with the embedding model
    Settings.embed_model = embed_model

    # Convert emails to documents
    documents = []
    for i, row in df.iterrows():
        # ensure body is a string
        if not isinstance(row["body"], str) or not row["body"]:
            continue
        text = str(row["body"])
        metadata = {
            "email_id": i,
            "label": int(row.get("label", -1)),
        }

        doc = Document(text=text, metadata=metadata)
        documents.append(doc)

    # Create a docstore to store nodes
    docstore = SimpleDocumentStore()

    # Create storage context
    storage_context = StorageContext.from_defaults(docstore=docstore)

    # Create a node parser with a larger chunk size to accommodate metadata
    splitter = SentenceSplitter(chunk_size=1024)  # Increased from default 1024

    nodes = splitter.get_nodes_from_documents(documents)

    # We can pass in the index, docstore, or list of nodes to create the retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=2,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    bm25_retriever.persist(RETRIEVER_PATH)


class EmailResult(BaseModel):
    """Represents an email search result with similarity score."""

    score: float = Field(description="Similarity score of the result")
    text: str = Field(description="Email body content")
    label: str = Field(description="Label of the email (`safe` or `phishing`)")
    metadata: Dict[str, Any] = Field(description="Additional metadata about the email")

    @field_validator("label", mode="before")
    @classmethod
    def parse_label(cls, value):
        """Convert integer labels to string representations."""
        if isinstance(value, int):
            return "safe" if value == 0 else "phishing"
        return value

    def format(self) -> str:
        """Format the email result for display."""
        return f"""
Similarity Score: {self.score * 100:.2f}%
Label: {self.label}
Text: {utils.sanitize_email(self.text[:100])}...
""".strip()


def load_bm25_retriever(persist_dir: str) -> BM25Retriever:
    """
    Load a pre-built BM25 retriever from its persist directory.

    Args:
        persist_dir: Directory where the BM25 retriever is stored

    Returns:
        The loaded BM25 retriever
    """
    return BM25Retriever.from_persist_dir(persist_dir)


@observe()
def find_similar_emails(
    retriever: BaseRetriever,
    query: str,
) -> List[EmailResult]:
    """
    Find emails similar to the query using the BM25 retriever.

    Args:
        retriever: The BM25 retriever to use for searching
        query: The query text to search for
        top_k: Maximum number of results to return

    Returns:
        List of EmailResult objects containing similar emails and their scores
    """
    # Retrieve nodes from the retriever
    retrieved_nodes: List[NodeWithScore] = retriever.retrieve(query)

    # Convert nodes to EmailResult objects
    results = []
    for node in retrieved_nodes:
        results.append(
            EmailResult(
                score=node.get_score(),
                text=node.get_text(),
                metadata=node.metadata,
                label=node.metadata.get("label", 0),  # Default to 0 if label not found
            )
        )

    return results


if __name__ == "__main__":
    # Example usage
    retriever = load_bm25_retriever("./bm25_retriever")
    results = find_similar_emails(retriever, "Diet pills")

    for i, result in enumerate(results):
        print(f"Result {i + 1}:")
        print(f"Score: {result.score}")
        print(f"Text: {result.text[:100]}...")  # Show first 100 chars
        print(f"Metadata: {result.metadata}")
        print("-" * 50)
