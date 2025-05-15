import os
from dotenv import load_dotenv

load_dotenv()

RETRIEVER_PATH = "./bm25_retriever"

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
assert LANGFUSE_SECRET_KEY, "LANGFUSE_SECRET_KEY is not set"

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
assert LANGFUSE_PUBLIC_KEY, "LANGFUSE_PUBLIC_KEY is not set"

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", default="https://cloud.langfuse.com")


DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
