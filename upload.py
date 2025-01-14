import os
from dotenv import load_dotenv
from supabase import create_client, Client
import openai
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_pinecone import Pinecone as LangchainPinecone

# Load environment variables
load_dotenv()

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Service role key for Supabase
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # Pinecone environment, e.g., "us-east-1"
PINECONE_INDEX = os.getenv("PINECONE_INDEX")  # Pinecone index name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Create Supabase client using the service role key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Ensure the index exists in Pinecone
if PINECONE_INDEX not in pinecone_client.list_indexes().names():
    print(f"Index '{PINECONE_INDEX}' not found. Creating it...")
    pinecone_client.create_index(
        name=PINECONE_INDEX,
        dimension=3072,  # Match your embedding dimensions (text-embedding-3-large)
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENV  # Ensure this matches your Pinecone environment
        )
    )
else:
    print(f"Using existing index '{PINECONE_INDEX}'.")

# 1. Fetch unvectorized data from Supabase
def fetch_unvectorized_data():
    print("Fetching unvectorized data...")
    messages_res = supabase.table("messages") \
        .select("*") \
        .eq("is_vectorized", False) \
        .execute()

    direct_messages_res = supabase.table("direct_messages") \
        .select("*") \
        .eq("is_vectorized", False) \
        .execute()

    replies_res = supabase.table("message_replies") \
        .select("*") \
        .eq("is_vectorized", False) \
        .execute()

    print(f"Fetched {len(messages_res.data or [])} messages.")
    print(f"Fetched {len(direct_messages_res.data or [])} direct messages.")
    print(f"Fetched {len(replies_res.data or [])} message replies.")

    return (
        messages_res.data or [],
        direct_messages_res.data or [],
        replies_res.data or []
    )

# 2. Convert rows to LangChain Documents
def rows_to_documents(messages, direct_messages, replies):
    docs = []
    for m in messages:
        if not m["content"]:
            continue
        docs.append(Document(
            page_content=m["content"],
            metadata={
                "table": "messages",
                "id": m["id"],
                "user_id": m["user_id"]
            }
        ))
    for dm in direct_messages:
        if not dm["message"]:
            continue
        docs.append(Document(
            page_content=dm["message"],
            metadata={
                "table": "direct_messages",
                "id": dm["id"],
                "user_id": dm["sender_id"]  # Map sender_id to user_id
            }
        ))
    for r in replies:
        if not r["content"]:
            continue
        docs.append(Document(
            page_content=r["content"],
            metadata={
                "table": "message_replies",
                "id": r["id"],
                "user_id": r["user_id"]
            }
        ))
    return docs

# 3. (Optional) Split large texts into chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitted_docs = text_splitter.split_documents(docs)
    return splitted_docs

# 4. Insert documents into Pinecone
def upload_to_pinecone(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # 3072 dim
    # Upload documents to Pinecone
    vectorstore = LangchainPinecone.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX
    )

# 5. Mark rows as vectorized in Supabase
def mark_as_vectorized(metadata_list):
    for m in metadata_list:
        table = m["table"]
        row_id = m["id"]

        result = supabase.table(table) \
            .update({"is_vectorized": True}) \
            .eq("id", row_id) \
            .execute()
        print(f"Updated row {row_id} in table {table}. Result: {result}")

def main():
    # Fetch unvectorized data
    msgs, dms, reps = fetch_unvectorized_data()

    # Convert to Documents
    docs = rows_to_documents(msgs, dms, reps)

    if not docs:
        print("No unvectorized rows found.")
        return

    # Split if large
    splitted_docs = split_documents(docs)
    print(f"Ready to embed {len(splitted_docs)} total chunks.")

    # Upload to Pinecone
    upload_to_pinecone(splitted_docs)

    # Mark them as vectorized
    original_meta = []
    for d in splitted_docs:
        meta = d.metadata
        if meta not in original_meta:
            original_meta.append(meta)

    mark_as_vectorized(original_meta)

    print("Done uploading to Pinecone and marking rows as vectorized.")

if __name__ == "__main__":
    main()