import os
from dotenv import load_dotenv
from supabase import create_client, Client
from pinecone import Pinecone, Index

# Load environment variables
load_dotenv()

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Service role key for Supabase
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")  # Use full Pinecone host URL
PINECONE_INDEX = os.getenv("PINECONE_INDEX")  # Pinecone index name

# Create Supabase client using the service role key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Ensure Pinecone index exists
if PINECONE_INDEX not in pinecone_client.list_indexes().names():
    raise ValueError(f"Index '{PINECONE_INDEX}' does not exist.")

# Reset `is_vectorized` in Supabase
def reset_supabase_is_vectorized():
    print("Resetting is_vectorized to false for all tables in Supabase...")

    tables = ["messages", "direct_messages", "message_replies"]

    for table in tables:
        try:
            result = supabase.table(table).update({"is_vectorized": False}).neq("is_vectorized", False).execute()
            if result.data:  # Check if rows were updated
                print(f"Updated {len(result.data)} rows in table {table}.")
            else:
                print(f"No rows needed updating in table {table}.")
        except Exception as e:
            print(f"Error resetting is_vectorized for table {table}: {e}")

# Clear Pinecone index
def clear_pinecone_index():
    print(f"Deleting all vectors in Pinecone index '{PINECONE_INDEX}'...")
    try:
        from pinecone import Index
        # Use the full host URL
        index = Index(
            index_name=PINECONE_INDEX,
            api_key=PINECONE_API_KEY,
            host=PINECONE_HOST  # Use the full host URL from Pinecone
        )
        index.delete(delete_all=True)  # Deletes all vectors in the index
        print("Successfully deleted all vectors from the index.")
    except Exception as e:
        print(f"Error deleting vectors from Pinecone index: {e}")

# Main cleanup function
def main():
    reset_supabase_is_vectorized()
    clear_pinecone_index()
    print("Cleanup complete. Supabase and Pinecone are reset.")

if __name__ == "__main__":
    main()