import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Use service role key

# Ensure environment variables are loaded
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Missing required environment variables. Check your .env file.")

# Initialize the Supabase client with the service role key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def fetch_data():
    """Fetch only unvectorized rows from Supabase."""
    print("Fetching unvectorized data...")
    try:
        # Fetch all messages for debugging
        messages_res = supabase.table("messages").select("*").execute()
        print(f"All messages fetched: {messages_res.data}")

        # Fetch unvectorized messages
        messages_res = supabase.table("messages").select("*").eq("is_vectorized", False).execute()
        print(f"Unvectorized messages: {messages_res.data}")

        # Fetch unvectorized direct messages
        dms_res = supabase.table("direct_messages").select("*").eq("is_vectorized", False).execute()
        print(f"Fetched {len(dms_res.data or [])} direct messages.")

        # Fetch unvectorized message replies
        replies_res = supabase.table("message_replies").select("*").eq("is_vectorized", False).execute()
        print(f"Fetched {len(replies_res.data or [])} message replies.")

        print(f"Fetched {len(messages_res.data or [])} messages.")
        print(f"Fetched {len(dms_res.data or [])} direct messages.")
        print(f"Fetched {len(replies_res.data or [])} message replies.")

        return (
            messages_res.data or [],
            dms_res.data or [],
            replies_res.data or []
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        return [], [], []

if __name__ == "__main__":
    fetch_data()