from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

# Initialize OpenAI Embeddings (must match the model used during data upload)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_profiles():
    """Fetch all profiles from the Supabase `profiles` table."""
    print("Fetching profiles from Supabase...")
    try:
        profiles_res = supabase.table("profiles").select("id, username").execute()
        if profiles_res.data:
            profiles = {profile["id"]: profile["username"] for profile in profiles_res.data}
            print(f"Fetched {len(profiles)} profiles.")
            return profiles
        else:
            print("No profiles found.")
            return {}
    except Exception as e:
        print(f"Error fetching profiles: {e}")
        return {}


def replace_user_ids_with_usernames(docs, profiles):
    """Replace `user_id` in metadata with `username`."""
    for doc in docs:
        user_id = doc.metadata.get("user_id")
        if user_id and user_id in profiles:
            doc.metadata["username"] = profiles[user_id]
            del doc.metadata["user_id"]  # Remove `user_id` if `username` was added
    return docs


def get_ai_response(event):
    """Main function to handle the AI response logic."""
    # Extract the query from the incoming Lambda event
    query = event.get("query", "What are the key features of the product we discussed?")

    # Fetch all profiles
    profiles = fetch_profiles()

    # Query the vector database
    print("Querying Pinecone vector store for relevant documents...")
    document_vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings)
    retriever = document_vectorstore.as_retriever()

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        print("No relevant documents found.")
        return {"status": "no_documents", "response": "No relevant documents found."}

    # Replace user IDs with usernames
    print("Replacing user_id with username in document metadata...")
    updated_docs = replace_user_ids_with_usernames(retrieved_docs, profiles)

    # Build a richer context by including metadata such as username
    context = "\n\n".join([
        f"Username: {doc.metadata.get('username', 'Unknown')}\nContent: {doc.page_content}"
        for doc in updated_docs
    ])

    # Adjust the prompt template
    template = PromptTemplate(
        template="You are a product manager reviewing team discussions. Based on the context below, answer the query. If it's not clear based on the context, don't include irrelevant or made-up info, just answer the best that you can. \n\nQuery: {query}\n\nContext:\n{context}\n\nYour response:",
        input_variables=["query", "context"]
    )
    prompt_with_context = template.invoke({"query": query, "context": context})

    # Ask the LLM for a response
    print("Querying LLM for a response...")
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
    results = llm.invoke(prompt_with_context)

    # Return the LLM response
    print("LLM Response:")
    print(results.content)
    return {"status": "success", "response": results.content}


def handler(event, context):
    """AWS Lambda handler."""
    try:
        response = get_ai_response(event)
        return {
            "statusCode": 200,
            "body": response
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": {"status": "error", "error": str(e)}
        }