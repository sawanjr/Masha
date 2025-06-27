import os
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
import warnings

# --- SETUP: Configure your details here ---

# Your ActiveLoop Token
os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTc0ODUzODYzMiwiZXhwIjoxNzgwMDc0NjEwfQ.eyJpZCI6InNhd2Fua3VtYXJhcHNnIiwib3JnX2lkIjoic2F3YW5rdW1hcmFwc2cifQ.HL3TP72dEkDYCGYkt1xI3LIOZEPVsmg0HkmfPnNX7VnjGkrgdFvQoDeyX8Wi9LEfi6Q4xjwZuCFy08DBr_aikA"

# The query you want to test
TEST_QUERY = "i feel cheated, how do i overcome this sorrow"

# --- SCRIPT: No need to edit below this line ---

def run_debugger():
    """
    A simple script to test retrieval from Deep Lake.
    """
    print("--- Starting Retrieval Debugger ---")
    
    # Ignore common warnings
    warnings.filterwarnings("ignore")

    try:
        # Step 1: Initialize the embedding model
        print("\n[STEP 1/3] Initializing embedding model...")
        print("(This may take a few minutes on the first run to download the model)")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("✅ Embeddings initialized successfully.")

        # Step 2: Connect to the Deep Lake vector store
        print("\n[STEP 2/3] Connecting to Deep Lake...")
        db = DeepLake(
            read_only=True, 
            dataset_path='hub://sawankumarapsg/vector_store_mental_health', 
            embedding=embeddings
        )
        print("✅ Connected to Deep Lake successfully.")

        # Step 3: Perform the similarity search
        print(f"\n[STEP 3/3] Performing similarity search for query: '{TEST_QUERY}'")
        results = db.similarity_search(TEST_QUERY, k=3)
        print("✅ Search complete.")

        # Step 4: Display the results
        print("\n\n--- 🔍 RETRIEVED CONTEXT 🔍 ---\n")
        
        if not results:
            print("‼️ No results were returned from the search.")
        else:
            print(f"Found {len(results)} documents.\n")
            for i, doc in enumerate(results):
                print(f"--- Document {i+1} ---")
                print(f"  Input (page_content): {doc.page_content}")
                
                # Safely check for metadata before trying to access it
                if doc.metadata:
                    output = doc.metadata.get('output', 'NOT FOUND')
                    row_index = doc.metadata.get('row_index', 'NOT FOUND')
                    print(f"  Metadata (output): {output}")
                    print(f"  Metadata (row_index): {row_index}")
                else:
                    print("  Metadata: ‼️ EMPTY")
                print("-" * 20)

    except Exception as e:
        print(f"\n\n❌ AN ERROR OCCURRED ❌")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debugger()