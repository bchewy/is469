import boto3

# Initialize the client
client = boto3.client('s3vectors', region_name='ap-southeast-1')

BUCKET_NAME = 'is469-genai-grp-project'
INDEX_NAME = 'rag-vector'

print("--- Checking Index ---")

# Method 1: List vectors (Great for checking if *anything* is there)
# This will return a page of vectors currently in your index.
list_response = client.list_vectors(
    vectorBucketName=BUCKET_NAME,
    indexName=INDEX_NAME,
    returnData=True,
    returnMetadata=True 
)

vectors_found = list_response.get('vectors', [])
print(f"Successfully retrieved a batch of {len(vectors_found)} vectors.")

for vec in vectors_found[:3]: # Print the first 3 keys to verify
    print(f"Found Vector ID: {vec.get('key')}")


# Method 2: Get a specific vector (Great if you know an exact ID)
# Replace 'your-specific-id' with an actual ID string from your JSONL file
try:
    get_response = client.get_vectors(
        vectorBucketName=BUCKET_NAME,
        indexName=INDEX_NAME,
        keys=["23851"],
        returnData=True,
        returnMetadata=True
    )
    
    match = get_response.get('vectors', [])
    if match:
        print(f"\nSuccess! Found specific vector: {match[0].get('key')}")
        
        # Note: S3 Vectors automatically handles the data as float32 arrays
        vector_data = match[0].get('data', {}).get('float32', [])
        print(f"First 5 dimensions: {vector_data[:5]}")
except Exception as e:
    print(f"\nError fetching specific vector: {e}")