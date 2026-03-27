import boto3
import json

# Initialize the S3 Vectors client
# (Ensure your boto3 library is updated to the latest version!)
client = boto3.client('s3vectors', region_name='ap-southeast-1') 

BUCKET_NAME = 'is469-genai-grp-project'
INDEX_NAME = 'rag-vector'
BATCH_SIZE = 500

vectors_batch = []

with open('../kb/knowledge_base_vectors.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        
        # S3 Vectors requires 'key', 'data', and 'metadata'
        vector_record = {
            'key': str(data['id']),               # Unique ID for the vector
            'data': {'float32': data['embedding']},  # PutVectors expects a typed vector payload
            'metadata': data.get('metadata', {})  # Optional metadata (like the source text)
        }
        vectors_batch.append(vector_record)
        
        # Push to AWS when the batch hits 500
        if len(vectors_batch) >= BATCH_SIZE:
            client.put_vectors(
                vectorBucketName=BUCKET_NAME,
                indexName=INDEX_NAME,
                vectors=vectors_batch
            )
            vectors_batch = [] # Reset the batch

# Push any remaining vectors
if vectors_batch:
    client.put_vectors(
        vectorBucketName=BUCKET_NAME,
        indexName=INDEX_NAME,
        vectors=vectors_batch
    )

print("Ingestion into S3 Vector Bucket complete!")