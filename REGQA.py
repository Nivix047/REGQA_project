# Import necessary libraries
import os
import dotenv
import openai
import pinecone
import PyPDF2

# Load environment variables from .env file
dotenv.load_dotenv(dotenv_path="./.env.local")

# Initialize OpenAI and Pinecone with API keys from environment variables
openai.api_key = os.getenv("gpt_api_secret")
pinecone.init(api_key=os.getenv("pinecone_api_key"),
              environment="us-west1-gcp-free")

# Check Pinecone user details
pinecone.whoami()

# Define the path to the PDF file and the query to search for
pdf_path = "/Users/nivix047/Desktop/Mongodb CRUD.pdf"
query = "How do I nest things?"

# Define a function to generate completions using OpenAI


def complete(prompt):
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

# Define a function to retrieve text based on a given query


def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )
    xq = res['data'][0]['embedding']
    res = index.query(xq, top_k=1, include_metadata=True)
    context = res['matches'][0]['metadata']['text']
    prompt = "Answer the question based on the context below.\n\ncontext:\n" + \
        context + f"\n\nQuestion: {query}\n\nAnswer:"
    return prompt

# Define a function to extract text from a PDF file


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file_obj:
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page_obj = pdf_reader.pages[page_num]
            text += page_obj.extract_text()
    return text


# Extract text from the PDF file
text = extract_text_from_pdf(pdf_path)

# Break the extracted text into chunks
chunks = []
chunk_size = 10000
overlap_size = 5000
for i in range(0, len(text), chunk_size - overlap_size):
    chunks.append(text[i:i + chunk_size])

# Create embeddings for the chunks
embed_model = "text-embedding-ada-002"
res = openai.Embedding.create(
    input=chunks,
    engine=embed_model
)

# Define Pinecone index details
index_name = "regqa"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine'
    )

# Create Pinecone index
index = pinecone.Index(index_name=index_name)

# Retrieve details about the Pinecone index
print(index.describe_index_stats())

# Upsert vectors into the index
to_upsert = [(f"id{i}", res['data'][i]['embedding'], {"text": chunks[i]})
             for i in range(len(res['data']))]
index.upsert(vectors=to_upsert)

# Retrieve and print completion
query_with_context = retrieve(query)
print(complete(query_with_context))

# Delete Pinecone index
pinecone.delete_index(index_name)
