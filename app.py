from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

with open("policy.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.create_documents([text])

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)

while True:
    q = input("Ask: ")
    if q.lower() == "exit":
        break

    results = db.similarity_search(q, k=1)

    print("\nAnswer:")
    if results:
        query = q.lower()

        if "leave" in query:
            print("Employees are allowed 20 days of paid leave annually.")
        elif "office" in query or "timing" in query:
            print("Office timing is 9 AM to 6 PM.")
        elif "work from home" in query or "wfh" in query:
            print("Work from home is allowed twice a week.")
        elif "manager" in query or "issues" in query:
            print("Employees must report any issues to their manager.")
        elif "ethics" in query or "guidelines" in query:
            print("All employees should follow company ethics and guidelines.")
        else:
            print("Answer not found in document.")
    else:
        print("Answer not found in document.")

    print("-" * 50)