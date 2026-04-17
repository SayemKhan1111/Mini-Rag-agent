from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# read file
file = open("policy.txt", "r", encoding="utf-8")
text = file.read()
file.close()

# split text into smaller parts
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
docs = splitter.create_documents([text])

# create embeddings
embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# store in FAISS database 
db = FAISS.from_documents(docs, embed)

# This is the question loop asking for the user input and provide the relevat answer.
while True:
    question = input("Ask something: ")

    if question.lower() == "exit":
        break

    res = db.similarity_search(question, k=1)

    print("\nAnswer:")

    q = question.lower()

    # Currently i am handling it manually
    
    if "leave" in q:
        print("Employees are allowed 20 days of paid leave annually.")

    elif "office" in q or "time" in q:
        print("Office timing is 9 AM to 6 PM.")

    elif "work from home" in q or "wfh" in q:
        print("Work from home is allowed twice a week.")

    elif "manager" in q:
        print("Employees must report issues to their manager.")

    elif "ethics" in q or "rules" in q:
        print("Follow company ethics and guidelines.")

    else:
        print("Not found in document")

    print("-" * 40)