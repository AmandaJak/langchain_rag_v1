# pip install -U langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu
# export OPENAI_API_KEY="your_api_key"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Together, HuggingFaceHub
from langchain_together import TogetherEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 1) Load
loader = TextLoader("docs.txt")
docs = loader.load()

# 2) Split
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 3) Embed + store + retriever
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4) Model + prompt
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_hf = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token="YOUR_HF_API_TOKEN"
)
llm = Together(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant for question-answering tasks. "
     "Use the following pieces of retrieved context to answer the question. "
     "If you don't know the answer, just say you don't know. "
     "Use at most three sentences and keep the answer concise.\n\nContext:\n{context}"),
    ("human", "{question}")
])

# 5) LCEL chain
def join_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | join_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6) Ask
print(rag_chain.invoke("What are the main components of an LLM-powered autonomous agent system?"))
