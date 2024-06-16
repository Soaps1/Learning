from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import bs4
from langchain import hub
from chromadb import Client
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
openai_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI API
llm = ChatCompletion(engine="gpt-3.5-turbo", api_key=openai_key)

# Load, chunk, and index the contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | prompt
    | llm
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = rag_chain({"context": "", "question": question})
    return answer['choices'][0]['text']

if __name__ == '__main__':
    app.run(debug=True)
