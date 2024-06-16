import time
import os
import re
from langchain_community.document_loaders import SRTLoader, TextLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import BedrockChat
from langchain.globals import set_verbose, set_debug
# for experiments with local embeddings
# from langchain_community.embeddings.sentence_transformers import SentenceTransformersEmbeddings

set_debug(True)


def get_embeddings():
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
    )
    return embeddings


def get_llm():
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    model_kwargs = {
        "max_tokens": 1500,
        "anthropic_version": "bedrock-2023-05-31",
        "stop_sequences": ["User:"],
        "temperature": 1,
        "top_k": 250,
        "top_p": 0.999
    }

    llm = BedrockChat(
        model_id=model_id,
        model_kwargs=model_kwargs)
    return llm


def get_prompt():
    prompt_template = ChatPromptTemplate.from_template("""
    Answer the question briefly based only on the following context. 
    If the context don't contain the answer, type "I don't know".
    Do not include preambles like "according to the documents", "based on the provided context" or "the answer is". In this case just skip to the answer.

    After answer, give a brief list of the context documents. 
    List must be in bullet form in this format:
    - Source: [filename/url], Page: [page number], Relevance: [0-100]

    <context>
    {context}
    </context>

    Question: <question>{question}</question>
    """)
    return prompt_template


def remove_emojis(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )

    return emoji_pattern.sub(r'', string)


def read_documents():
    # loading pdf, but this can be replaced with any other document loader, including web crawlers
    print("Step 1: Reading documents...")

    docs = []
    # read all srt files from data/ folder1.1. Reading
    print("1.1. Reading SRT files...")
    filenames = os.listdir("data")
    time_start = time.time()
    for filename in filenames:
        if filename.endswith(".srt"):
            print(f"- {filename}...")
            file_path = f"data/{filename}"
            loader = SRTLoader(file_path)
            docs.extend(loader.load())

    # read all JSON summaries from data/ folder
    print("1.2. Reading JSON summaries...")
    for filename in filenames:
        if filename.endswith(".json"):
            print(f"- {filename}...")
            file_path = f"data/{filename}"
            loader = TextLoader(file_path)
            docs.extend(loader.load())

    # clear all document contents from weird characters
    for doc in docs:
        doc.page_content = remove_emojis(doc.page_content)

    time_end = time.time()
    print(f"Time elapsed: {time_end - time_start:.2f} s")
    print("Number of documents: ", len(docs))
    # get only first 10 documents
    # docs = docs[:10]
    return docs


def split_documents(docs):
    print("Step 2: Splitting documents...")
    time_start = time.time()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100)
    docs_split = text_splitter.split_documents(docs)
    time_end = time.time()
    print(f"Time elapsed: {time_end - time_start:.2f} s")
    print("Number of split documents: ", len(docs_split))
    return docs_split


def load_documents(docs_split, embeddings):
    print("Step 3: Creating Chroma database...")
    time_start = time.time()
    db = Chroma.from_documents(docs_split, embedding=embeddings)
    time_end = time.time()
    print(f"Time elapsed: {time_end - time_start:.2f} s")
    return db


def process_query(query, db, prompt_template, llm):
    chain = (
        {
            "question": RunnablePassthrough(),
            "context": db.as_retriever(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    print("Running chain...")
    time_start = time.time()
    output = chain.invoke(query)
    time_end = time.time()
    print(f"Time elapsed: {time_end - time_start:.2f} s")
    print(output)
    print("\n\n")


if __name__ == "__main__":
    embeddings = get_embeddings()
    llm = get_llm()
    prompt = get_prompt()
    docs = read_documents()
    docs_split = split_documents(docs)
    db = load_documents(docs_split, embeddings)

    # infinite loop, get query from keyboard, run against db, print results
    while True:
        query = input("Enter query: ")
        if query == "exit":
            break
        # checking ChromaDB similarity search returning documents
        # time_start = time.time()
        # results = db.similarity_search(query)
        # time_end = time.time()
        # print(f"Time elapsed: {time_end - time_start:.2f} s")
        # print("Number of results: ", len(results))
        # for result in results:
        #     print(result)
        # print("")
        process_query(query, db, prompt, llm)
