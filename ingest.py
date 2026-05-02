from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Главная_тема"),
    ("##", "Раздел"),
    ("###", "Подраздел")
]

data_id_list = []

with open('knowledge_base.txt', 'r', encoding='utf-8') as f:
    full_text = f.read()

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

chunks = markdown_splitter.split_text(full_text)

for index, _ in enumerate(chunks):
    data_id_list.append(f"id_{index}")

hf = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

db = Chroma.from_documents(
    ids = data_id_list,
    documents = chunks, 
    embedding = hf, 
    persist_directory="./db"
    )

print(db.similarity_search(query = "Как взять отпуск?", k = 3))