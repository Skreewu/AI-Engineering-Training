import os
import json
import typing
import logging
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from prompts import ASSISTANT_SYSTEM_PROMPT_TEMPLATE, DECOMPOSER_SYSTEM_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

handler = logging.FileHandler("Logs/bot_system.log", mode="a", encoding="utf-8")
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(stream_handler)

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def decompose_query(query: str, chat_history: list[dict[str, str]]) -> dict:
    
    formatted_history = [f"{item['role']}: {item['content']}" for item in chat_history]

    system_prompt = DECOMPOSER_SYSTEM_PROMPT_TEMPLATE.format(all_headers = all_headers, history=formatted_history)

    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        temperature=0,
        model="llama-3.3-70b-versatile",
        response_format={"type": "json_object"},
        messages=messages
    )

    parsed_data = json.loads(response.choices[0].message.content)

    return parsed_data

def search_in_db(query: dict) -> list:
    context = []

    search_query = query["question"]
    search_category = query["category"]
    
    if search_category:
        search_result = db.similarity_search(
                query = search_query, 
                k = 2,
                filter={"Подраздел": search_category}
            )
    else:
        search_result = db.similarity_search(
                query = search_query, 
                k = 2,
            )
        
    for doc in search_result:
        context.append(doc.page_content)
    
    return context

def get_context(query: str, chat_history: list[dict[str, str]]) -> str:
    questions = decompose_query(query=query, chat_history=chat_history)

    if "isCorrectQuestion" not in questions:
        return "Warning"
    
    if not questions["isCorrectQuestion"]:
        return "Warning"
    
    context = set()

    for question in questions['queries']:
        results = search_in_db(query=question) 
        context.update(results) 

    return ' '.join(context)


def ask_bot(query: str, chat_history: list[dict[str, str]]) -> str:
    context = get_context(query=query, chat_history=chat_history)

    if context == "Warning":
        logging.info("Некорректный запрос: %s", query)
        return "Не могу ответить на данный вопрос"
    
    system_prompt = ASSISTANT_SYSTEM_PROMPT_TEMPLATE.format(context=context)
    system_message = [{"role": "system", "content": system_prompt}]
    
    chat_history.append({"role": "user", "content": f"<user_input>\n{query.replace('<', '').replace('>', '').replace('/', '')}\n</user_input>"})

    if len(chat_history) > 6:
        del chat_history[:-6]
    else:
        chat_history = chat_history

    messages = system_message + chat_history

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1,
    )

    chat_history.append({"role": "assistant", "content": response.choices[0].message.content})

    return response.choices[0].message.content
    
def init_database(db_path="./db", model_name="paraphrase-multilingual-MiniLM-L12-v2"):

    logger.info("Запуск инициализации баз данных. Путь: '%s', Модель: '%s'", db_path, model_name)

    try:
        logger.debug("Загрузка модуля HuggingFaceEmbeddings")
        hf = HuggingFaceEmbeddings(
            model_name = model_name
        )

        logger.debug("Подключение к директории ChromaDB")
        db = Chroma(
            persist_directory = db_path,
            embedding_function = hf
        )
        
        logger.info("Векторная база данных инициализирована")

        return db
    except FileNotFoundError:
        logger.error("Директория с базой данных не найдена по пути: %s", db_path)
        raise
    except Exception as e:
        logger.critical("Ошибка при загрузке базы данных: %s", str(e), exc_info=True)
        raise

def get_unique_headers(db):
    metadata_list = db.get(include=["metadatas"])["metadatas"]
    for meta in metadata_list:
        if "Подраздел" in meta:
            all_headers.add(meta["Подраздел"])


if __name__ == "__main__":
    db = init_database()
    all_headers = set()
    headers = get_unique_headers(db)

    session_messages = []
    
    print("Помощник запущен. Напишите 'Stop' для выхода.\n")
    
    while True:
        question = input("Вы: ")
        if question.lower() == "stop":
            break
        try:
            reply = ask_bot(query=question, chat_history=session_messages)
            print(reply)
        except Exception as e:
            session_messages.pop()
            print(f"Произошла ошибка {e}. Повторите попытку позже")
