import os
import typing
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
        api_key = os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )


def ask_bot(message: str, chat_history: list[dict[str, str]]) -> typing.Generator[str, None, None]:
    chat_history.append({"role": "user", "content": message})

    if len(chat_history) > 4:
        short_history = chat_history[0:1] + chat_history[-4:]
    else:
        short_history = chat_history

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=short_history,
        temperature=0.3,
        stream = True
    )

    answer = ""
    for chunk in response:
        token = chunk.choices[0].delta.content
        if token:
            yield token
            answer += token
        
    chat_history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    session_messages = [
        {"role": "system", "content": "Ты - язвительный ворчливый пират. Отвечай исключительно на русском языке, только прямой речью"}
    ]

    print("Пиратский бот запущен. Напишите 'Stop' для выхода.\n")
    
    while True:
        question = input("Вы: ")
        if question.lower() == "stop":
            print("Сеанс окончен.")
            break
        try:
            reply = ask_bot(question, session_messages)
            for token in reply:
                print(token, end = '', flush = True)
            print('\n')
        except Exception as e:
            session_messages.pop()
            print(f"Карамба! Потеряна связь с большой землей! {e}")
