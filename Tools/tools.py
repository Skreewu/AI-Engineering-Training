def get_ticket_status(ticket_id: int):
    tickets = [1033, 1034, 1035]
    if ticket_id in tickets:
        return f"Ваша заявка {ticket_id} сейчас в работе. Приблизительное время готовности 2 часа"
    else:
        tickets.append(ticket_id)
        return f"Заявка {ticket_id} создана. Ожидайте."

GET_TICKET_STATUS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_ticket_status",
        "description": "Возвращает статус заявки/тикета пользователя. Вызывется, если пользователь спрашивает про заявку с конкретным номером",
        "parameters": {
        "type": "object",
        "properties": {
            "ticket_id": {
            "type": "integer",
            "description": "Номер заявки для проверки"
            }
        },
        "required": ["ticket_id"]
        }
    }
}