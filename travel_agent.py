#!/usr/bin/env python3
"""
Advanced Travel Agent using the Game framework
This agent conducts a structured interview with users about their travel preferences
and includes error handling for invalid responses.
"""

import importlib
import os
import json
import asyncio
import random
from typing import List, Dict, Any
from dotenv import load_dotenv
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from fuzzywuzzy import fuzz
import logging

# Import the Game framework
import game.core
importlib.reload(game.core)
from game.core import Environment, Goal, register_tool, PythonActionRegistry, Agent, \
    AgentFunctionCallingActionLanguage, generate_response, Memory

# Load environment variables from .env file
load_dotenv()
# Настройка логирования в файл и консоль
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("travel_agent.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("travel_agent")

# Список моделей: основная и резервная (платные версии)
PRIMARY_MODEL = "openrouter/google/gemini-2.0-flash-001"
FALLBACK_MODEL = "openrouter/openai/gpt-4o-mini"


# Configure litellm for OpenRouter function calling
import litellm
from litellm import RateLimitError, ServiceUnavailableError
litellm.drop_params=True
# Note: add_function_to_prompt can cause issues with newer litellm versions
# litellm.add_function_to_prompt = True

# Global state to track current goal and user responses
#
agent_state = {
    "current_goal": 1,
    "user_responses": {},
    "error_count": 0,
    "max_errors": 11,
    "goal_completed": False,
    "conversation_active": True,
    "has_asked_goal_1": False,
    "dynamic_questions_count": 0,
    "dynamic_completed": False
}

def log_conversation(user_id: str, sender: str, text: str):
    msk = timezone(timedelta(hours=3))
    now = datetime.now(msk).strftime("%Y-%m-%d %H:%M:%S")
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / f"Conversation_VK_{user_id}.txt"
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {sender}: {text}\n\n")

def reset_agent_state():
    """Reset the agent state for a new conversation"""
    global agent_state
    print(f"🔄 Resetting agent state for new conversation")
    # Clear the existing state
    agent_state.clear()
    # Set new values
    agent_state.update({
        "current_goal": 1,
        "user_responses": {},
        "error_count": 0,
        "max_errors": 11,
        "goal_completed": False,
        "conversation_active": True,
        "has_asked_goal_1": False,
        "dynamic_questions_count": 0,
        "dynamic_completed": False
    })

# Define the main goal for the travel agent - sequential execution
goals = [
    Goal(
        priority=1,
        name="Sequential Travel Planning",
        description=(
            "Execute travel planning goals sequentially: "
            "1) Capture travel dates, 2) Ask group size, 3) Ask if children will travel, "
            "4) Ask children age (if applicable), 5) Ask destination preferences, "
            "6) Ask budget, 7) Ask departure city, 8) Run dynamic Perplexity clarification loop, "
            "9) Ask final catch-all question, 10) Terminate conversation. Handle errors gracefully."
        )
    )
]

@register_tool(tags=["sequential", "main"])
def execute_sequential_travel_planning() -> str:
    """Execute the travel planning process sequentially through all goals."""
    current_goal = agent_state["current_goal"]
    if current_goal == 2:
        return ask_group_size()
    elif current_goal == 3:
        return ask_children_exist()
    elif current_goal == 4:
        return ask_children_age()
    elif current_goal == 5:
        return ask_destination_preferences()
    elif current_goal == 6:
        return ask_budget()
    elif current_goal == 7:
        return ask_departure_city()
    elif current_goal == 9:
        return ask_final_catch_all()
    elif current_goal == 10:
        return terminate("Отлично! Все ваши вводные учтем максимально. Эксперт подготовит для вас подборку туров, и свяжется с вами")
    else:
        return "Travel planning session completed. Thank you!"


@register_tool(tags=["interview", "goal_2"])
def ask_group_size() -> str:
    """Ask who will travel and how many adults."""
    return "Хорошо. Кто поедет? Сколько взрослых?"


@register_tool(tags=["interview", "goal_3"])
def ask_children_exist() -> str:
    """Ask if children will travel."""
    return "Поедут ли дети?"


@register_tool(tags=["interview", "goal_4"])
def ask_children_age() -> str:
    """Ask about children's ages."""
    return "Уточните возраст детей?"


@register_tool(tags=["interview", "goal_5"])
def ask_destination_preferences() -> str:
    """Ask about destination preferences."""
    return "Куда Вы хотели ли бы поехать?"


@register_tool(tags=["interview", "goal_6"])
def ask_budget() -> str:
    """Ask about desired budget."""
    return "Хорошо. В какой общий бюджет хотели бы уложиться?"


@register_tool(tags=["interview", "goal_7"])
def ask_departure_city() -> str:
    """Ask about departure city."""
    return "Откуда планируете стартовать?"


@register_tool(tags=["interview", "goal_9"])
def ask_final_catch_all() -> str:
    """Ask final catch-all question before termination."""
    return (
        "Спасибо за Ваши ответы. Подскажите, какие еще моменты важно учесть "
        "при составлении подборки туров, которые ранее не обсудили?"
    )


def clean_perplexity_response(text: str) -> str:
    import re

    # 1. Удаляем Markdown жирность (превращаем **текст** в текст)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    # 2. Удаляем сноски [1], [s1] и т.д.
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[s\d+\]', '', text)

    # 3. Список фраз-приветствий для удаления из начала
    """bad_starts = [
        "Добрый день", "Здравствуйте", "Приветствую",
        "Спасибо за уточнение", "Понял Вас", "Понял Вашу позицию",
        "Рад помочь", "Отличный выбор"
    ]"""

    # Чистим начало текста от вежливости
    """for start in bad_starts:
        # Ищем фразу в начале, учитывая возможные знаки препинания после неё
        pattern = re.compile(r'^' + re.escape(start) + r'[\s\d.,!]*', re.IGNORECASE)
        text = pattern.sub('', text).strip()"""

    # 4. ВАЖНО: Вместо замены всех пробелов на один,
    # заменяем только ТРОЙНЫЕ и более переносы на двойные, чтобы сохранить абзацы
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Убираем лишние пробелы в концах строк, но сохраняем сами строки
    text = "\n".join([line.strip() for line in text.split('\n')])

    # Капитализируем первую букву результата
    if text:
        text = text[0].upper() + text[1:]

    return text.strip()


async def call_llm_with_retry(messages: List[Dict[str, str]], max_retries=2):
    """
    Профессиональная обертка для вызова LLM с ретраями и переключением на запасную модель.
    """
    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]

    for model in models_to_try:
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Запрос к модели {model} (попытка {attempt + 1})")
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    timeout=30
                )
                return response.choices[0].message.content

            except (RateLimitError, ServiceUnavailableError) as e:
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Лимит достигнут ({model}). Ждем {wait_time}с... Ошибка: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Попытки для модели {model} исчерпаны.")

            except Exception as e:
                logger.error(f"Критическая ошибка при вызове {model}: {str(e)}")
                break  # Переходим к следующей модели в списке

    return None

# make it separate action
async def run_dynamic_interview(agent_state: dict, user_input: str, history_dialogue: str):
    """
    Основная логика динамического интервью с защитой от зацикливания.
    """

    responses = agent_state.get("user_responses", {})
    trip_type = responses.get("trip_type", "Организованный туризм через турфирму")
    destination = responses.get("destination", "")
    group_size = responses.get("group_size", "")
    travel_dates = responses.get("travel_dates", "")
    departure_city = responses.get("departure_city", "")
    budget = responses.get("budget", "")
    children_info = responses.get("children_info", "")

    dynamic_questions_count = agent_state.get("dynamic_questions_count", 0)
    limit_instruction = "\nВАЖНО: Лимит вопросов исчерпан. Спроси, что еще учесть, и попроси номер телефона." if dynamic_questions_count >= 10 else ""

    try:
        # Твои базовые вопросы
        all_questions = [
            "Для вас предпочтительнее уже знакомое направление или интереснее отправиться в новое? *(уместен ТОЛЬКО, если хочет в новое направление)*",
            "В какое новое место хотели бы отправиться? *(уместен ТОЛЬКО, если хочет в новое направление)*",
            "Готовы ли рассматривать перелеты с пересадками для удешевления? *(уместен ТОЛЬКО, если без маленьких детей)*",
            "Есть ли предпочтения по авиакомпаниям? ",
            "Строгие ли даты поездки или возможны гибкие даты ±3 дня?",
            "Есть ли особые пожелания по времени перелета? Готовы ли рассматривать ночной перелет?",
            "Что важно учесть при подборе отеля?",
            "Готовы ли рассматривать отели дальше 3-й линии от пляжа? *(уместен ТОЛЬКО, при небольшом бюджете)*",
            "Готовы ли рассмотреть отели ниже 4 звезд? *(уместен ТОЛЬКО, при небольшом бюджете)*",
            "Какие отели предпочитаете? С активной развлекательной программой или более тихие семейные?",
            "Есть ли предпочтения к определенным брендам отелей (Hilton, Rixos, TUI или другие)?",
            "Есть ли предпочтения по туроператору (Библиоглобус, Анекс, Интурист или другой)?",
            "Чем бы вы хотели заниматься на месте (пляж, экскурсии, шопинг, активный отдых)?",
            "Какие экскурсии вам наиболее интересны?",
            "Требуется ли подбор экскурсий заранее или планируете приобретать их на месте?",
            "Планируете ли активный отдых?",
            "Нужны ли доп. услуги (SPA, дайвинг, аренда авто)?",
            "Какая медицинская страховка вам больше подойдет — стандартная или расширенная?",
            "Удобно ли вам созвониться или встретиться у нас в офисе для обсуждения подборки туров?",
            "Какой бюджет на человека (включая перелет, отель и доп. услуги)?",
            "Рассмотрели бы увеличение бюджета, если появится хороший вариант?",
            "На какую максимальную сумму готовы увеличить бюджет?",
            "Уточните номер телефона для связи *(если удобно созвониться)*?",
            "В какое время вам удобнее будет поговорить по телефону или подъехать в офис?"
        ]

        filtered_questions = []
        for i, q in enumerate(all_questions, 1):
            clean_q = re.sub(r'\*\(.*?\)\*', '', q).strip()
            clean_q = clean_q.split('?')[0].strip()
            if clean_q not in history_dialogue:
                filtered_questions.append(f"{i}. {q}")


        if not filtered_questions:
            filtered_questions = ["Подскажите, какие еще моменты важно учесть при составлении подборки туров?"]

        available_questions_str = "\n".join(filtered_questions)

        # Определяем, на каком мы этапе (сохраняем твою логику)
        current_step = agent_state.get("dynamic_questions_count", 0)

        # Формируем контекст для модели
        prompt = f"""
Ты ведущий менеджер турагентства. 
Я - клиент. Задавай ТОЛЬКО вопросы. Здороваться не надо!

УЖЕ ИЗВЕСТНО:
1. Тип поездки: {trip_type} 
2. Направление: {destination} 
3. Размер группы: {group_size}
4. Время поездки: {travel_dates}
5. Место отправления: {departure_city} 
6. Бюджет поездки: {budget}
7. Информация о детях: {children_info}

ИСТОРИЯ ДИАЛОГА (НЕ ЗАДАВАЙ ВОПРОСЫ ИЗ ИСТОРИИ!):
{history_dialogue}

ПРАВИЛА ТВОЕГО ПОВЕДЕНИЯ: 
- Выбери СЛЕДУЮЩИЙ, наиболее подходящий вопрос из списка "ОСТАВШИЕСЯ ВОПРОСЫ" ниже и задай его.
- Задавай СТРОГО по одному вопросу. Жди ответа. 
- Не пиши сноски, пиши кратко, без точек в конце.{limit_instruction}
- ВАЖНО: Если я хамлю или отказываюсь отвечать, отправь: "Спасибо за Ваши ответы. Подскажите, какие еще моменты важно учесть при составлении подборки туров?"
- Если клиент дал номер телефона, ответь СТРОГО: "Отлично! Все ваши вводные учтем максимально. Эксперт подготовит для вас подборку туров и свяжется с вами."

ОСТАВШИЕСЯ ВОПРОСЫ:
{available_questions_str}
"""

        messages = [{"role": "system", "content": prompt}]

        # Вызываем LLM через нашу безопасную функцию
        assistant_text = await call_llm_with_retry(messages)

        if not assistant_text:
            # Если все API упали, не пугаем пользователя ошибкой, а задаем вопрос "вручную"
            logger.critical("Все модели LLM недоступны! Переход на ручной режим.")
            if current_step < len(all_questions):
                fallback_q = all_questions[current_step]
                return f"Понял вас. А подскажите еще такой момент: {fallback_q.lower()}"
            return "Спасибо! Я записал ваши основные пожелания. Наш менеджер скоро свяжется с вами для уточнения деталей."

        # Обновляем состояние (имитация прогресса)
        agent_state["dynamic_questions_count"] += 1
        if agent_state["dynamic_questions_count"] >= len(all_questions):
            agent_state["dynamic_completed"] = True

        return assistant_text

    except Exception as e:
        # Логируем ошибку со стектрейсом для отладки
        logger.exception(f"Ошибка в run_dynamic_interview: {str(e)}")
        # Возвращаем "безопасную" фразу, которая не выглядит как сбой
        return "Очень интересно! Давайте еще уточним: есть ли у вас особые пожелания по отелю или перелету?"


@register_tool(tags=["error_handling", "goal_9"])
def handle_user_error() -> str:
    """Handle user errors or invalid responses.
    
    Returns:
        A polite message asking for clarification
    """
    current_goal = agent_state["current_goal"]
    error_count = agent_state["error_count"]
    
    if error_count >= agent_state["max_errors"]:
        return "I apologize, but I'm having trouble understanding your responses. Please try again later or contact our support team for assistance. Thank you for your time!"
    
    goal_messages = {
        1: "travel dates",
        2: "group size (number of adults)",
        3: "whether children will travel",
        4: "children's ages",
        5: "destination preferences",
        6: "budget",
        7: "departure city",
        8: "clarifying question",
        9: "any additional important details",
    }
    
    current_question = goal_messages.get(current_goal, "the current question")
    
    return f"""I apologize, but I didn't understand your response clearly. 

Could you please answer the question about {current_question}? 

If you're unsure or embarrassed to answer, please let me know when would be a good time to ask you again about this.

Please provide a clear answer so we can continue with your travel planning."""


async def validate_user_input(question_asked: str, user_answer: str) -> dict:
    prompt = f"""
    Ты — опытный и понимающий тревел-эксперт. 
    ВОПРОС: "{question_asked}"
    ОТВЕТ ПОЛЬЗОВАТЕЛЯ: "{user_answer}"

    ТВОЯ ЗАДАЧА:
    1. Если в ответе есть ПОНЯТНЫЙ СМЫСЛ, подходящий к вопросу (например, "актиной" вместо "активной", "масква" вместо "Москва", "да", "тут", "здесь") — считай ответ ВАЛИДНЫМ (true).
    2. Опечатки и сокращения — это НОРМАЛЬНО.
    3. Короткие ответы ("исторические", "тихие", "ничего", "всё") — СТРОГО ВАЛИДНЫ.
    4. НЕВАЛИДНЫМ (false) считай только полный бессмысленный бред или мат.

    ФОРМАТ ОТВЕТА (JSON):
    Если смысл понятен: {{"is_valid": true}}
    Если совсем бред: {{"is_valid": false, "reason": "Придумай вежливую фразу, почему ты не понял"}}
    """

    try:
        # ИСПОЛЬЗУЕМ НАТИВНЫЙ АСИНХРОННЫЙ ВЫЗОВ
        response = await litellm.acompletion(
            model="openrouter/google/gemini-2.0-flash-001",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    except Exception as e:
        print(f"⚠️ Ошибка в валидаторе: {e}")
        return {"is_valid": True}




@register_tool(tags=["system"], terminal=True)
def terminate(message: str = "Thank you for using our travel planning service!") -> str:
    """Terminates the agent's execution with a final message.
    
    Args:
        message: The final message to return before terminating
        
    Returns:
        The message with a termination note appended
    """
    return f"{message}\nДо связи..."

# Custom environment to handle state management
class TravelAgentEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.state = agent_state
    
    def execute_action(self, action, args: dict) -> dict:
        """Execute an action and return the result with state management."""
        try:
            result = action.execute(**args)
            
            # Update state based on action type
            if "sequential" in action.name or "main" in action.name:
                self._handle_sequential_response(result)
            elif "error" in action.name:
                self._handle_error_response(result)
            elif "terminate" in action.name:
                agent_state["conversation_active"] = False
                agent_state["goal_completed"] = True
            
            return self.format_result(result)
        except Exception as e:
            return {
                "tool_executed": False,
                "error": str(e),
                "traceback": str(e)
            }
    
    def _handle_sequential_response(self, result):
        """Handle response for sequential travel planning."""
        current_goal = agent_state["current_goal"]
        
        # Store the user's response based on current goal
        if current_goal == 1:
            agent_state["user_responses"]["trip_type"] = "User specified trip type"
        elif current_goal == 2:
            agent_state["user_responses"]["destination"] = "User specified destination"
        elif current_goal == 3:
            agent_state["user_responses"]["group_size"] = "User specified group size"
        elif current_goal == 4:
            agent_state["user_responses"]["travel_dates"] = "User specified dates"
        elif current_goal == 5:
            agent_state["user_responses"]["departure_city"] = "User specified departure city"
        elif current_goal == 6:
            # Summary generated, conversation complete
            agent_state["conversation_active"] = False
            agent_state["goal_completed"] = True
            return
        
        # Move to next goal
        agent_state["current_goal"] += 1
        agent_state["error_count"] = 0
    
    def _handle_error_response(self, result):
        """Handle error response."""
        agent_state["error_count"] += 1
        # Stay on current goal to retry

def create_travel_agent():
    """Create and configure the advanced travel agent"""
    
    # Reset global state
    global agent_state
    agent_state = {
        "current_goal": 1,
        "user_responses": {},
        "error_count": 0,
        "max_errors": 11,
        "goal_completed": False,
        "conversation_active": True,
        "has_asked_goal_1": False,
        "dynamic_questions_count": 0,
        "dynamic_completed": False,
    }
    
    # Define the agent language and environment
    agent_language = AgentFunctionCallingActionLanguage()
    environment = TravelAgentEnvironment()
    
    # Create the agent with the specified goals and tools
    travel_agent = Agent(
        goals=goals,
        agent_language=AgentFunctionCallingActionLanguage(),
        # The ActionRegistry automatically loads tools with these tags
        action_registry=PythonActionRegistry(tags=["sequential", "main", "error_handling", "system"]),
        generate_response=generate_response,
        environment=environment
    )
    
    return travel_agent

def _skip_goals_with_existing_data():
    """Продвигаем current_goal до первой цели, для которой ещё нет данных в user_responses."""
    responses = agent_state["user_responses"]
    while True:
        current_goal = agent_state["current_goal"]
        if current_goal == 1 and responses.get("travel_dates"):
            agent_state["current_goal"] = 2
            continue
        if current_goal == 2 and responses.get("group_size"):
            agent_state["current_goal"] = 3
            continue
        if current_goal == 3 and responses.get("children_exist"):
            if "нет" in (responses.get("children_exist") or "").lower() or "no" in (responses.get("children_exist") or "").lower():
                agent_state["current_goal"] = 5
            else:
                agent_state["current_goal"] = 4
            continue
        if current_goal == 4 and responses.get("children_age"):
            agent_state["current_goal"] = 5
            continue
        if current_goal == 5 and responses.get("destination"):
            agent_state["current_goal"] = 6
            continue
        if current_goal == 6 and responses.get("budget"):
            agent_state["current_goal"] = 7
            continue
        if current_goal == 7 and responses.get("departure_city"):
            agent_state["current_goal"] = 8
            continue
        if current_goal == 9 and responses.get("final_notes"):
            agent_state["current_goal"] = 10
            continue
        break


async def run_travel_agent_with_input(user_input: str):
    """Run the travel agent with provided input and return the memory (async wrapper)."""
    from game.core import Memory
    memory = Memory()
    memory.add_memory({"type": "user", "content": user_input})
    
    # Пропуск целей, для которых данные уже есть (избегаем повторных вопросов про даты/кол-во и т.д.)
    _skip_goals_with_existing_data()
    current_goal = agent_state["current_goal"]
    
    # Generate next question or response based on current goal
    if current_goal == 2:
        response = ask_group_size()
    elif current_goal == 3:
        response = ask_children_exist()
    elif current_goal == 4:
        response = ask_children_age()
    elif current_goal == 5:
        response = ask_destination_preferences()
    elif current_goal == 6:
        response = ask_budget()
    elif current_goal == 7:
        response = ask_departure_city()
    elif current_goal == 8:
        # Первая итерация динамического цикла Perplexity
        responses = agent_state["user_responses"]
        trip_type = responses.get("trip_type", "Организованный туризм через турфирму")
        destination = responses.get("destination", "")
        group_size = responses.get("group_size", "")
        travel_dates = responses.get("travel_dates", "")
        departure_city = responses.get("departure_city", "")
        budget = responses.get("budget", "")
        children_info = responses.get("children_info", "")
        dialogue_history = responses.get("dialogue_history", [])
        history_dialogue = "\n".join(dialogue_history) if dialogue_history else ""
        
        # Вызываем новую функцию для первого вопроса динамического интервью
        # Передаем пустое сообщение от пользователя, так как это инициация этапа
        response = await run_dynamic_interview(agent_state, "Пожалуйста, задай первый уточняющий вопрос из твоего списка.", history_dialogue)
        
        agent_state["last_agent_response"] = response
        agent_state["dynamic_questions_count"] = agent_state.get("dynamic_questions_count", 0) + 1
    elif current_goal == 9:
        response = ask_final_catch_all()
    elif current_goal == 10:
        # Завершающее сообщение и терминатор
        final_message = (
            "Отлично! Все ваши вводные учтем максимально. Эксперт подготовит для вас "
            "подборку туров, и свяжется с вами"
        )
        response = terminate(final_message)
        agent_state["conversation_active"] = False
        agent_state["goal_completed"] = True
    else:
        response = "Travel planning session completed. Thank you!"
    memory.add_memory({"type": "assistant", "content": response})
    agent_state["last_agent_response"] = response
    return memory

#
async def process_user_response(user_response: str):
    global agent_state
    current_goal = agent_state["current_goal"]
    responses = agent_state["user_responses"]
    from game.core import Memory

    # --- 1. ОПРЕДЕЛЯЕМ ВОПРОС ДЛЯ ВАЛИДАТОРА ---
    question_text_map = {
        1: "Пожалуйста, укажите примерные или точные даты поездки",
        2: "Хорошо. Кто поедет? Сколько взрослых?",
        3: "Поедут ли с вами дети?",
        4: "Уточните возраст детей?",
        5: "Куда Вы хотели ли бы поехать?",
        6: "Хорошо. В какой примерный общий бюджет хотели бы уложиться?",
        7: "Откуда планируете стартовать?"
    }

    if current_goal in [8, 9]:
        question_asked = agent_state.get("last_agent_response", "")
    else:
        question_asked = question_text_map.get(current_goal, "")

    # --- 2. ВАЛИДАЦИЯ ---
    if question_asked:
        validation_result = await validate_user_input(question_asked, user_response)
        if not validation_result.get("is_valid", True):
            reason = validation_result.get("reason") or "Пожалуйста, ответьте на вопрос по существу"
            memory = Memory()
            memory.add_memory({"type": "user", "content": user_response})
            memory.add_memory({"type": "assistant", "content": reason})
            return memory

    # --- 3. ЖЕСТКАЯ ЛОГИКА ШАГОВ 1-7 ---
    if current_goal == 1:
        responses["travel_dates"] = user_response
        agent_state["current_goal"] = 2
        agent_state["dynamic_questions_count"] = 0
    elif current_goal == 2:
        responses["group_size"] = user_response
        agent_state["current_goal"] = 3
    elif current_goal == 3:
        responses["children_info"] = user_response
        if any(word in user_response.lower() for word in ["да", "есть", "ребенок", "дети"]):
            agent_state["current_goal"] = 4
        else:
            agent_state["current_goal"] = 5
    elif current_goal == 4:
        responses["children_ages"] = user_response
        agent_state["current_goal"] = 5
    elif current_goal == 5:
        responses["destination"] = user_response
        agent_state["current_goal"] = 6
    elif current_goal == 6:
        responses["budget"] = user_response
        agent_state["current_goal"] = 7
    elif current_goal == 7:
        responses["departure_city"] = user_response
        agent_state["current_goal"] = 8
        responses["dialogue_history"] = []
        first_q = "Вам будет удобнее ответить на пару уточняющих вопросов здесь в чате или лучше созвониться?"
        agent_state["last_agent_response"] = first_q
        memory = Memory()
        memory.add_memory({"type": "assistant", "content": first_q})
        return memory

        # --- 4. ШАГ 8: ДИНАМИЧЕСКИЙ ДИАЛОГ ---
    elif current_goal == 8:
        if "dialogue_history" not in responses:
            responses["dialogue_history"] = []

        # Записываем ответ в историю
        last_q = agent_state.get("last_agent_response", "")
        if last_q:
            history_entry = f"Менеджер: {last_q} | Клиент: {user_response}"
            if history_entry not in responses["dialogue_history"]:
                responses["dialogue_history"].append(history_entry)

        # ЖЕСТКИЙ ЛИМИТ: Если это уже 11-й вопрос, принудительно уходим на 9-й шаг
        agent_state["dynamic_questions_count"] = agent_state.get("dynamic_questions_count", 0) + 1
        if agent_state["dynamic_questions_count"] > 10:
            agent_state["current_goal"] = 9
            final_q = "Спасибо за Ваши ответы. Подскажите, какие еще моменты важно учесть при составлении подборки туров, которые ранее не обсудили?"
            agent_state["last_agent_response"] = final_q
            memory = Memory()
            memory.add_memory({"type": "assistant", "content": final_q})
            return memory

        history_str = "\n".join(responses["dialogue_history"])
        llm_answer = await run_dynamic_interview(agent_state, user_response, history_str)

        # ПРОВЕРКА: Если ИИ сам решил перейти к финалу
        if "какие еще моменты важно учесть" in llm_answer.lower():
            agent_state["current_goal"] = 9
            agent_state["last_agent_response"] = llm_answer
            memory = Memory()
            memory.add_memory({"type": "assistant", "content": llm_answer})
            return memory

        # ПРОВЕРКА: Если ИИ получил номер телефона
        if "учтем максимально" in llm_answer.lower():
            agent_state["conversation_active"] = False
            agent_state["goal_completed"] = True
            agent_state["current_goal"] = 10
            llm_answer = "Отлично! Все ваши вводные учтем максимально. Эксперт подготовит для вас подборку туров и свяжется с вами."

        agent_state["last_agent_response"] = llm_answer
        memory = Memory()
        memory.add_memory({"type": "assistant", "content": llm_answer})
        return memory

    # --- 5. ШАГ 9: ФИНАЛЬНЫЙ ВОПРОС (ЗДЕСЬ БЫЛ БАГ) ---
    elif current_goal == 9:
        responses["final_notes"] = user_response

        # ГАРАНТИРОВАННОЕ ЗАВЕРШЕНИЕ ДИАЛОГА
        agent_state["conversation_active"] = False
        agent_state["goal_completed"] = True
        agent_state["current_goal"] = 10

        final_msg = "Отлично! Все ваши вводные учтем максимально. Эксперт подготовит для вас подборку туров и свяжется с вами."
        agent_state["last_agent_response"] = final_msg

        memory = Memory()
        memory.add_memory({"type": "assistant", "content": final_msg})
        return memory

    return await run_travel_agent_with_input(user_response)


async def process_travel_request(message: str, user_id: str = None) -> Dict[str, Any]:
    # (Код этой функции остается БЕЗ ИЗМЕНЕНИЙ, он работает корректно)
    try:
        keywords = ["привет", "здравствуйте", "старт", "начать", "здрасте", "приветствую"]
        message_text = message.lower()

        # Разбиваем сообщение на слова и проверяем каждое
        is_match = False
        for word in message_text.split():
            for key in keywords:
                # score — это процент схожести (от 0 до 100)
                score = fuzz.ratio(word, key)
                if score > 80:  # 80 — оптимальный порог для опечаток
                    is_match = True
                    break
        if is_match or message_text.startswith('/start'):
            reset_agent_state()

        conversation_active = agent_state.get("conversation_active", True)
        current_goal = agent_state.get("current_goal", 1)

        if not conversation_active:
            return {
                "memory": [],
                "status": "completed",
                "text": None,
                "current_goal": current_goal,
                "conversation_active": False
            }

        final_memory = await process_user_response(message)

        memory_list = []
        for item in final_memory.get_memories():
            memory_list.append({
                "type": item["type"],
                "content": item["content"]
            })
        if user_id:
            for item in reversed(memory_list):
                if item["type"] == "assistant":
                    log_conversation(user_id, "Ассистент", item["content"])
                    break

        if agent_state["current_goal"] > 8 or agent_state.get("goal_completed", False):
            status = "completed"
        else:
            status = "in_progress"

        assistant_text = None
        for item in reversed(memory_list):
            if item["type"] == "assistant":
                assistant_text = item["content"]
                break

        if assistant_text:
            assistant_text = assistant_text.strip()

            # 1. Если нейронка по ошибке влепила точку в самом конце — сносим её
            if assistant_text.endswith('.'):
                assistant_text = assistant_text[:-1]

            # 2. Исключения: фразы, которые точно НЕ должны заканчиваться вопросом
            # (добавь сюда любые другие финальные фразы из логики, если нужно)
            final_phrases = [
                "спасибо за ваши ответы",
                "отлично! все ваши вводные учтем",
                "я здесь, чтобы помочь"
            ]

            is_final = any(phrase in assistant_text.lower() for phrase in final_phrases)

            # 3. Если это не финальная фраза и на конце нет вопроса — принудительно ставим "?"
            if not is_final and not assistant_text.endswith('?'):
                assistant_text += '?'

        import asyncio
        import random
        await asyncio.sleep(random.randint(1, 1))

        return {
            "memory": memory_list,
            "status": status,
            "text": assistant_text,
            "current_goal": agent_state["current_goal"],
            "conversation_active": agent_state.get("conversation_active", True)
        }

    except Exception as e:
        return {
            "memory": [{"type": "error", "content": f"Error processing request: {str(e)}"}],
            "status": "error",
            "current_goal": agent_state.get("current_goal", 1),
            "conversation_active": False
        }


def run_travel_agent():
    print("🌍 Advanced Travel Agent - Comprehensive Travel Planning")
    print("=" * 62)
    print("Welcome! I'm here to help you plan your perfect trip.")
    print("I'll ask you a series of questions to understand your travel preferences.\n")

    user_input = input("Let's start planning your trip! Please tell me what you're looking for: ")

    if not user_input.strip():
        user_input = "I want to plan a trip"

    print("\n🤖 Agent is processing your request...")
    import asyncio
    final_memory = asyncio.run(run_travel_agent_with_input(user_input))

    print("\n" + "=" * 62)
    print("📝 AGENT MEMORY:")
    print("=" * 62)

    for item in final_memory.get_memories():
        print(f"\n{item['type'].upper()}: {item['content']}")

    print("\n" + "=" * 62)
    print("✅ Agent session completed!")


if __name__ == "__main__":
    run_travel_agent()
