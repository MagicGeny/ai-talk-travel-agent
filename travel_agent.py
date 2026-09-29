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

# Import the Game framework
import game.core
importlib.reload(game.core)
from game.core import Environment, Goal, register_tool, PythonActionRegistry, Agent, \
    AgentFunctionCallingActionLanguage, generate_response

# Load environment variables from .env file
load_dotenv()

# Configure litellm for OpenRouter function calling
import litellm
litellm.set_verbose=True
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
    return "Есть ли пожелания по направлению/стране/городу назначения?"


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


def get_perplexity_recommendations(
    trip_type: str,
    destination: str,
    group_size: str,
    travel_dates: str,
    departure_city: str,
    budget: str,
    children_info: str,
    history_dialogue: str = "",
) -> str:
    """Get dynamic clarification question and short answer from Perplexity API."""
    import requests
    import os
    
    # Get API key from environment
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        # Fallback to hardcoded key for testing (remove in production)
        api_key = "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"
        if not api_key:
            # Debug information
            all_env_vars = {k: v for k, v in os.environ.items() if 'PERPLEXITY' in k or 'API' in k}
            debug_info = f"Доступные переменные окружения: {list(all_env_vars.keys())}"
            return f"❌ API ключ не найден. Пожалуйста, проверьте настройки.\n{debug_info}"
    
    # Construct the prompt according to new requirements
    prompt = f"""
Ты — ведущий и опытный тревел-эксперт оказывающий консультативную помощь для подбора тура. Ты продолжаешь уже начатый диалог, который тебе передают со мной. Критерии 1-8 (находятся в конце запроса).  Я - турист который хочет консультацию. 
Твоя задача — провести глубокую и вежливую консультацию и беседу, общаясь строго на \"Вы\". Для этого нужно пообщаться со мной, объяснить мне, так чтобы мне стало интересно то, что ты предлагаешь и созвониться с менеджером. Ты должен показать свою экспертность. Ты работаешь только с пакетными турами. 
Тебя никто не заставляет ничего продавать, главное помочь мне определиться с выбором тура и раскрыть мою потребность.
ОЧЕНЬ ВАЖНО:  в ответе давай только описание без названия цен! Если допытываюсь по поводу цен и стоимости, то вежливо предложи созвониться с менеджером. Цены может назвать мне менеджер на звонке в случае заинтересованности мной. 
Скажи что цены уточнит менеджер.  Всегда отвечай на русском языке даже в случае ошибок. Не отвечай в духе: "в доступных мне данных информация о турах в эту заброшенную страну отсутствует". Всегда помогай конкретно по тому направлению по которому пришел запрос в данный момент, остальные забудь, только если явно не спросят, начтут сомневаться и просить совета.


ФОРМАТ: 
Твой ответ должен состоять из двух частей.
1. Часть 1: Твой комментарий эксперта.
2. Часть 2: Уточняющий вопрос.
3. МЕЖДУ НИМИ ОБЯЗАТЕЛЬНО ДОЛЖНО БЫТЬ ДВА ПЕРЕНОСА СТРОКИ (пустая строка).
4. ФОРМАТ ОТВЕТА : [Конкретный ответ] + [Ссылка на отель или достопримечательности. Только если я настаиваю или прошу дополнительную информацию по отелям. Даже в этом случае, все равно нужно спрашивать хочу ли я ссылки на отели] + [Один короткий и конкретный вопрос].
5. ЛИМИТ:  Старайся уложиться в 21-25 слов для основного комментария, чтобы сохранить динамику диалога (ссылки и финальный вопрос в конце не считаются)


ПРАВИЛА ОТВЕТА и ТЕХНИЧЕСКИЙ РЕГЛАМЕНТ:
1. Информация должна быть только по сути, никакой воды.
2. НЕ использовать приветствия, фразы: 'Конечно', 'Я нашел для вас', 'Рад помочь', 'Согласно вашему запросу' и прочую воду в тексте,"Добрый день", "Здравствуйте", "Спасибо, что обратились" и т.д . по понятным причинам выше.
3.НЕ пересказывать мой запрос.
4. ФАКТЧЕКИНГ: Предлагай только то, что реально существует в выбранном направлении. Например: Не выдумывай пляжи и не задавай вопросы про них там, где их нет.
5. ВОПРОС: Задавай только ОДИН уточняющий вопрос, который логически вытекает из диалога. Задавай вопрос только на основе контекста диалога. Не используй общие шаблоны пляжного отдыха, если я выбрал направление где нет пляжного отдыха.
6. Если бюджет мой явно нереален, вежливо и экспертно объясни, сколько реально стоит такой тур, и предложи альтернативы. Без повода тоже не обрезай варианты.
7. НЕ использовать ломаные фразы. Лучше 23 красивых слов, чем 12 корявых. Пиши на естественном, грамотном русском языке. Если лимит слов мешает смыслу — отдай приоритет смыслу.
8. НЕ ставить точку на отдельной строке или использовать любые символы после финального вопроса.
9. СРАЗУ ПЕРЕХОДИ К СУТИ: дай комментарий по текущим данным (не более 2-3 предложений).
10. ФОРМАТ: Вопрос который в конце должен быть отделен от основного текста ПУСТОЙ СТРОКОЙ (двойной перенос строки) и идти отдельным абзацем!
11. СНОСКИ:  Твой ответ должен быть чистым текстом для человека. Если ты используешь данные из поиска, не вставляй цифровые сноски в сам текст, просто синтезируй информацию в единый текст.
12. Выдавай только обычный текст. Ссылки только если пользователь сам попросит и будет проявлять повышенный интерес.
Ссылки приветствуются в основном только на отели, достопримечательности. На перелеты и транспортные рейсы и остальное - не надо давать ссылки!
13. НЕ навязывай свое мнение, но при этом твои советы должны раскрывать мою потребность и твою экспертность. 
14. Задавай ВОПРОСЫ для уточнения потребностей.
15. Если прошу конкретику (отели, места и т.п.) — В конце такого ответа добавь вопрос: \"Хотите, я подготовлю для Вас подборку прямых ссылок на проверенные отели/достопримечательности по этому направлению?\".
16. Говори как живой человек, естественно.
17. Если я проявляю к чему-то интерес, например к отелю, то можешь при его запросе дать больше информации или ссылку на этот объект, но сам без спроса ничего не присылай.
18. НЕ предлагай Черногорию, Казахстан,  Россию, Италию и другие страны, если я явно о них не спросил.
19. Если я не определился с местом или путаюсь, не выбирай за  меня. Вместо этого задай наводящий вопрос (например, о предпочтительном климате, типе пляжа или длительности перелёта), чтобы помочь мне сузить выбор!
20. Также не надоедай постоянно вопросом в конце
когда предлагаешь созвониться или встретиться с менеджером  Делай это не чаще чем через 2-3 ответа и только когда это уместно.


ИНСТРУКЦИИ ПО ТОНУ:
- Говори как живой человек, но сохраняй статус эксперта.
- Будь вежлив, избегай фамильярности.
- Если чувствуешь, что я запутался, не навязывай страну и место (Черногорию, Казахстан, Россию, Италию  другие страны), если я явно о них не спросил, мягко помоги мне определиться вопросом для выбора места.

ТВОЙ ОТВЕТ:
- Максимально человечный и полезный комментарий по ситуации.
- Проанализируй всю переписку и в конце задай ОДИН уточняющий вопрос для следующего диалога или приближения моего желания поговорить с менеджером по туризму. Вопрос должен идти в самом конце ответа и самом последнем абзаце.
- Вопрос в конце должен раскрывать мою потребность и при этом чтобы каждый следующий вопрос не повторял предыдущие, проверял на адекватность


ВЫБОР УМНОГО ВОПРОСА:
- Выбирай ОДИН вопрос из предложенного списка умных вопросов, только если он идеально подходит под текущий контекст диалога.
- Перелёты: вопросы про рейсы и пересадки уместны, если детей нет или дети старше 10–11 лет.
- Пляж/звёздность/линия пляжа: вопросы про компромисс между первой линией и уровнем сервиса уместны, если бюджет ограничен.
- Экскурсии/активность: вопросы про экскурсии и активный отдых уместны, если уже обсуждается конкретное направление или тип отдыха.
- Пляж (песок/галька): уместно, если выбран пляжный отдых.
- Гибкость дат: уместно, если указаны жёсткие конкретные даты.
- Призыв к действию (предложение созвониться или встретиться с тур агентом): уместен, если диалог почти завершен и клиент понимает свои потребности.

ПРИМЕРНЫЙ СПИСОК УМНЫХ ДОПОЛНИТЕЛЬНЫХ ВОПРОСОВ (используй подходящий по контексту):
1. Отели: \"Какие отели Вы предпочитаете: с активной анимацией или более тихие, семейные?\"
2. Пляж: \"Какой берег Вам ближе: золотистый песок или аккуратная галька?\"
(вопрос уместен, если я выбрал пляжный отдых)
3. Логистика: \"Готовы ли Вы на рейсы с короткими пересадками, если это даст выгоду в цене?\" (только если дети старше 10 лет или их нет)
4. Бюджет: \"Допускаете ли Вы варианты отелей чуть дальше от моря, чтобы повысить уровень самого сервиса и сделать цену оптимальной?\"
5. Активность: \"Интересны ли Вам экскурсии, или в этот раз хочется максимально спокойного отдыха?\"
(вопрос уместен, если указано направление, где большой спрос на экскурсии или клиента не интересует ленивый отдых)
6. Целевое действие: \"Удобно ли Вам будет обсудить детали в коротком звонке или встретиться у нас в офисе за чашкой чая или кофе?\"
7. Активность: \"Планируете ли активный отдых или хотели бы более лениво или может баланс?\" (вопрос уместен, если детей нет, либо возраст детей более 7 лет)
8. Время: \"У Вас строгие ли даты поездки или возможны гибкие даты ±3 дня?\"
(вопрос уместен, если я выбрал конкретные даты поездки)
9. Целевое действие: \"Удобно ли Вам созвониться или встретиться у нас в офисе для обсуждения подборки туров?\"

АКТУАЛЬНЫЙ КОНТЕКСТ: Мы обсуждаем тур в {destination}. Забудь про все другие страны, обсуждавшиеся ранее, если я явно не попросил сменить направление

Используй следующие критерии:
Критерий 1 - Тип поездки: {trip_type}
Критерий 2 — Пункт назначения: {destination}
Критерий 3 - Количество человек: {group_size}
Критерий 4 — Даты поездки: {travel_dates}
Критерий 5 - Место отправления: {departure_city}
Критерий 6 - Бюджет: {budget}
Критерий 7 - Наличие и возраст детей: {children_info}
Критерий 8 - История уточняющих вопросов и ответов: {history_dialogue}
"""

    try:
        # Make request to Perplexity API using the cheapest model
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "sonar",  # Valid model with web search capabilities
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1100,
            "temperature": 0.25
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=100
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # ВОТ ЭТА ПРАВКА: вызываем функцию очистки перед тем как вернуть текст
            return clean_perplexity_response(content)
        else:
            # 1. Записываем ошибку в лог, чтобы видеть её в PyCharm/PM2
            logger.error(
                f"❌ Ошибка API: {response.status_code} - {response.text} - {response.text}")
            return get_perplexity_recommendations(
                trip_type,
                destination,
                group_size,
                travel_dates,
                departure_city,
                budget,
                children_info,
                history_dialogue
            )

    except (requests.exceptions.Timeout, requests.exceptions.RequestException, Exception) as e:
        # 1. Записываем ошибку в лог, чтобы видеть её в PyCharm/PM2
        logger.error(f"Произошла ошибка: {str(e)}. Пробую вызвать функцию get_perplexity_recommendations еще раз...")
        # 2. Небольшая пауза, чтобы не спамить API мгновенно при ошибке сети
        import time
        time.sleep(2)
        # 3. РЕКУРСИЯ: функция вызывает саму себя с теми же параметрами
        return get_perplexity_recommendations(
            trip_type,
            destination,
            group_size,
            travel_dates,
            departure_city,
            budget,
            children_info,
            history_dialogue
        )

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
    """
    Валидация ответа с упором на понимание смысла и прощение опечаток.
    """
    prompt = f"""
    Ты — опытный и понимающий и харизматичный тревел-эксперт. Твоя цель — понять, ответил ли я на вопрос, даже если я допустил опечатки. Неважно где они, в конце или начале. 

    ВОПРОС: "{question_asked}"
    ОТВЕТ ПОЛЬЗОВАТЕЛЯ: "{user_answer}"

    ТВОЯ ЗАДАЧА:
    1. Если в ответе есть ПОНЯТНЫЙ СМЫСЛ, подходящий к вопросу (например, "актиной" вместо "активной", "масква" вместо "Москва", "да" вместо "Да, хочу" и так дадлее по всем пунктам) — считай ответ ВАЛИДНЫМ (true).
    2. Опечатки, сокращения или пропуск букв — это НОРМАЛЬНО. Не будь занудой.
    3. Если вопрос про количество людей, и я ввел просто цифру ('1', '2', '5') — это ВАЛИДНЫЙ ответ.
    4. Если вопрос про даты, и я ввел название месяца ('сентябрь') — это тоже ВАЛИДНЫЙ ответ.
    5. Будь максимально лоялен. Если из ответа понятна суть, возвращай is_valid: True."
    6. НЕВАЛИДНЫМ (false) считай только полный бред:
       - Случайный набор букв (фыва, ghj, th).
       - Одиночные символы, не несущие смысла (., !, ?).
       - Ответы, которые технически невозможно соотнести с вопросом (например, на вопрос про отели ответить "синий").
       - на неадекварные ответы, вроде мата отвечай вежливо, но напоминай пользователю что от него ждут ответ на вопрос

    ФОРМАТ ОТВЕТА (JSON):
    Если смысл понятен: {{"is_valid": true}}
    Если совсем бред: {{"is_valid": false, "reason": "Придумай ОЧЕНЬ человечную, разную каждый раз фразу, почему ты не понял"}}
    """

    try:
        response = await asyncio.to_thread(
            litellm.completion,
            model="openrouter/google/gemini-2.0-flash-001",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()
        # Очистка от Markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        return result

    except Exception as e:
        print(f"⚠️ Ошибка в валидаторе: {e}")
        return {"is_valid": True}  # В любой непонятной ситуации пропускаем ответ




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
        1: "trip type (1, 2, or 3)",
        2: "destination (country, city, or resort)",
        3: "number of people traveling",
        4: "travel dates",
        5: "departure city"
    }
    
    current_question = goal_messages.get(current_goal, "the current question")
    
    return f"""I apologize, but I didn't understand your response clearly. 

Could you please answer the question about {current_question}? 

If you're unsure or embarrassed to answer, please let me know when would be a good time to ask you again about this.

Please provide a clear answer so we can continue with your travel planning."""

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
        
        response = get_perplexity_recommendations(
            trip_type,
            destination,
            group_size,
            travel_dates,
            departure_city,
            budget,
            children_info,
            history_dialogue,
        )
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

async def process_user_response(user_response: str):
    """Process user response and advance to next goal (async with smart validation)."""
    
    from game.core import Memory
    
    current_goal = agent_state["current_goal"]
    responses = agent_state["user_responses"]
    
    # Жестко фиксируем тип поездки
    responses["trip_type"] = "Организованный туризм через турфирму"
    
    # Подготовка текста вопроса для валидатора по текущей цели
    question_text_map = {
        1: "Пожалуйста, укажите даты поездки",
        2: "Хорошо. Кто поедет? Сколько взрослых?",
        3: "Поедут ли дети?",
        4: "Уточните возраст детей?",
        5: "Есть ли пожелания по направлению/стране/городу назначения?",
        6: "Хорошо. В какой общий бюджет хотели бы уложиться?",
        7: "Откуда планируете стартовать?",
        8: "Пожалуйста, ответьте на уточняющий вопрос по путешествию",
        9: (
            "Спасибо за Ваши ответы. Подскажите, какие еще моменты важно учесть "
            "при составлении подборки туров, которые ранее не обсудили?"
        ),
    }
    question_asked = question_text_map.get(current_goal, "")
    
    # Валидация ответа пользователя (защита от "дурака")
    if question_asked:
        validation_result = await validate_user_input(question_asked, user_response)
        if not validation_result.get("is_valid", True):
            # Не меняем current_goal, возвращаем вежливую просьбу
            reason = validation_result.get("reason") or question_asked
            memory = Memory()
            memory.add_memory({"type": "user", "content": user_response})
            memory.add_memory({"type": "assistant", "content": reason})
            return memory
    
    # Сохранение ответа и переходы по целям
    if current_goal == 1:
        # Первый ответ всегда трактуем как даты поездки
        responses["travel_dates"] = user_response
        agent_state["current_goal"] = 2
    elif current_goal == 2:
        responses["group_size"] = user_response
        agent_state["current_goal"] = 3
    elif current_goal == 3:
        responses["children_exist"] = user_response
        # Если детей нет, сразу прыгаем на Goal 5
        if "нет" in user_response.lower() or "no" in user_response.lower():
            responses["children_info"] = "Детей нет"
            agent_state["current_goal"] = 5
        else:
            agent_state["current_goal"] = 4
    elif current_goal == 4:
        responses["children_age"] = user_response
        # Формируем сводную информацию о детях
        responses["children_info"] = f"Дети: {user_response}"
        agent_state["current_goal"] = 5
    elif current_goal == 5:
        responses["destination"] = user_response
        # При смене направления полностью очищаем контекст для Perplexity (нет утечки старой страны)
        if "dialogue_history" in responses:
            responses["dialogue_history"] = []
        agent_state["dynamic_questions_count"] = 0
        agent_state["dynamic_completed"] = False
        agent_state["last_agent_response"] = ""
        agent_state["current_goal"] = 6
    elif current_goal == 6:
        responses["budget"] = user_response
        agent_state["current_goal"] = 7
    elif current_goal == 7:
        responses["departure_city"] = user_response
        agent_state["current_goal"] = 8
    elif current_goal == 8:
        # Динамический цикл Perplexity
        # Накапливаем историю уточняющих вопросов и ответов ТОЛЬКО после валидного ответа Perplexity
        last_agent_question = agent_state.get("last_agent_response", "")
        if "dialogue_history" not in responses:
            responses["dialogue_history"] = []
        
        # Увеличиваем счетчик динамических вопросов
        agent_state["dynamic_questions_count"] = agent_state.get("dynamic_questions_count", 0) + 1
        
        # Если достигнут лимит, переходим на Goal 9
        if agent_state["dynamic_questions_count"] >= 11:
            agent_state["dynamic_completed"] = True
            agent_state["current_goal"] = 9
            
            # Задаем финальный вопрос (Goal 9)
            memory = Memory()
            memory.add_memory({"type": "user", "content": user_response})
            memory.add_memory({"type": "assistant", "content": ask_final_catch_all()})
            agent_state["last_agent_response"] = ask_final_catch_all()
            return memory
        else:
            # Остаемся на Goal 8 и генерируем следующий вопрос через Perplexity
            trip_type = responses.get("trip_type", "Организованный туризм через турфирму")
            destination = responses.get("destination", "")
            group_size = responses.get("group_size", "")
            travel_dates = responses.get("travel_dates", "")
            departure_city = responses.get("departure_city", "")
            budget = responses.get("budget", "")
            children_info = responses.get("children_info", "")
            dialogue_history = responses.get("dialogue_history", [])
            history_dialogue = "\n".join(dialogue_history) if dialogue_history else ""

            # Retry logic: скрываем "восстание" модели от пользователя
            stop_phrases = [
                "perplexity", "я не могу", "i cannot", "поисковый ассистент",
                "ассистент", "прошу прощения",
                "я не являюсь", "i'm a search assistant"
            ]
            max_retries = 11
            attempt = 0
            valid_response = False
            perplexity_answer = ""

            while attempt < max_retries and not valid_response:
                attempt += 1

                # Твой вызов Perplexity (оставляем переменные как в коде)
                perplexity_answer = get_perplexity_recommendations(
                    trip_type, destination, group_size, travel_dates,
                    departure_city, budget, children_info, history_dialogue
                )

                # ПРОВЕРКА: если есть хоть одна фраза - входим в True
                is_bad = any(phrase.lower() in perplexity_answer.lower() for phrase in stop_phrases)

                if not is_bad:
                    valid_response = True
                else:
                    # Если ответ плохой - просто ждем секунду и идем на следующий круг While
                    await asyncio.sleep(1)

                # Если после всех попыток всё еще плохо - выдаем заглушку
            if not valid_response:
                perplexity_answer = "Подождите, пожалуйста, я сейчас занимаюсь вашим запросом. Подождете, хорошо?"
            else:
                # Очистка артефактов (убираем '\n.' в конце и лишние переносы/точки)
                perplexity_answer = re.sub(r"[\n\.]+\Z", "", perplexity_answer.strip())

                if "dialogue_history" not in responses:
                    responses["dialogue_history"] = []
                dialogue_entry = f"Вопрос: {agent_state.get('last_agent_response', '')} | Ответ: {user_response}"
                responses["dialogue_history"].append(dialogue_entry)

            agent_state["last_agent_response"] = perplexity_answer

            memory = Memory()
            memory.add_memory({"type": "user", "content": user_response})
            memory.add_memory({"type": "assistant", "content": perplexity_answer})
            return memory
    elif current_goal == 9:
        # Финальный открытый ответ перед завершением
        responses["final_notes"] = user_response
        agent_state["current_goal"] = 10
        # Следующее сообщение будет терминальным
        return await run_travel_agent_with_input("continue")
    elif current_goal == 10:
        # Если пользователь что-то пишет после финального сообщения, просто завершаем
        memory = Memory()
        memory.add_memory({"type": "user", "content": user_response})
        final_message = (
            "Отлично! Все ваши вводные учтем максимально. Эксперт подготовит для вас "
            "подборку туров, и свяжется с вами"
        )
        memory.add_memory({"type": "assistant", "content": terminate(final_message)})
        agent_state["conversation_active"] = False
        agent_state["goal_completed"] = True
        return memory
    
    # По умолчанию — отдаем управление генерации следующего шага
    return await run_travel_agent_with_input("continue")

async def process_travel_request(message: str, user_id: str = None) -> Dict[str, Any]:
    """
    Common function to process travel requests for both API and Telegram.
    
    Args:
        message: User's message/input
        user_id: Optional user ID for session management (for Telegram)
        
    Returns:
        Dictionary with response data including memory and status
    """
    try:
        # 1. Сначала проверяем, не является ли сообщение командой сброса
        if message.lower() in ["/start", "привет", "старт", "начать", "start"]:
            reset_agent_state()
            # После сброса диалог активен
        
        # 2. Проверяем состояние разговора
        conversation_active = agent_state.get("conversation_active", True)
        goal_completed = agent_state.get("goal_completed", False)
        current_goal = agent_state.get("current_goal", 1)

        # 3. Если диалог уже завершен — игнорируем последующие сообщения
        if not conversation_active:
            return {
                "memory": [],
                "status": "completed",
                "text": None,
                "current_goal": current_goal,
                "conversation_active": False
            }

        # Сбрасываем состояние только если разговор явно завершен или не активен
        # if goal_completed or not conversation_active:
        #     reset_agent_state()


        # Формируем запрос с явным указанием локации, чтобы очистить поиск Perplexity
        current_dest = agent_state["user_responses"].get("destination", "любое направление")
        #enriched_message = f"ЛОКАЦИЯ: {current_dest}. ЗАПРОС: {message}"

        # Всегда трактуем входящее сообщение как ответ на текущую цель
        final_memory = await process_user_response(message)

        # Convert memory to list format for JSON response
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

        # Determine status based on current goal
        if agent_state["current_goal"] > 8 or agent_state.get("goal_completed", False):
            status = "completed"
        else:
            status = "in_progress"

        # Получаем текст последнего сообщения ассистента для удобства
        assistant_text = None
        for item in reversed(memory_list):
            if item["type"] == "assistant":
                assistant_text = item["content"]
                break

        if assistant_text:
            # Убираем точку, если она висит отдельной строкой в конце
            assistant_text = assistant_text.replace('\n.', '').strip()

        # Имитация человеческой задержки перед ответом
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
    """Run the travel agent and display the final memory (standalone version)"""
    
    print("🌍 Advanced Travel Agent - Comprehensive Travel Planning")
    print("=" * 62)
    print("Welcome! I'm here to help you plan your perfect trip.")
    print("I'll ask you a series of questions to understand your travel preferences.\n")
    
    # Get user input
    user_input = input("Let's start planning your trip! Please tell me what you're looking for: ")
    
    if not user_input.strip():
        user_input = "I want to plan a trip"
    
    print("\n🤖 Agent is processing your request...")
    
    # Run the agent
    final_memory = asyncio.run(run_travel_agent_with_input(user_input))
    
    # Display the final memory
    print("\n" + "=" * 62)
    print("📝 AGENT MEMORY:")
    print("=" * 62)
    
    for item in final_memory.get_memories():
        print(f"\n{item['type'].upper()}: {item['content']}")
    
    print("\n" + "=" * 62)
    print("✅ Agent session completed!")

if __name__ == "__main__":
    run_travel_agent()
