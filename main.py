import json
import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import litellm
from litellm import completion
from dotenv import load_dotenv
from simple_travel_agent import run_simple_travel_agent
from travel_agent import run_travel_agent_with_input, process_travel_request
from telegram_bot import process_webhook_update, set_webhook, get_bot_info, process_travel_agent_message
#import pydevd_pycharm

from vkbottle.bot import Bot, Message
from vkbottle import Keyboard, Text

# Твой класс агента (мозг)
# from core.agent import TravelAgent
botVK = Bot(token=os.environ.get("VK_POOL_KEY"))
# Load environment variables
#pydevd_pycharm.settrace('localhost', port=12388, stdoutToServer=True, stderrToServer=True)
load_dotenv()
litellm.set_verbose=True
#litellm.turn_on_debug()


# Configure litellm for OpenRouter function calling
# Note: add_function_to_prompt can cause issues with newer litellm versions
# litellm.add_function_to_prompt = True

app = FastAPI(title="AI Talk Travel Agent", description="A FastAPI application that forwards messages to neural networks using litellm")

class MessageRequest(BaseModel):
    message: str
    model: str = "openrouter/google/gemini-2.0-flash-exp:free"
    max_tokens: int = 1024

class MessageResponse(BaseModel):
    response: str
    model_used: str
    tokens_used: int = None

class TravelAgentRequest(BaseModel):
    message: str

class TravelAgentResponse(BaseModel):
    memory: list
    status: str

class TelegramWebhookRequest(BaseModel):
    update_id: int
    message: Dict[str, Any] = None
    edited_message: Dict[str, Any] = None
    channel_post: Dict[str, Any] = None
    edited_channel_post: Dict[str, Any] = None
    inline_query: Dict[str, Any] = None
    chosen_inline_result: Dict[str, Any] = None
    callback_query: Dict[str, Any] = None
    shipping_query: Dict[str, Any] = None
    pre_checkout_query: Dict[str, Any] = None
    poll: Dict[str, Any] = None
    poll_answer: Dict[str, Any] = None
    my_chat_member: Dict[str, Any] = None
    chat_member: Dict[str, Any] = None
    chat_join_request: Dict[str, Any] = None

def generate_ai_response(message: str, model: str = "openrouter/google/gemini-2.0-flash-exp:free", max_tokens: int = 1024) -> Dict[str, Any]:
    """
    Generate response using litellm, following the pattern from the reference implementation
    """
    try:
        # Prepare messages in the format expected by litellm
        messages = [
            {"role": "user", "content": message}
        ]
        
        # Call the LLM using litellm completion function
        response = completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        
        # Extract the response content
        result = response.choices[0].message.content
        
        return {
            "response": result,
            "model_used": model,
            "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "AI Talk Travel Agent API",
        "description": "Send messages to neural networks using litellm",
        "endpoints": {
            "/chat": "POST - Send a message and get AI response",
            "/travel-agent": "POST - Run travel agent with trip purpose interview",
            "/telegram/webhook": "POST - Telegram webhook endpoint",
            "/telegram/set-webhook": "POST - Set Telegram webhook URL",
            "/telegram/bot-info": "GET - Get Telegram bot information",
            "/health": "GET - Check API health"
        }
    }

@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """
    Accept a message and forward it to the neural network using litellm
    """
    try:
        result = generate_ai_response(
            message=request.message,
            model=request.model,
            max_tokens=request.max_tokens
        )
        
        return MessageResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/travel-agent", response_model=TravelAgentResponse)
async def travel_agent(request: TravelAgentRequest):
    """
    Run the travel agent to interview user about trip purpose
    """
    try:
        # Use the common function to process the travel request
        result = await process_travel_request(request.message)
        
        return TravelAgentResponse(
            memory=result["memory"],
            status=result["status"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running travel agent: {str(e)}")

@app.post("/telegram_webhook")
async def telegram_webhook(request: Request):
    """
    Telegram webhook endpoint to receive updates from Telegram
    """
    try:
        # Get the raw request body
        body = await request.body()
        update_data = json.loads(body)
        
        # Process the webhook update
        success = process_webhook_update(update_data)
        
        if success:
            return {"status": "ok"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process webhook update")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")

@app.post("/telegram/set-webhook")
async def set_telegram_webhook(request: Request):
    """
    Set Telegram webhook URL
    """
    try:
        body = await request.json()
        webhook_url = body.get("webhook_url")
        
        if not webhook_url:
            raise HTTPException(status_code=400, detail="webhook_url is required")
        
        success = set_webhook(webhook_url)
        
        if success:
            return {"status": "ok", "webhook_url": webhook_url}
        else:
            raise HTTPException(status_code=500, detail="Failed to set webhook")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting webhook: {str(e)}")

@app.get("/telegram/bot-info")
async def get_telegram_bot_info():
    """
    Get Telegram bot information
    """
    try:
        bot_info = get_bot_info()
        
        if bot_info:
            return {"status": "ok", "bot_info": bot_info}
        else:
            raise HTTPException(status_code=500, detail="Failed to get bot info")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting bot info: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ai-talk-travel-agent"}


async def keep_typing(peer_id):
    """Фоновая задача для поддержания статуса 'печатает'"""
    try:
        while True:
            await botVK.api.messages.set_activity(peer_id=peer_id, type="typing")
            await asyncio.sleep(5)  # Повторяем каждые 5 секунд
    except asyncio.CancelledError:
        # Задача будет отменена, когда придет ответ
        pass

@botVK.on.message()
async def travel_handler(message: Message):
    user_id = f"vk_{message.from_id}"  # Добавляем префикс платформы
    user_text = message.text

    # 0. Логируем входящее сообщение сразу в самом начале
    from travel_agent import log_conversation
    log_conversation(user_id, "Пользователь", user_text)

    # 1. Проверка команды сброса (ручной сброс) ПЕРЕД всем остальным
    if user_text.lower() in ["/start", "привет", "старт", "начать", "start"]:
        from telegram_bot import reset_user_session
        reset_user_session(user_id)
        await message.answer("Давайте начнем сначала:) Напишите когда вы планируете вашу поездку?")
        return

    # 2. Запускаем "вечное" печатание в фоновом режиме
    typing_task = asyncio.create_task(keep_typing(message.peer_id))
    # ---------------------

    try:
        # 3. Ждем ответа от агента (тут может быть долгая задержка)
        response = await process_travel_agent_message(user_id, user_text)

        # 4. Как только ответ готов, останавливаем "печатание"
        typing_task.cancel()

        # 5. Если ответ не "__ignore__", отправляем его
        if response != "__ignore__":
            await message.answer(response)

    except Exception as e:
        typing_task.cancel()
        print(f"Ошибка: {e}")
        await message.answer("Извините, подождите пожалуйста")

    # # Пример простого ответа:
    # if "привет" in user_text.lower():
    #     await message.answer(
    #         "Здравствуйте! Я ваш ИИ-помощник по путешествиям. "
    #         "Чтобы я подобрал лучший тур, скажите, когда вы планируете отпуск?"
    #     )
    # else:
    #     # Логика твоего агента
    #     # response = agent.get_answer(user_id, user_text)
    #     await message.answer("Интересный выбор! Записываю в блокнот...")


# --- ЗАПУСК (ИСПРАВЛЕННЫЙ) ---
if __name__ == "__main__":
    import uvicorn
    import asyncio


    async def run_vk_polling(bot):
        """
        Исправленный ручной запуск.
        Разворачивает пачки обновлений (updates) от ВК.
        """
        print("🤖 Обработчики VK запущены...")

        for hook in bot.loop_wrapper.on_startup:
            await hook()

        try:
            async for raw_response in bot.polling.listen():
                # ВК присылает либо одно событие, либо словарь с ключом 'updates'
                updates = []
                if isinstance(raw_response, dict):
                    if "updates" in raw_response:
                        updates = raw_response["updates"]
                    elif "type" in raw_response:
                        updates = [raw_response]

                for event in updates:
                    # Теперь проверяем каждое конкретное событие внутри пачки
                    if not isinstance(event, dict) or "type" not in event:
                        continue

                    print(f"✅ Обработка события: {event['type']}")

                    for view in bot.router.views.values():
                        try:
                            if await view.process_event(event):
                                await view.handle_event(event, bot.api, bot.state_dispenser)
                        except Exception as e:
                            print(f"⚠️ Ошибка во вьюхе {view.__class__.__name__}: {e}")

        except Exception as e:
            print(f"🛑 Критическая ошибка поллинга VK: {e}")
        finally:
            for hook in bot.loop_wrapper.on_shutdown:
                await hook()


    async def main():
        # Настройка сервера FastAPI
        config = uvicorn.Config(app, host="0.0.0.0", port=8080)
        server = uvicorn.Server(config)

        print("🚀 СИСТЕМА ЗАПУСКАЕТСЯ: API + VK BOT")

        # Запускаем две задачи параллельно:
        # 1. Сервер API (FastAPI)
        # 2. Наш ручной поллинг бота
        await asyncio.gather(
            server.serve(),
            run_vk_polling(botVK)
        )


    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Система остановлена пользователем")