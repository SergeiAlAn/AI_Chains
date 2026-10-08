import asyncio
import os
import sys
import random
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    logging.error("BOT_TOKEN не установлен.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)

router = Router()

class GameStates(StatesGroup):
    waiting_for_choice = State()

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await message.answer("Привет! Это бот для игры в Камень, Ножницы, Бумага. Напиши /help, чтобы узнать команды.")
    
@router.message(Command(commands=['help']))
async def cmd_help(message: Message) -> None:
    help_text = (
        "/start - Запускает бота и приветствует пользователя.\n"
        "/help - Предоставляет справочную информацию о командах и игре.\n"
        "/play - Начинает игру в Камень, Ножницы, Бумага."
    )
    await message.answer(help_text)

@router.message(Command(commands=['play']))
async def cmd_play(message: Message, state: FSMContext) -> None:
    await message.answer("Выберите: Камень, Ножницы или Бумага.")
    await state.set_state(GameStates.waiting_for_choice)

@router.message(GameStates.waiting_for_choice, F.text)
async def handle_user_choice(message: Message, state: FSMContext) -> None:
    user_choice = message.text.strip().lower()
    choices = ['камень', 'ножницы', 'бумага']
    
    if user_choice not in choices:
        await message.answer("Неверный выбор! Пожалуйста, выберите: Камень, Ножницы или Бумага.")
        return

    bot_choice = random.choice(choices)
    result = determine_winner(user_choice, bot_choice)
    
    await message.answer(f"Вы выбрали: {user_choice.capitalize()}\nБот выбрал: {bot_choice.capitalize()}\nРезультат: {result}")
    await state.clear()

def determine_winner(user_choice: str, bot_choice: str) -> str:
    if user_choice == bot_choice:
        return "Ничья!"
    elif (user_choice == 'камень' and bot_choice == 'ножницы') or \
         (user_choice == 'ножницы' and bot_choice == 'бумага') or \
         (user_choice == 'бумага' and bot_choice == 'камень'):
        return "Вы выиграли!"
    else:
        return "Вы проиграли!"

async def main() -> None:
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
