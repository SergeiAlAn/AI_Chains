# Генератор Telegram-ботов на LangChain

Скрипт принимает текстовое описание бота и с помощью цепочки LLM (через ProxyAPI)
генерирует готовый к запуску `.py`-файл бота на **aiogram 3.x**.

## Как устроена цепочка

```
описание ──► analysis_chain ──► code_chain ──► review_chain ──► generated_bot.py
```

| Звено | Что делает |
|---|---|
| `analysis_chain` | Разбирает задание и строит спецификацию (Pydantic `BotSpec`): ключевые функции, команды, обработчики, формат ответа, нужна ли БД, дополнительные зависимости, внешние API, план структуры кода |
| `code_chain` | Генерирует полный код бота по спецификации: aiogram 3.x, `async def`-хендлеры, `Router`, токен из `BOT_TOKEN`, запуск через `asyncio.run(main())` |
| `review_chain` | LLM-ревью и статическая проверка: синтаксис (`ast`/`compile`), импорты, отсутствие API aiogram 2.x, структура запуска, async-хендлеры, отсутствие заглушек. Если остались ошибки, код уходит на повторное исправление (до `MAX_REVIEW_ATTEMPTS` раз) |

Звенья собраны в LCEL-пайплайн (`RunnablePassthrough.assign`), каждый шаг и каждый
вызов LLM логируется (время, токены, найденные проблемы) в консоль и в `generation.log`.

## Установка

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Скопируйте `.env.example` в `.env` и укажите ключ ProxyAPI:

```bash
copy .env.example .env      # Windows
cp .env.example .env        # Linux / macOS
```

## Запуск генератора

```bash
python script.py "Бот, который отправляет случайные мемы"
```

Дополнительные опции:

```bash
# описание из файла, свой путь для результата, сохранить спецификацию
python script.py -f task.txt -o bots/meme_bot.py --save-spec
```

| Параметр | Описание |
|---|---|
| `description` | Техническое описание бота (строка) |
| `-f, --file` | Файл с описанием (вместо строки) |
| `-o, --output` | Путь к итоговому файлу бота (по умолчанию `generated_bot.py`) |
| `--save-spec` | Сохранить спецификацию от `analysis_chain` в `<имя>_spec.json` |

Результат:

- `generated_bot.py` — код бота;
- `generated_bot_requirements.txt` — зависимости бота;
- `generation.log` — лог генерации.

Коды возврата: `0` — успех, `1` — ошибка генерации, `2` — файл сохранён, но
статическая проверка нашла ошибки (подробности в логе).

## Запуск сгенерированного бота

```bash
pip install -r generated_bot_requirements.txt

# Windows (PowerShell)
$env:BOT_TOKEN="123456789:ABC..."
python generated_bot.py

# Linux / macOS
BOT_TOKEN="123456789:ABC..." python generated_bot.py
```

## Пример лога

```
20:41:02 | INFO     | bot_generator | Старт генерации. Модель: gpt-4o-mini, endpoint: https://api.proxyapi.ru/openai/v1
20:41:02 | INFO     | bot_generator | ▶ Шаг 'analysis_chain' запущен
20:41:07 | INFO     | bot_generator | LLM ответила за 4.8 c (prompt=612, completion=455, total=1067 токенов)
20:41:07 | INFO     | bot_generator | Команды: /start, /help, /meme
20:41:07 | INFO     | bot_generator | Доп. зависимости: aiohttp
20:41:07 | INFO     | bot_generator | ✔ Шаг 'analysis_chain' завершён за 4.9 c
20:41:07 | INFO     | bot_generator | ▶ Шаг 'code_chain' запущен
...
20:41:31 | INFO     | bot_generator | Код прошёл синтаксическую и структурную проверку
20:41:31 | INFO     | bot_generator | Код бота сохранён: C:\...\generated_bot.py
```

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `PROXYAPI_KEY` | — | Ключ ProxyAPI (обязательно) |
| `PROXYAPI_BASE_URL` | `https://api.proxyapi.ru/openai/v1` | OpenAI-совместимый endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | Модель для `analysis_chain` |
| `LLM_CODE_MODEL` | = `LLM_MODEL` | Модель для `code_chain` и `review_chain` (рекомендуется `gpt-4o`) |
| `LLM_TEMPERATURE` | `0.2` | Температура |
| `MAX_REVIEW_ATTEMPTS` | `3` | Максимум итераций ревью |
| `LOG_LEVEL` | `INFO` | Уровень логирования |
| `LOG_FILE` | `generation.log` | Файл лога |
| `BOT_TOKEN` | — | Токен Telegram для запуска сгенерированного бота |
