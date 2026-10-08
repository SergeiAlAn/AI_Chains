"""
Генератор Telegram-ботов (aiogram 3.x) по текстовому описанию.

Цепочка LangChain из трёх звеньев:
    1. analysis_chain — разбирает задание и строит спецификацию бота (JSON).
    2. code_chain     — генерирует полный Python-код бота по спецификации.
    3. review_chain   — проверяет код (LLM-ревью + статические проверки) и исправляет его.

Пример запуска:
    python script.py "Бот, который отправляет случайные мемы"
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Конфигурация и логирование
# ---------------------------------------------------------------------------

load_dotenv()

PROXYAPI_KEY = os.getenv("PROXYAPI_KEY", "")
PROXYAPI_BASE_URL = os.getenv("PROXYAPI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# Модель для code_chain и review_chain (качество кода сильно зависит от модели)
LLM_CODE_MODEL = os.getenv("LLM_CODE_MODEL") or LLM_MODEL
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_REVIEW_ATTEMPTS = int(os.getenv("MAX_REVIEW_ATTEMPTS", "3"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "generation.log")

# Корректный вывод кириллицы в консоль Windows
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
# Сторонние библиотеки слишком многословны на уровне INFO
for _noisy in ("httpx", "openai", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("bot_generator")


class LLMLoggingCallback(BaseCallbackHandler):
    """Логирует каждый вызов LLM: время ответа и расход токенов."""

    def __init__(self) -> None:
        self._started: dict[Any, float] = {}
        self.total_tokens = 0

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._started[run_id] = time.perf_counter()
        log.debug("LLM-запрос отправлен (run_id=%s)", run_id)

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs) -> None:
        elapsed = time.perf_counter() - self._started.pop(run_id, time.perf_counter())
        usage = (response.llm_output or {}).get("token_usage") or {}
        tokens = usage.get("total_tokens", 0)
        self.total_tokens += tokens
        log.info(
            "LLM ответила за %.1f c (prompt=%s, completion=%s, total=%s токенов)",
            elapsed,
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
            tokens or "?",
        )

    def on_llm_error(self, error: BaseException, *, run_id, **kwargs) -> None:
        self._started.pop(run_id, None)
        log.error("Ошибка вызова LLM: %s", error)


def build_llm(model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE) -> ChatOpenAI:
    """Создаёт клиента OpenAI-совместимой модели через ProxyAPI."""
    if not PROXYAPI_KEY:
        log.critical("Не задана переменная окружения PROXYAPI_KEY (см. .env.example)")
        sys.exit(1)
    return ChatOpenAI(
        model=model,
        api_key=PROXYAPI_KEY,
        base_url=PROXYAPI_BASE_URL,
        temperature=temperature,
        timeout=180,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# Схема спецификации бота (результат analysis_chain)
# ---------------------------------------------------------------------------


class BotCommand(BaseModel):
    name: str = Field(description="Команда без слэша, например 'start'")
    description: str = Field(description="Что делает команда")


class BotHandler(BaseModel):
    name: str = Field(description="Имя async-функции обработчика, например 'cmd_start'")
    trigger: str = Field(
        description="Что вызывает обработчик: 'command /start', 'text', 'photo', "
        "'callback_query data=...' и т.п."
    )
    description: str = Field(description="Логика обработчика")


class BotSpec(BaseModel):
    bot_name: str = Field(description="Короткое название бота")
    summary: str = Field(description="Краткое описание назначения бота")
    key_features: list[str] = Field(description="Ключевые функции бота")
    commands: list[BotCommand] = Field(description="Список команд бота")
    handlers: list[BotHandler] = Field(description="Список обработчиков (включая команды)")
    response_format: str = Field(
        description="Формат ответов: текст/HTML/фото/клавиатуры и т.п."
    )
    needs_database: bool = Field(description="Нужна ли база данных")
    database: Optional[str] = Field(
        default=None, description="Какая БД и что в ней хранить (если нужна)"
    )
    dependencies: list[str] = Field(
        description="Дополнительные pip-зависимости помимо aiogram (например 'aiohttp')"
    )
    external_apis: list[str] = Field(
        default_factory=list, description="Внешние API (желательно без ключей)"
    )
    env_vars: list[str] = Field(
        default_factory=list, description="Переменные окружения помимо BOT_TOKEN"
    )
    code_structure: list[str] = Field(
        description="План структуры кода: блоки/функции в порядке следования в файле"
    )


# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM = """Ты — senior Python-разработчик и архитектор Telegram-ботов на aiogram 3.x.
Проанализируй техническое задание и составь спецификацию бота.

Правила:
- Выдели ключевые функции и преврати их в команды и обработчики.
- Всегда включай команды start и help.
- Подбирай инструменты: для HTTP — aiohttp; для БД — aiosqlite (SQLite в локальном файле).
- Если нужны внешние данные — выбирай бесплатные публичные API без ключей
  (например, https://meme-api.com/gimme для мемов). Если ключ неизбежен — укажи его в env_vars.
- Не включай aiogram и модули стандартной библиотеки в dependencies.
- code_structure — пошаговый план файла: импорты, конфиг, клавиатуры, хендлеры, main().

{format_instructions}"""

ANALYSIS_HUMAN = "Техническое задание:\n{task}"

AIOGRAM3_SKELETON = """Эталонный каркас aiogram 3.x (эти строки корректны — используй их как есть):
import asyncio
import os
import sys
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
router = Router()

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None: ...

async def main() -> None:
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())"""

CODE_SYSTEM = """Ты — senior Python-разработчик. Напиши полный, рабочий код Telegram-бота
в одном файле строго по спецификации.

""" + AIOGRAM3_SKELETON + """

Обязательные требования:
- aiogram 3.x (НЕ 2.x): from aiogram import Bot, Dispatcher, Router, F;
  from aiogram.filters import Command, CommandStart; from aiogram.types import Message, CallbackQuery;
  ParseMode импортируется из aiogram.enums; parse_mode задаётся через
  Bot(token=..., default=DefaultBotProperties(parse_mode=...)) из aiogram.client.default.
- Все обработчики — async def, регистрируются через router = Router() и декораторы
  @router.message(...) / @router.callback_query(...); dp = Dispatcher(); dp.include_router(router).
- В начале файла вызывается load_dotenv() (from dotenv import load_dotenv), чтобы
  подхватить переменные из файла .env.
- Токен читается ТОЛЬКО из переменной окружения BOT_TOKEN через os.getenv;
  если его нет — логировать ошибку и завершиться через sys.exit(1).
- Запуск: async def main() с await dp.start_polling(bot), затем
  if __name__ == "__main__": asyncio.run(main()).
- Запрещено: executor, @dp.message_handler, Dispatcher(bot), types.ParseMode, Bot.set_current.
- Никаких заглушек, TODO, '...', пустых pass, вымышленных токенов и URL-плейсхолдеров —
  код должен запускаться без модификаций.
- Сетевые запросы — через aiohttp с таймаутом и try/except, с понятным ответом пользователю при ошибке.
- ОБЯЗАТЕЛЬНО вызови dp.include_router(router) — без этого ни один обработчик не сработает.
- Бот должен работать сразу после запуска, без ручного наполнения данными:
  если нужны данные (словари, слова, вопросы, цитаты) — встрой их в код константой
  (если в задании указан объём — встрой весь объём, иначе не меньше 50 реальных записей,
  без повторяющихся ключей); если используется БД — заполняй её этими данными при старте.
- Многошаговые сценарии (викторина, вопрос → ответ, анкета) реализуй через FSM aiogram
  (from aiogram.fsm.context import FSMContext; from aiogram.fsm.state import State, StatesGroup):
  каждый вопрос выбирай случайно (random.choice), ожидаемый ответ сохраняй через
  state.update_data(...), ответ принимай обработчиком с фильтром по состоянию
  (@router.message(QuizStates.waiting_for_answer, F.text)), сравнивай без учёта регистра и
  пробелов, сообщай результат и задавай следующий вопрос.
- Обработчик произвольного текста без фильтра состояния регистрируй ПОСЛЕДНИМ, после всех команд.
- Если нужна БД — aiosqlite, таблицы создаются автоматически при старте.
- Используй logging, добавляй краткие комментарии к ключевым блокам.
- Используй только зависимости из спецификации, aiogram и стандартную библиотеку.

Отвечай только кодом, без объяснений: один блок ```python ... ```."""

CODE_HUMAN = "Спецификация бота (JSON):\n{spec_json}"

REVIEW_SYSTEM = """Ты — строгий code reviewer Python и aiogram 3.x.
Проверь код Telegram-бота:
1. Нет ли синтаксических ошибок.
2. Корректная структура запуска (async def main, dp.start_polling(bot), asyncio.run(main())
   под if __name__ == "__main__").
3. Корректные импорты для aiogram 3.x (нет API из aiogram 2.x), все используемые имена импортированы.
4. Все хендлеры — async def; вызывается load_dotenv(); токен берётся из os.getenv("BOT_TOKEN").
5. Роутер подключён через dp.include_router(router) (хендлеры остаются на роутере).
6. Нет заглушек, TODO и комментариев вида "здесь будет логика"; реализованы все функции
   из спецификации; данные, без которых бот бесполезен, встроены в код.
7. Ошибки из статической проверки (ERROR) исправлены все до одной.
Не придумывай несуществующих проблем: всё, что есть в эталонном каркасе ниже, —
корректный API aiogram 3.x, его не нужно «исправлять».

""" + AIOGRAM3_SKELETON + """

Исправь все найденные проблемы. Формат ответа строго такой:
ISSUES:
- <проблема 1>
- <проблема 2>
(или "- нет", если проблем нет)

```python
<полный итоговый код файла>
```"""

REVIEW_HUMAN = """Спецификация бота (JSON):
{spec_json}

Результаты автоматической статической проверки:
{static_issues}

Код для проверки:
```python
{code}
```"""


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Достаёт Python-код из ответа LLM (самый длинный блок ```python```)."""
    blocks = CODE_BLOCK_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    # Ответ без markdown — считаем, что это чистый код
    return text.strip().strip("`").strip() + "\n"


def extract_issues(text: str) -> list[str]:
    """Достаёт список замечаний ревьюера из секции ISSUES."""
    head = text.split("```", 1)[0]
    issues = [
        line.strip()[1:].strip()
        for line in head.splitlines()
        if line.strip().startswith("-")
    ]
    return [i for i in issues if i.lower().rstrip(".") not in {"нет", "none", "no issues"}]


# Pip-пакеты, у которых имя модуля отличается от имени пакета
PIP_TO_MODULE = {
    "python-dotenv": "dotenv",
    "beautifulsoup4": "bs4",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "python-dateutil": "dateutil",
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
}


def _pip_to_module(pkg: str) -> str:
    name = re.split(r"[<>=!~\[ ;]", pkg.strip(), maxsplit=1)[0].lower()
    return PIP_TO_MODULE.get(name, name.replace("-", "_"))


def check_imports_resolve(tree: ast.AST, warnings: list[str]) -> list[str]:
    """Проверяет, что `from X import Y` действительно импортируется в текущем окружении."""
    errors: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.module and node.level == 0):
            continue
        try:
            module = importlib.import_module(node.module)
        except ModuleNotFoundError as exc:
            root = node.module.split(".")[0]
            if exc.name and exc.name.split(".")[0] == root and importlib.util.find_spec(root) is None:
                warnings.append(f"Пакет '{root}' не установлен — импорты из него не проверены")
            else:
                errors.append(f"Модуль '{node.module}' не существует")
            continue
        except Exception as exc:  # noqa: BLE001 — сторонний модуль может упасть при импорте
            warnings.append(f"Не удалось импортировать '{node.module}': {exc}")
            continue
        for alias in node.names:
            if alias.name == "*" or hasattr(module, alias.name):
                continue
            try:
                is_submodule = importlib.util.find_spec(f"{node.module}.{alias.name}") is not None
            except (ImportError, ValueError):
                is_submodule = False
            if not is_submodule:
                errors.append(
                    f"ImportError: в '{node.module}' нет имени '{alias.name}'"
                    + _import_hint(alias.name)
                )
    return errors


def _import_hint(name: str) -> str:
    known = {
        "DefaultBotProperties": "aiogram.client.default",
        "ParseMode": "aiogram.enums",
        "FSMContext": "aiogram.fsm.context",
        "State": "aiogram.fsm.state",
        "StatesGroup": "aiogram.fsm.state",
        "MemoryStorage": "aiogram.fsm.storage.memory",
        "InlineKeyboardBuilder": "aiogram.utils.keyboard",
        "ReplyKeyboardBuilder": "aiogram.utils.keyboard",
    }
    return f" (правильно: from {known[name]} import {name})" if name in known else ""


def check_bot_kwargs(tree: ast.AST) -> list[str]:
    """Проверяет именованные аргументы Bot(...) по сигнатуре установленного aiogram."""
    try:
        from aiogram import Bot
    except ImportError:
        return []
    params = {
        name
        for name, p in inspect.signature(Bot.__init__).parameters.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Bot":
            for kw in node.keywords:
                if kw.arg and kw.arg not in params:
                    errors.append(
                        f"Bot() не принимает аргумент '{kw.arg}' "
                        f"(допустимые: {', '.join(sorted(params))})"
                    )
    return errors


def check_handler_order(tree: ast.AST, handler_events: set[str]) -> list[str]:
    """Обработчик «любого текста» не должен стоять выше команд — иначе команды не сработают."""
    errors = []
    catch_all_seen: Optional[str] = None
    for node in getattr(tree, "body", []):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "message"):
                continue
            filters = [ast.unparse(a) for a in dec.args] + [ast.unparse(k) for k in dec.keywords]
            is_command = any("Command" in f for f in filters)
            is_catch_all = not filters or filters == ["F.text"]
            if is_command and catch_all_seen:
                errors.append(
                    f"Обработчик команды '{node.name}' зарегистрирован после "
                    f"'{catch_all_seen}' (ловит любой текст) и никогда не сработает — "
                    "перенеси общий обработчик текста в конец или добавь ему фильтр по состоянию"
                )
            elif is_catch_all:
                catch_all_seen = node.name
    return errors


def static_check(code: str, dependencies: list[str]) -> tuple[list[str], list[str]]:
    """
    Минимальная статическая проверка сгенерированного кода.
    Возвращает (errors, warnings): ошибки отправляются ревьюеру на исправление.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Синтаксис
    try:
        tree = ast.parse(code)
        compile(code, "<generated_bot>", "exec")
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} (строка {exc.lineno})"], warnings

    # 2. Импорты
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])

    if "aiogram" not in imported:
        errors.append("Не импортирован aiogram")

    allowed = {"aiogram", "dotenv"} | {_pip_to_module(d) for d in dependencies}
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    for module in sorted(imported - allowed - stdlib):
        warnings.append(f"Модуль '{module}' не указан в зависимостях спецификации")

    # Импортируемые имена реально существуют (проверяем по установленным пакетам)
    errors.extend(check_imports_resolve(tree, warnings))
    errors.extend(check_bot_kwargs(tree))

    # 3. API aiogram 2.x, которого нет в 3.x
    legacy_patterns = {
        r"from aiogram\.utils import[^\n]*\bexecutor\b|\bexecutor\.start_polling": "используется executor из aiogram 2.x",
        r"\.message_handler\(": "используется @dp.message_handler из aiogram 2.x",
        r"\.callback_query_handler\(": "используется callback_query_handler из aiogram 2.x",
        r"Dispatcher\(\s*bot": "Dispatcher(bot) — синтаксис aiogram 2.x",
        r"from aiogram\.types import[^\n]*\bParseMode\b": "ParseMode нужно импортировать из aiogram.enums",
        r"Bot\((?:(?!DefaultBotProperties)[^)])*parse_mode\s*=": "parse_mode в Bot() нужно передавать через DefaultBotProperties",
    }
    for pattern, message in legacy_patterns.items():
        if re.search(pattern, code):
            errors.append(message)

    # 4. Токен и запуск
    if "BOT_TOKEN" not in code or not re.search(r"os\.(getenv|environ)", code):
        errors.append("Токен должен читаться из переменной окружения BOT_TOKEN")
    if not re.search(r"\bload_dotenv\(", code):
        errors.append("Нет вызова load_dotenv() — BOT_TOKEN из .env не подхватится")
    if "start_polling" not in code:
        errors.append("Нет вызова dp.start_polling(bot)")
    if "asyncio.run(" not in code:
        errors.append("Нет запуска через asyncio.run(main())")
    if not re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", code):
        errors.append("Нет блока if __name__ == '__main__'")

    # 5. Хендлеры должны быть async, а роутеры — подключены к диспетчеру
    handler_events = {"message", "callback_query", "inline_query", "edited_message"}
    routers_with_handlers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if not (isinstance(target, ast.Attribute) and target.attr in handler_events):
                continue
            if isinstance(node, ast.FunctionDef):
                errors.append(f"Обработчик '{node.name}' должен быть async def")
            if isinstance(target.value, ast.Name):
                routers_with_handlers.add(target.value.id)

    errors.extend(check_handler_order(tree, handler_events))

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            dupes = sorted({str(k) for k in keys if keys.count(k) > 1})
            if dupes:
                errors.append(
                    f"Повторяющиеся ключи в словаре (строка {node.lineno}): {', '.join(dupes[:10])}"
                )

    dispatchers = {
        t.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "Dispatcher"
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    for name in sorted(routers_with_handlers - dispatchers):
        included = rf"include_router\(\s*{name}\s*\)|include_routers\([^)]*\b{name}\b"
        if not re.search(included, code):
            errors.append(
                f"Роутер '{name}' не подключён: нужен dp.include_router({name}), "
                "иначе ни один обработчик не сработает"
            )

    # 6. Заглушки
    if re.search(r"#\s*TODO|YOUR_[A-Z_]*TOKEN|<[A-Z_]+>", code):
        errors.append("В коде остались заглушки (TODO / YOUR_TOKEN / <PLACEHOLDER>)")
    stub_phrases = re.findall(
        r"здесь (?:будет|можно|нужно|должн\w*)|не реализован\w*|заглушк\w*"
        r"|not implemented",
        code,
        flags=re.IGNORECASE,
    )
    if stub_phrases:
        errors.append(
            "Логика не реализована до конца (найдено: "
            + ", ".join(sorted({p.lower() for p in stub_phrases}))
            + ") — допиши рабочую реализацию"
        )
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [
                s for s in node.body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                        and isinstance(s.value.value, str))
            ]
            if all(
                isinstance(s, ast.Pass)
                or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                    and s.value.value is Ellipsis)
                for s in body
            ):
                errors.append(f"Функция '{node.name}' пустая (pass / ...)")

    return errors, warnings


def format_static_report(errors: list[str], warnings: list[str]) -> str:
    lines = [f"ERROR: {e}" for e in errors] + [f"WARNING: {w}" for w in warnings]
    return "\n".join(lines) if lines else "Проблем не обнаружено."


def logged_step(name: str, func):
    """Оборачивает шаг цепочки в RunnableLambda с логированием начала/конца и времени."""

    def _wrapper(inputs: dict, config: RunnableConfig) -> Any:
        log.info("▶ Шаг '%s' запущен", name)
        started = time.perf_counter()
        try:
            result = func(inputs, config)
        except Exception:
            log.exception("✖ Шаг '%s' завершился с ошибкой", name)
            raise
        log.info("✔ Шаг '%s' завершён за %.1f c", name, time.perf_counter() - started)
        return result

    return RunnableLambda(_wrapper).with_config(run_name=name)


# ---------------------------------------------------------------------------
# Звенья цепочки
# ---------------------------------------------------------------------------


def build_analysis_chain(llm: ChatOpenAI) -> Runnable:
    """1. analysis_chain: task -> BotSpec."""
    parser = PydanticOutputParser(pydantic_object=BotSpec)
    prompt = ChatPromptTemplate.from_messages(
        [("system", ANALYSIS_SYSTEM), ("human", ANALYSIS_HUMAN)]
    ).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm | parser

    def run(inputs: dict, config: RunnableConfig) -> BotSpec:
        spec: BotSpec = chain.invoke({"task": inputs["task"]}, config=config)
        log.info("Бот: %s — %s", spec.bot_name, spec.summary)
        log.info("Ключевые функции: %s", "; ".join(spec.key_features))
        log.info("Команды: %s", ", ".join(f"/{c.name}" for c in spec.commands))
        log.info("Обработчики: %s", ", ".join(h.name for h in spec.handlers))
        log.info("Формат ответа: %s", spec.response_format)
        log.info(
            "База данных: %s",
            spec.database if spec.needs_database else "не требуется",
        )
        log.info("Доп. зависимости: %s", ", ".join(spec.dependencies) or "нет")
        log.debug("Структура кода: %s", spec.code_structure)
        return spec

    return logged_step("analysis_chain", run)


def build_code_chain(llm: ChatOpenAI) -> Runnable:
    """2. code_chain: BotSpec -> исходный код бота."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", CODE_SYSTEM), ("human", CODE_HUMAN)]
    )
    chain = prompt | llm | StrOutputParser() | RunnableLambda(extract_code)

    def run(inputs: dict, config: RunnableConfig) -> str:
        spec: BotSpec = inputs["spec"]
        code: str = chain.invoke(
            {"spec_json": spec.model_dump_json(indent=2)}, config=config
        )
        log.info("Сгенерировано %d строк кода", code.count("\n"))
        return code

    return logged_step("code_chain", run)


def build_review_chain(llm: ChatOpenAI) -> Runnable:
    """3. review_chain: LLM-ревью + статическая проверка, до MAX_REVIEW_ATTEMPTS итераций."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", REVIEW_SYSTEM), ("human", REVIEW_HUMAN)]
    )
    chain = prompt | llm | StrOutputParser()

    def run(inputs: dict, config: RunnableConfig) -> dict:
        spec: BotSpec = inputs["spec"]
        code: str = inputs["code"]
        spec_json = spec.model_dump_json(indent=2)

        errors, warnings = static_check(code, spec.dependencies)
        best = (code, errors, warnings)
        for attempt in range(1, MAX_REVIEW_ATTEMPTS + 1):
            log.info(
                "Ревью, попытка %d/%d (статических ошибок до ревью: %d)",
                attempt, MAX_REVIEW_ATTEMPTS, len(errors),
            )
            for e in errors:
                log.warning("  статическая ошибка: %s", e)

            answer = chain.invoke(
                {
                    "spec_json": spec_json,
                    "static_issues": format_static_report(errors, warnings),
                    "code": code,
                },
                config=config,
            )
            issues = extract_issues(answer)
            if issues:
                log.info("Ревьюер нашёл и исправил %d проблем(ы):", len(issues))
                for issue in issues:
                    log.info("  • %s", issue)
            else:
                log.info("Ревьюер замечаний не нашёл")

            # Версию ревьюера принимаем всегда: он исправляет и то, что статика не видит.
            # Если в ней появились статические ошибки — их исправит следующая итерация.
            code = extract_code(answer)
            errors, warnings = static_check(code, spec.dependencies)
            if len(errors) <= len(best[1]):
                best = (code, errors, warnings)
            if not errors:
                break
            log.warning("После ревью осталось статических ошибок: %d", len(errors))

        if errors:
            code, errors, warnings = best

        for w in warnings:
            log.warning("Предупреждение: %s", w)
        if errors:
            log.error("После %d попыток остались ошибки: %s", MAX_REVIEW_ATTEMPTS, errors)
        else:
            log.info("Код прошёл синтаксическую и структурную проверку")

        return {"code": code, "errors": errors, "warnings": warnings}

    return logged_step("review_chain", run)


def build_pipeline(llm: ChatOpenAI, code_llm: Optional[ChatOpenAI] = None) -> Runnable:
    """Полная цепочка: {task} -> {task, spec, code, review}."""
    code_llm = code_llm or llm
    return (
        RunnablePassthrough.assign(spec=build_analysis_chain(llm))
        | RunnablePassthrough.assign(code=build_code_chain(code_llm))
        | RunnablePassthrough.assign(review=build_review_chain(code_llm))
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_outputs(result: dict, output: Path, save_spec: bool) -> None:
    spec: BotSpec = result["spec"]
    code: str = result["review"]["code"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(code, encoding="utf-8")
    log.info("Код бота сохранён: %s", output.resolve())

    # requirements для сгенерированного бота
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    deps = ["aiogram>=3.4", "python-dotenv>=1.0"] + [
        d for d in spec.dependencies
        if _pip_to_module(d) not in stdlib and not d.lower().startswith("aiogram")
    ]
    req_path = output.with_name(f"{output.stem}_requirements.txt")
    req_path.write_text("\n".join(dict.fromkeys(deps)) + "\n", encoding="utf-8")
    log.info("Зависимости бота сохранены: %s", req_path.resolve())

    if save_spec:
        spec_path = output.with_name(f"{output.stem}_spec.json")
        spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        log.info("Спецификация сохранена: %s", spec_path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генерация Telegram-бота (aiogram 3.x) по текстовому описанию"
    )
    parser.add_argument("description", nargs="?", help="Техническое описание бота")
    parser.add_argument("-f", "--file", type=Path, help="Файл с техническим описанием")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("generated_bot.py"),
        help="Куда сохранить код бота (по умолчанию generated_bot.py)",
    )
    parser.add_argument(
        "--save-spec", action="store_true", help="Сохранить спецификацию бота в JSON"
    )
    args = parser.parse_args()
    if args.file:
        args.description = args.file.read_text(encoding="utf-8")
    if not args.description or not args.description.strip():
        parser.error("Передайте описание бота аргументом или через --file")
    return args


def main() -> int:
    args = parse_args()
    task = args.description.strip()

    log.info("=" * 60)
    log.info(
        "Старт генерации. Модель анализа: %s, модель кода: %s, endpoint: %s",
        LLM_MODEL, LLM_CODE_MODEL, PROXYAPI_BASE_URL,
    )
    log.info("Задание: %s", task)

    callback = LLMLoggingCallback()
    pipeline = build_pipeline(build_llm(LLM_MODEL), build_llm(LLM_CODE_MODEL))

    started = time.perf_counter()
    try:
        result = pipeline.invoke({"task": task}, config={"callbacks": [callback]})
    except Exception as exc:
        log.critical("Генерация прервана: %s", exc)
        return 1

    write_outputs(result, args.output, args.save_spec)

    log.info(
        "Готово за %.1f c, израсходовано токенов: %s",
        time.perf_counter() - started, callback.total_tokens or "?",
    )
    log.info("Запуск бота: set BOT_TOKEN=... && python %s", args.output)
    return 2 if result["review"]["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
