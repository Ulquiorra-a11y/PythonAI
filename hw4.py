import os
import logging

import google.generativeai as genai
from google.api_core.exceptions import (
    ResourceExhausted,  # 429 — превышен rate limit / квота
    DeadlineExceeded,  # таймаут
    ServiceUnavailable,  # 503 — временная недоступность
    InternalServerError,  # 500 — временная ошибка сервера
)
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Не найден ключ API. Задайте переменную окружения GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


@retry(
    stop=(stop_after_attempt(5) | stop_after_delay(60)),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(
        (ResourceExhausted, DeadlineExceeded, ServiceUnavailable, InternalServerError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _generate(prompt: str, request_timeout: int = 30):
    return model.generate_content(
        prompt,
        request_options={"timeout": request_timeout},
    )


def ask_gemini(prompt: str, request_timeout: int = 30) -> str:
    try:
        response = _generate(prompt, request_timeout=request_timeout)
        return response.text

    except ResourceExhausted:
        logger.error("Превышен лимит запросов (429). Все попытки исчерпаны.")
        return "Ошибка: превышен лимит запросов к API. Попробуйте позже."

    except DeadlineExceeded:
        logger.error("Таймаут запроса. Все попытки исчерпаны.")
        return "Ошибка: сервер не ответил вовремя (таймаут)."

    except (ServiceUnavailable, InternalServerError) as e:
        logger.error(f"Сервис временно недоступен: {e}")
        return "Ошибка: сервис временно недоступен, попробуйте позже."

    except Exception as e:
        logger.exception("Непредвиденная ошибка")
        return f"Непредвиденная ошибка: {e}"


if __name__ == "__main__":
    result = ask_gemini(
        "как реализовать безопасное хранение API-ключа (.env или переменные окружения)?"
    )
    print(result)