import requests
import json
import csv
from datetime import datetime


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"


def send_request(prompt: str, model: str = MODEL_NAME) -> str:

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


def run_inference(prompts: list[str]) -> list[dict]:

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Запрос: {prompt[:60]}...")
        response = send_request(prompt)
        print(f"         Ответ:  {response[:80]}...\n")
        results.append({"prompt": prompt, "response": response})

    return results


def save_report(results: list[dict], output_path: str = "inference_report.csv") -> None:

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "response"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Отчёт сохранён: {output_path}")


def print_report(results: list[dict]) -> None:

    print("\n" + "=" * 80)
    print("ОТЧЁТ ИНФЕРЕНСА")
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Модель: {MODEL_NAME}")
    print("=" * 80)

    for i, item in enumerate(results, 1):
        print(f"\n[{i}] ЗАПРОС:\n{item['prompt']}")
        print(f"\n    ОТВЕТ:\n{item['response']}")
        print("-" * 80)


PROMPTS = [
    "If you could have dinner with any historical figure, who would it be and why?",
    "Describe the taste of coffee to someone who has never tried it.",
    "What would happen to Earth if the Moon suddenly disappeared?",
    "Invent a name and brief description for a new sport that combines chess and swimming.",
    "What would happen if humans could photosynthesize like plants?",
    "Write a two-sentence horror story.",
    "Explain blockchain technology as if I am a 10-year-old.",
    "What are three unusual facts about octopuses?",
    "Suggest a creative gift idea for someone who already has everything.",
    "Explain why cats knock things off tables from the cat's perspective?",
    "Summarize the plot of Romeo and Juliet in 3 sentences.",
]


if __name__ == "__main__":
    print("Запуск инференса Qwen2.5:0.5B через Ollama...\n")

    results = run_inference(PROMPTS)

    print_report(results)
    save_report(results, "inference_report.csv")
