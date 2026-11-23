"""第1回: LLMへの単発質問と、会話履歴を保つ対話。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話を続ける（exit / quit / 終了で終了）
    uv run python -m app.ask --chat "LoRAとは何ですか？"

通常は会話履歴を持たない単発の呼び出し。--chat では起動中だけ履歴を保持する。
ツールや用語集の参照は行わない。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    print_usage(response)
    return response.text


def print_usage(response: types.GenerateContentResponse) -> None:
    usage = response.usage_metadata
    if usage is not None:
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        print(f"[入力トークン数] {input_tokens if input_tokens is not None else '不明'}")
        print(f"[出力トークン数] {output_tokens if output_tokens is not None else '不明'}")
    else:
        print("[トークン数] 使用量情報を取得できませんでした")


def chat_loop(first_question: str | None = None) -> None:
    # 同じChatを使い続けると、SDKがユーザーとモデルの履歴を次の送信に含める。
    with genai.Client(api_key=get_api_key()) as client:
        chat = client.chats.create(model=MODEL)
        print("会話を開始します。exit / quit / 終了、または Ctrl+D で終了します。")
        question = first_question
        while True:
            if question is None:
                try:
                    question = input("\n[質問] ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n会話を終了します。")
                    return
            else:
                question = question.strip()
                print(f"[質問] {question}")

            if question.lower() in {"exit", "quit", "終了"}:
                print("会話を終了します。")
                return
            if question:
                response = chat.send_message(question)
                print_usage(response)
                print(response.text or "[回答テキストなし]")
            question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMに質問する（--chat で会話を継続）")
    parser.add_argument("question", nargs="?", help="最初の質問")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って質問を続ける")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            chat_loop(args.question)
        else:
            question = args.question if args.question is not None else DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
