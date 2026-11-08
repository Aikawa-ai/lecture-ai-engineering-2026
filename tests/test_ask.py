"""実APIを呼ばず、SDKが次のリクエストへ会話履歴を渡すことを確認する。"""

from copy import deepcopy

import pytest
from google.genai import models, types

from app import ask


def test_chat_sends_previous_question_and_answer(monkeypatch, capsys):
    requests = []
    responses = iter(["覚えました。", "青いキツネです。"])

    def generate_content(self, *, model, contents, config=None):
        requests.append(deepcopy(contents))
        return types.GenerateContentResponse(
            candidates=[types.Candidate(
                content=types.Content(
                    role="model", parts=[types.Part(text=next(responses))]
                ),
                finish_reason="STOP",
            )],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10, candidates_token_count=5
            ),
        )

    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(models.Models, "generate_content", generate_content)
    entries = iter(["", "合言葉は何ですか？", "終了"])
    monkeypatch.setattr("builtins.input", lambda _: next(entries))
    ask.chat_loop("合言葉は青いキツネです。")

    assert len(requests) == 2
    assert [content.role for content in requests[1]] == ["user", "model", "user"]
    assert [content.parts[0].text for content in requests[1]] == [
        "合言葉は青いキツネです。", "覚えました。", "合言葉は何ですか？"
    ]
    output = capsys.readouterr().out
    assert "青いキツネです。" in output
    assert output.count("[入力トークン数] 10") == 2
    assert output.count("[出力トークン数] 5") == 2


@pytest.mark.parametrize("ending", ["exit", "quit", "終了", EOFError, KeyboardInterrupt])
def test_chat_exit_does_not_send_request(monkeypatch, ending):
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    def no_api(*args, **kwargs):
        pytest.fail("終了時にAPIを呼んではいけない")

    def read_input(_):
        if isinstance(ending, type):
            raise ending
        return ending

    monkeypatch.setattr(models.Models, "generate_content", no_api)
    monkeypatch.setattr("builtins.input", read_input)
    ask.chat_loop()


def test_default_question_still_works(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["ask"])
    questions = []
    monkeypatch.setattr(ask, "ask", lambda question: questions.append(question) or "回答")
    ask.main()
    assert questions == [ask.DEFAULT_QUESTION]
    assert "回答" in capsys.readouterr().out
