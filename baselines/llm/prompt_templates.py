"""Shared prompt and template constants for LLM train/infer."""

DEFAULT_CONTEXT_SEP = "</s>"

SYSTEM_PROMPT_TEMPLATES = {
    "ja": (
        "あなたはWEB検索エンジンのクエリ推薦システムです。\n"
        "次に与えられる検索セッションのクエリ列（{sep} 区切り）を読んで、"
        "ユーザーが次に入力しそうな検索クエリを1件推薦してください。"
    ),
    "en": (
        "You are a query suggestion system for a web search engine.\n"
        "Read the given sequence of search session queries separated by {sep}, "
        "and recommend one possible next query that the user is likely to enter."
    ),
}


def get_system_prompt(lang: str, sep: str = DEFAULT_CONTEXT_SEP) -> str:
    return SYSTEM_PROMPT_TEMPLATES[lang].format(sep=sep)


CHAT_TEMPLATES = {
    "ja": (
        "{% for message in messages -%}"
        "{%- if message['role'] == 'system' -%}"
        "{{ message['content'] ~ '\\n' }}"
        "{%- elif message['role'] == 'user' -%}"
        "{{ '検索セッション： ' ~ message['content'] ~ '\\n' }}"
        "{%- elif message['role'] == 'assistant' -%}"
        "{{ '推薦： ' ~ message['content'] }}"
        "{%- endif -%}"
        "{%- if loop.last and add_generation_prompt -%}"
        "{{ '推薦： ' }}"
        "{%- endif -%}"
        "{% endfor -%}"
    ),
    "en": (
        "{% for message in messages -%}"
        "{%- if message['role'] == 'system' -%}"
        "{{ message['content'] ~ '\\n' }}"
        "{%- elif message['role'] == 'user' -%}"
        "{{ 'Search session: ' ~ message['content'] ~ '\\n' }}"
        "{%- elif message['role'] == 'assistant' -%}"
        "{{ 'Suggestion: ' ~ message['content'] }}"
        "{%- endif -%}"
        "{%- if loop.last and add_generation_prompt -%}"
        "{{ 'Suggestion: ' }}"
        "{%- endif -%}"
        "{% endfor -%}"
    ),
}

RESPONSE_TEMPLATES = {"ja": "推薦：", "en": "Suggestion:"}
INSTRUCTION_TEMPLATES = {"ja": "検索セッション：", "en": "Search session:"}
