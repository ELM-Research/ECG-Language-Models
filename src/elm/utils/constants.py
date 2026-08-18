import re

ECG_TOKEN_PLACEHOLDER = "<ecg>"
THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"
RL_TOKENS = [THINK_START, THINK_END,
             ANSWER_START, ANSWER_END]

ROLES = {
"human": "user", "user": "user", "q": "user",
"assistant": "assistant", "gpt": "assistant", "model": "assistant", "a": "assistant",
"system": "system",
}
LEADING_PREFIX_RE = re.compile(
r"^\s*(?:(?:user|assistant|human|gpt|model|system|q|a)\s*[:：]\s*|[:：]\s*)+",
re.IGNORECASE,
)
TAG_RE = re.compile(r"<\s*(?:ecg|image)\s*>\s*", re.IGNORECASE)
IMAGE_WORD_RE = re.compile(r"\b(image|picture)\b", re.IGNORECASE)