"""LLM 공통 헬퍼: Claude(Anthropic) 와 GPT(OpenAI) 둘 다 지원.

config.yaml 의 llm.provider 가
  auto   : Claude 키 있으면 Claude, 없으면 OpenAI 키가 있으면 GPT, 둘 다 없으면 None(규칙 기반)
  claude : Claude 만
  openai : GPT 만
키는 프로젝트 루트 .env 파일 또는 환경변수 ANTHROPIC_API_KEY / OPENAI_API_KEY 에서 읽음.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILES = [ROOT / "api_keys.txt", ROOT / ".env"]   # 둘 다 읽음 (먼저 적힌 파일 우선)
_env_loaded = False
KEY_SOURCE: dict[str, str] = {}   # 어느 파일에서 읽었는지
BAD_KEYS: set[str] = set()         # 인증 실패한 제공자 (auto 모드에서 제외)


def load_env_file(paths: list[Path] | None = None) -> None:
    """api_keys.txt / .env 의 KEY=VALUE 를 환경변수로 등록 (이미 있으면 유지, 빈 값·한글 안내문은 무시)."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    for path in paths or ENV_FILES:
        if path.exists():
            _load_one(path)


def _load_one(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not k or not v or not v.isascii():   # "sk-ant-여기에_키" 같은 미입력 값은 건너뜀
            continue
        os.environ[k] = v          # 키 파일이 환경변수보다 우선
        KEY_SOURCE[k] = path.name


def key_source(cfg: dict) -> str:
    """현재 선택된 제공자의 키 출처 설명."""
    p = provider(cfg)
    k = "ANTHROPIC_API_KEY" if p == "claude" else "OPENAI_API_KEY"
    return KEY_SOURCE.get(k, "환경변수")


def _has_claude() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or Path("~/.config/anthropic").expanduser().exists())


def _has_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def provider(cfg: dict) -> str | None:
    load_env_file()
    want = (cfg.get("llm", {}).get("provider") or "auto").lower()
    if want == "claude":
        return "claude" if _has_claude() else None
    if want == "openai":
        return "openai" if _has_openai() else None
    if _has_claude() and "claude" not in BAD_KEYS:
        return "claude"
    if _has_openai() and "openai" not in BAD_KEYS:
        return "openai"
    return None


def available(cfg: dict) -> bool:
    return provider(cfg) is not None


def model_name(cfg: dict) -> str:
    p = provider(cfg)
    llm = cfg.get("llm", {})
    return llm.get("claude_model", "claude-opus-5-5") if p == "claude" else llm.get("openai_model", "gpt-5")


def _b64(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower().lstrip(".")
    media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    return media, base64.b64encode(path.read_bytes()).decode()


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
        e = max(text.rfind("}"), text.rfind("]"))
        if s < 0 or e < 0:
            raise
        return json.loads(text[s:e + 1])


# ---------------- Claude ----------------
def _ask_claude(cfg, prompt, images, max_tokens, effort, stream):
    import anthropic

    client = anthropic.Anthropic()
    content = []
    for p in images or []:
        media, data = _b64(p)
        content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
    content.append({"type": "text", "text": prompt})
    kwargs = dict(model=cfg["llm"].get("claude_model", "claude-opus-5-5"), max_tokens=max_tokens,
                  betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                  output_config={"effort": effort}, messages=[{"role": "user", "content": content}])
    try:
        if stream:
            with client.beta.messages.stream(**kwargs) as s:
                resp = s.get_final_message()
        else:
            resp = client.beta.messages.create(**kwargs)
    except anthropic.AuthenticationError as e:
        raise RuntimeError("Claude API 인증 실패: api_keys.txt(.env) 의 ANTHROPIC_API_KEY 확인") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError("Claude API 요청 한도 초과") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude API 오류 {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError("Claude API 네트워크 오류") from e
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude가 요청을 거부했습니다.")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Claude 응답이 잘렸습니다 (max_tokens).")
    return "".join(b.text for b in resp.content if b.type == "text")


# ---------------- OpenAI (GPT) ----------------
def openai_error(e) -> str:
    """OpenAI 오류를 사용자용 한 줄 메시지로."""
    body = getattr(e, "body", None) or {}
    err = body.get("error", body) if isinstance(body, dict) else {}
    code = (err.get("code") or err.get("type") or "") if isinstance(err, dict) else ""
    msg = (err.get("message") if isinstance(err, dict) else None) or str(e)
    if code in ("credit_balance_exhausted", "insufficient_quota"):
        return "OpenAI 크레딧 없음 → https://platform.openai.com/settings/organization/billing 에서 충전 필요"
    if code == "model_not_found":
        return f"OpenAI 모델을 찾을 수 없음 → config.yaml 의 llm.openai_model / illustrate.openai_image_model 확인 ({msg[:80]})"
    status = getattr(e, "status_code", "")
    if status == 429:
        return f"OpenAI 요청 한도 초과: {msg[:120]}"
    return f"OpenAI API 오류 {status} {code}: {msg[:120]}"


def _ask_openai(cfg, prompt, images, max_tokens, effort, stream):
    import openai

    client = openai.OpenAI()
    model = cfg["llm"].get("openai_model", "gpt-5")
    content = []
    for p in images or []:
        media, data = _b64(p)
        content.append({"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}})
    content.append({"type": "text", "text": prompt})
    kwargs = dict(model=model, messages=[{"role": "user", "content": content}], max_completion_tokens=max_tokens)
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["reasoning_effort"] = {"low": "low", "medium": "medium"}.get(effort, "high")
    try:
        try:
            resp = client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            if "reasoning_effort" in str(e) and "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort")
                resp = client.chat.completions.create(**kwargs)
            else:
                raise
    except openai.AuthenticationError as e:
        raise RuntimeError("OpenAI API 인증 실패: api_keys.txt(.env) 의 OPENAI_API_KEY 확인") from e
    except openai.APIStatusError as e:
        raise RuntimeError(openai_error(e)) from e
    except openai.APIConnectionError as e:
        raise RuntimeError("OpenAI API 네트워크 오류") from e
    choice = resp.choices[0]
    if choice.finish_reason == "content_filter":
        raise RuntimeError("OpenAI가 요청을 거부했습니다 (content_filter).")
    if choice.finish_reason == "length":
        raise RuntimeError("OpenAI 응답이 잘렸습니다 (max_completion_tokens).")
    return choice.message.content or ""


def ask(cfg: dict, prompt: str, images: list[Path] | None = None, max_tokens: int = 16000,
        effort: str = "medium", stream: bool = False) -> str:
    p = provider(cfg)
    auto = (cfg.get("llm", {}).get("provider") or "auto").lower() == "auto"
    if p == "claude":
        try:
            return _ask_claude(cfg, prompt, images, max_tokens, effort, stream)
        except RuntimeError as e:
            if auto and "인증 실패" in str(e) and _has_openai():
                BAD_KEYS.add("claude")
                print("  [LLM] Claude 키 인증 실패 → GPT(OpenAI)로 전환")
                return _ask_openai(cfg, prompt, images, max_tokens, effort, stream)
            raise
    if p == "openai":
        try:
            return _ask_openai(cfg, prompt, images, max_tokens, effort, stream)
        except RuntimeError as e:
            if auto and "인증 실패" in str(e) and _has_claude() and "claude" not in BAD_KEYS:
                BAD_KEYS.add("openai")
                print("  [LLM] OpenAI 키 인증 실패 → Claude로 전환")
                return _ask_claude(cfg, prompt, images, max_tokens, effort, stream)
            raise
    raise RuntimeError("LLM API 키가 없습니다 (api_keys.txt 의 ANTHROPIC_API_KEY 또는 OPENAI_API_KEY).")


def ask_json(cfg: dict, prompt: str, images: list[Path] | None = None, **kw):
    return _extract_json(ask(cfg, prompt, images, **kw))
