"""
Server-side LiveAvatar REST helpers.

The API key stays on the server; the browser only ever receives short-lived
session tokens minted here.

Endpoints used (see https://docs.liveavatar.com):
  GET  /v1/avatars/public         , list public avatars (id, name, preview, default voice)
  POST /v1/contexts               , create a persona context (prompt + opening line)
  POST /v1/sessions/token         , mint a FULL-mode session token
"""

import httpx

from settings import get_settings

settings = get_settings()


class LiveAvatarError(Exception):
    pass


def _headers() -> dict:
    key = settings.LIVEAVATAR_API_KEY.get_secret_value()
    if not key:
        raise LiveAvatarError(
            "No LiveAvatar API key configured. Add LIVEAVATAR_API_KEY to your .env file."
        )
    return {"X-API-KEY": key, "Content-Type": "application/json"}


async def _request(method: str, path: str, **kwargs) -> dict:
    url = settings.LIVEAVATAR_API_URL.rstrip("/") + path
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request(method, url, headers=_headers(), **kwargs)
    if resp.status_code >= 400:
        raise LiveAvatarError(f"LiveAvatar API {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    return payload.get("data", payload)


async def list_public_avatars(page_size: int = 24) -> list[dict]:
    data = await _request("GET", f"/v1/avatars/public?page_size={page_size}")
    results = data.get("results", data if isinstance(data, list) else [])
    avatars = []
    for a in results:
        avatars.append(
            {
                "id": a.get("id"),
                "name": a.get("name"),
                "preview_url": a.get("preview_url"),
                "voice_id": (a.get("default_voice") or {}).get("id"),
            }
        )
    return avatars


async def create_context(name: str, prompt: str, opening_text: str) -> str:
    data = await _request(
        "POST",
        "/v1/contexts",
        json={"name": name[:80], "prompt": prompt, "opening_text": opening_text},
    )
    return data["id"]


async def create_full_session_token(
    avatar_id: str,
    voice_id: str | None,
    context_id: str | None,
    listening: bool,
    sandbox: bool | None = None,
) -> dict:
    """
    Mint a FULL-mode token. With a context the avatar converses (cloud mode);
    `listening=False` is used for local+avatar mode where the avatar only
    lip-syncs text we send via the SDK's repeat().
    """
    persona: dict = {}
    if context_id:
        persona["context_id"] = context_id
    if voice_id:
        persona["voice_id"] = voice_id
    body: dict = {
        "mode": "FULL",
        "avatar_id": avatar_id,
        "is_sandbox": settings.LIVEAVATAR_SANDBOX if sandbox is None else sandbox,
        "interactivity_type": "CONVERSATIONAL" if listening else "PUSH_TO_TALK",
    }
    if persona:
        body["avatar_persona"] = persona
    data = await _request("POST", "/v1/sessions/token", json=body)
    return {"session_id": data.get("session_id"), "session_token": data.get("session_token")}
