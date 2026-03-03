import os
import json
import logging
import requests
import time
from pathlib import Path

from app.prompt_engine.groq_prompts.few_shot_specs import FEW_SHOT_SPECS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("few_shot_generator")


def groq_chat(api_key: str, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 4000, retries: int = 5):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    delay = 5
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429:
                raise requests.exceptions.HTTPError("429", response=resp)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            logger.warning(f"[FEW_SHOTS] Request failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise last_err


def sarvam_chat(api_key: str, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 1200, retries: int = 5):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": max_tokens,
    }
    delay = 5
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post("https://api.sarvam.ai/v1/chat/completions", headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                raise requests.exceptions.HTTPError("429", response=resp)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            logger.warning(f"[FEW_SHOTS] Sarvam request failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise last_err


def extract_json_array(raw: str):
    if not raw:
        return None
    raw = raw.strip().strip("`")
    if raw.startswith("[") and raw.endswith("]"):
        return raw
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return None


def extract_json_objects(raw: str):
    items = []
    if not raw:
        return items
    buf = ""
    depth = 0
    for ch in raw:
        if ch == "{":
            depth += 1
        if depth > 0:
            buf += ch
        if ch == "}":
            depth -= 1
            if depth == 0 and buf:
                try:
                    items.append(json.loads(buf))
                except Exception:
                    pass
                buf = ""
    return items


def main():
    api_key = os.getenv("GROQ_API_KEY")
    provider = os.getenv("FEW_SHOT_PROVIDER", "groq").lower()
    if provider == "sarvam":
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise SystemExit("Missing SARVAM_API_KEY in environment.")
    else:
        if not api_key:
            raise SystemExit("Missing GROQ_API_KEY in environment.")

    out_dir = Path(__file__).resolve().parents[1] / "app" / "prompt_engine" / "groq_prompts" / ("few_shots_sarvam" if provider == "sarvam" else "few_shots")
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_filter = os.getenv("FEW_SHOT_STAGES")
    allowed = None
    if stage_filter:
        allowed = {s.strip() for s in stage_filter.split(",") if s.strip()}

    for stage, spec in FEW_SHOT_SPECS.items():
        if allowed and stage not in allowed:
            continue
        n = int(os.getenv("FEW_SHOT_COUNT", spec["examples"]))
        model = spec["model"]
        generator_model = spec.get("generator_model", model)
        if provider == "sarvam":
            generator_model = "sarvam-m"
        system_prompt = "You are a data generator. Output ONLY a JSON array, no prose."
        filename = f"{stage}_{model.replace('/', '_').replace('-', '_')}.json"
        out_path = out_dir / filename
        overwrite = os.getenv("FEW_SHOT_OVERWRITE", "false").lower() == "true"
        if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
            logger.info(f"[FEW_SHOTS] Skipping existing {out_path}")
            continue
        batch_size = int(os.getenv("FEW_SHOT_BATCH_SIZE", "50"))
        total = 0
        data = []
        while total < n:
            remaining = n - total
            current = batch_size if remaining > batch_size else remaining
            user_prompt = spec["prompt"].format(n=current)
            logger.info(f"[FEW_SHOTS] Generating {current}/{n} examples for {stage} using {generator_model} ({provider})")
            if provider == "sarvam":
                raw = sarvam_chat(api_key, generator_model, system_prompt, user_prompt)
            else:
                raw = groq_chat(api_key, generator_model, system_prompt, user_prompt)
            try:
                batch = json.loads(raw)
            except Exception:
                cleaned = extract_json_array(raw)
                if cleaned:
                    batch = json.loads(cleaned)
                else:
                    batch = extract_json_objects(raw)
            if isinstance(batch, dict):
                # Some models wrap outputs; try to extract a list
                for v in batch.values():
                    if isinstance(v, list):
                        batch = v
                        break
            if not isinstance(batch, list):
                raise RuntimeError(f"Expected list from model for {stage}, got {type(batch)}")
            if not batch:
                # One retry with a stricter JSON-only instruction
                strict_prompt = user_prompt + "\n\nIMPORTANT: Output ONLY a JSON array. No prose, no markdown."
                if provider == "sarvam":
                    raw = sarvam_chat(api_key, generator_model, system_prompt, strict_prompt)
                else:
                    raw = groq_chat(api_key, generator_model, system_prompt, strict_prompt)
                try:
                    batch = json.loads(raw)
                except Exception:
                    cleaned = extract_json_array(raw)
                    if cleaned:
                        batch = json.loads(cleaned)
                    else:
                        batch = extract_json_objects(raw)
                if not batch:
                    raise RuntimeError(f"Empty batch returned for {stage}")
            data.extend(batch)
            total = len(data)
        data = data[:n]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[FEW_SHOTS] Wrote {out_path}")


if __name__ == "__main__":
    main()
