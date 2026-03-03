import json
from pathlib import Path


def load(path: Path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def score(examples):
    if not examples:
        return 0
    users = [e.get("user", "") for e in examples if isinstance(e, dict)]
    users = [str(u) for u in users if u]
    unique_users = len(set(users))
    avg_len = sum(len(u) for u in users) / max(1, len(users))
    return unique_users + avg_len / 100.0


def main():
    repo = Path(__file__).resolve().parents[1]
    groq_dir = repo / "app" / "prompt_engine" / "groq_prompts" / "few_shots"
    sarvam_dir = repo / "app" / "prompt_engine" / "groq_prompts" / "few_shots_sarvam"
    if not sarvam_dir.exists():
        print("No Sarvam few-shots directory found.")
        return

    for groq_file in groq_dir.glob("*.json"):
        sarvam_file = sarvam_dir / groq_file.name
        groq_ex = load(groq_file)
        sarvam_ex = load(sarvam_file)
        if not sarvam_ex:
            continue
        if score(sarvam_ex) >= score(groq_ex):
            groq_file.write_text(sarvam_file.read_text(encoding="utf-8"), encoding="utf-8")
            sarvam_file.unlink(missing_ok=True)
            print(f"Replaced with Sarvam: {groq_file.name}")
        else:
            sarvam_file.unlink(missing_ok=True)
            print(f"Kept Groq: {groq_file.name}")


if __name__ == "__main__":
    main()
