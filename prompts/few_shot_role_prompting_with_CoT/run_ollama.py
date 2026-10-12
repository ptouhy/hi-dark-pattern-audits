import argparse
import asyncio
import json
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from browser_use import Agent, BrowserProfile, BrowserSession, Controller
from browser_use.llm import ChatOllama

from schema import DarkPatternRunResult


# --------- CONFIG ---------
# Same eval set, prompt, taxonomy, and schema as the gpt-5 baseline in run_openai.py.
BROKER_CSV = "broker_dataset/hi_vetted_11_brokers.csv"
DP_PROMPT_USER_PATH = "prompts/few_shot_role_prompting_with_CoT/prompt_regulator_role_hi_only.txt"
DP_TAXONOMY_JSON = "prompts/few_shot_role_prompting_with_CoT/hidden_information_only.json"
RUN_TYPE = "hidden_information_hi_only_few_shot_role_prompting_with_CoT"
ROOT_OUT_DIR = "agent_output"

MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST_URL", "http://localhost:11434")
# Ollama's default context is far too small for the task prompt + DOM + screenshots + history.
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "65536"))

# Chen's settings: shorter waits cause page-load failures on slow sites, and keeping them
# identical to the gpt-5 baseline keeps the comparison fair.
PAGE_WAIT_SECONDS = float(os.getenv("PAGE_WAIT_SECONDS", "60"))
ACTION_WAIT_SECONDS = float(os.getenv("ACTION_WAIT_SECONDS", "5"))

USE_EXTEND_SYSTEM_MESSAGE = False
EXTEND_SYSTEM_MESSAGE_PATH = "prompts/few_shot_role_prompting_with_CoT/extend_system_message.txt"

# Leave unset to let Browser Use find Chrome/Chromium on its own.
CHROME_PATH = os.getenv("CHROME_PATH") or None
# --------------------------


def extract_domain(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "")


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def write_text(path, content):
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(content, (list, tuple)):
            f.write("\n".join(str(x) for x in content))
        else:
            f.write(str(content))


def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_dark_pattern_task(template_user_path: str, taxonomy_json_path: str, url: str) -> str:
    template_user_text = Path(template_user_path).read_text(encoding="utf-8")
    taxonomy = json.loads(Path(taxonomy_json_path).read_text(encoding="utf-8"))
    taxonomy_block = json.dumps(taxonomy, ensure_ascii=False, indent=2)
    return template_user_text.replace("$dark_pattern_taxonomy", taxonomy_block).replace("$url", url)


def model_dir_name(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


async def process_one(url: str, browser_session: BrowserSession):
    url_domain = extract_domain(url)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(os.path.join(ROOT_OUT_DIR, url_domain, RUN_TYPE, model_dir_name(MODEL), stamp))
    shots_dir = ensure_dir(os.path.join(run_dir, "screenshots"))

    user_prompt = load_dark_pattern_task(DP_PROMPT_USER_PATH, DP_TAXONOMY_JSON, url)
    controller = Controller(output_model=DarkPatternRunResult)

    llm = ChatOllama(
        model=MODEL,
        host=OLLAMA_HOST,
        timeout=600,
        ollama_options={"temperature": 0, "num_ctx": NUM_CTX},
    )

    agent_kwargs = {
        "controller": controller,
        "task": user_prompt,
        "llm": llm,
        "use_vision": True,
        "llm_timeout": 600,
        "step_timeout": 600,
        "max_failures": 5,
        "final_response_after_failures": True,
        "browser_session": browser_session,
        "output_model_schema": DarkPatternRunResult,
        "save_conversation_path": os.path.join(run_dir, "conversation_history.json"),
        "save_conversation_path_encoding": "utf-8",
        "minimum_wait_page_load_time": PAGE_WAIT_SECONDS,
        "wait_for_network_idle_page_load_time": PAGE_WAIT_SECONDS,
        "wait_between_actions": ACTION_WAIT_SECONDS,
    }
    if USE_EXTEND_SYSTEM_MESSAGE:
        agent_kwargs["extend_system_message"] = Path(EXTEND_SYSTEM_MESSAGE_PATH).read_text(encoding="utf-8").strip()

    agent = Agent(**agent_kwargs)
    started = datetime.now()
    raw_output = None
    try:
        raw_output = await agent.run()

        result = raw_output.final_result() if raw_output else None
        if isinstance(result, str) and result.strip():
            try:
                parsed = DarkPatternRunResult.model_validate_json(result)
                write_text(os.path.join(run_dir, "final_result.json"), parsed.model_dump_json(indent=2))
            except Exception as e:
                print(f"CRITICAL: JSON decode/validation error: {e}")
                write_text(os.path.join(run_dir, "INVALID_JSON_PAYLOAD.txt"), str(result))
        else:
            write_text(os.path.join(run_dir, "FAILURE.txt"), f"Agent returned no final result: {result!r}")

    except Exception as e:
        print(f"CRITICAL: unexpected error for {url}: {e}")
        write_text(os.path.join(run_dir, "UNEXPECTED_ERROR.txt"), traceback.format_exc())

    finally:
        def _safe(fn):
            try:
                return fn() if callable(fn) else fn
            except Exception:
                return []

        urls = _safe(raw_output.urls) if raw_output else []
        action_names = _safe(raw_output.action_names) if raw_output else []
        extracted = _safe(raw_output.extracted_content) if raw_output else []
        model_actions = _safe(raw_output.model_actions) if raw_output else []
        errors = [str(e) for e in (_safe(raw_output.errors) if raw_output else [])]
        shots = _safe(raw_output.screenshot_paths) if raw_output else []

        write_text(os.path.join(run_dir, "model_thoughts.json"), _safe(raw_output.model_thoughts) if raw_output else [])
        write_text(os.path.join(run_dir, "action_results.json"), _safe(raw_output.action_results) if raw_output else [])
        write_text(os.path.join(run_dir, "urls.txt"), urls)
        write_text(os.path.join(run_dir, "action_names.txt"), action_names)
        write_text(os.path.join(run_dir, "extracted_content.txt"), extracted)
        write_text(os.path.join(run_dir, "errors.txt"), errors)
        write_json(os.path.join(run_dir, "model_actions.json"), model_actions)

        for p in shots or []:
            if p and os.path.exists(p):
                try:
                    shutil.copy(p, shots_dir)
                except Exception:
                    pass

        elapsed_min = (datetime.now() - started).total_seconds() / 60
        write_text(os.path.join(run_dir, "SUMMARY.txt"), [
            f"URL: {url}",
            f"Model: {MODEL}",
            f"Minutes: {elapsed_min:.1f}",
            f"URLs visited: {len(urls)}",
            f"Actions: {len(action_names)}",
            f"Errors: {len(errors)}",
            f"Page wait (s): {PAGE_WAIT_SECONDS}  Action wait (s): {ACTION_WAIT_SECONDS}",
        ])
        print(f"[{url}] done in {elapsed_min:.1f} min -> {run_dir}")


async def main(urls):
    profile_dir = Path(ROOT_OUT_DIR, "chromedata").resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    browser_profile = BrowserProfile(
        headless=True,
        stealth=True,
        viewport={"width": 1920, "height": 1080},
        locale="en-US",
        user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"),
        wait_for_network_idle_page_load_time=PAGE_WAIT_SECONDS,
    )

    for url in urls:
        browser_session = BrowserSession(
            browser_profile=browser_profile,
            headless=True,
            user_data_dir=str(profile_dir),
            executable_path=CHROME_PATH,
            enable_default_extensions=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-dev-shm-usage",
                "--mute-audio",
                "--lang=en-US,en",
                "--window-size=1920,1080",
            ],
        )
        await browser_session.start()
        try:
            await process_one(url, browser_session)
        except Exception as e:
            err_dir = ensure_dir(os.path.join(ROOT_OUT_DIR, "failures"))
            with open(os.path.join(err_dir, "failures.log"), "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} :: {url} :: {e}\n")
        finally:
            try:
                await browser_session.close()
            except Exception:
                pass

    print("All URLs processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Run only URLs containing this text, e.g. --only stirista")
    args = parser.parse_args()

    urls = pd.read_csv(BROKER_CSV)["url"].dropna().tolist()
    if args.only:
        urls = [u for u in urls if args.only in u]

    print(f"MODEL: {MODEL} @ {OLLAMA_HOST}  num_ctx={NUM_CTX}")
    print(f"Waits: page={PAGE_WAIT_SECONDS}s action={ACTION_WAIT_SECONDS}s")
    print(f"URLs ({len(urls)}): {urls}")
    asyncio.run(main(urls))
