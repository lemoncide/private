import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm.client import LLMClient


def main() -> int:
    if not os.getenv("GROQ_API_KEY"):
        print("Missing env var: GROQ_API_KEY")
        return 2

    llm = LLMClient()
    response = llm.generate("what's your model version?")
    if response.startswith("Error generating response:"):
        print(response)
        return 1


    preview = response.replace("\r", " ").replace("\n", " ").strip()[:]
    print(f"OK (preview): {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
