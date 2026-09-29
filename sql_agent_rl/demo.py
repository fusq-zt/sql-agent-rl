"""Run one question against an existing OpenAI-compatible model endpoint."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--endpoint", default=os.environ.get("OPENAI_API_BASE"))
    p.add_argument("--model", required=True, help="Model name served by the endpoint")
    p.add_argument("--tokenizer", type=Path, required=True, help="Local tokenizer matching the model")
    args = p.parse_args()
    if not args.endpoint:
        p.error("Provide --endpoint or OPENAI_API_BASE")
    os.environ["SQL_AGENT_TOKENIZER_PATH"] = str(args.tokenizer.resolve(strict=True))
    from .agent import SQLAgent
    agent = SQLAgent("sqlite:///" + str(args.database.resolve(strict=True)), max_turns=3,
                     feedback_mode="structured", endpoint=args.endpoint,
                     verl_replacement={"model": args.model, "temperature": 0.0, "top_p": 1.0})
    agent.graph().invoke({"question": args.question}, {"recursion_limit": 100})
    print(json.dumps({"sql": agent.candidates[-1]["sql"], "stop_reason": agent.stop_reason,
                      "candidates": agent.candidates}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
