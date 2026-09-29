# Copyright (c) Microsoft. All rights reserved.
# Adapted for SQL Agent RL: portable package imports/entrypoints; see THIRD_PARTY_NOTICES.md.

"""Sample code that demonstrates an SQL agent using LangGraph and LangChain,
trainable with Agent-lightning.

Adapted from https://python.langchain.com/docs/tutorials/sql_qa/
as well as https://langchain-ai.github.io/langgraph/tutorials/sql-agent/
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

import pandas as pd
import termcolor
from .feedback import render_feedback
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from .schema_database import SQLiteSchemaDatabase
from .sql_safety import execute_observation, read_only_connection, resolve_database
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

import agentlightning as agl

agl.setup_logging(apply_to=[__name__])

logger = logging.getLogger(__name__)


WRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an agent designed to interact with a SQL database.
     Given an input question, create a syntactically correct {dialect} query to run to help find the answer.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
GENERATED QUERY
```
""".strip(),
        ),
        ("user", "Question: {input}"),
    ]
)


CHECK_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- Explicit query execution failures
- Clearly unreasoable query execution results

## Table Schema ##

{table_info}

## Output Format ##

If any mistakes from the list above are found, list each error clearly.
After listing mistakes (if any), conclude with **ONE** of the following exact phrases in all caps and without surrounding quotes:
- If mistakes are found: `THE QUERY IS INCORRECT.`
- If no mistakes are found: `THE QUERY IS CORRECT.`

DO NOT write the corrected query in the response. You only need to report the mistakes.
""".strip(),
        ),
        (
            "user",
            """Question: {input}

Query:

```{dialect}
{query}
```

Execution result:

```
{execution}
```""",
        ),
    ]
)


REWRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an agent designed to interact with a SQL database.
Rewrite the previous {dialect} query to fix errors based on the provided feedback.
The goal is to answer the original question.
Make sure to address all points in the feedback.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
REWRITTEN QUERY
```
""".strip(),
        ),
        (
            "user",
            """Question: {input}

## Previous query ##

```{dialect}
{query}
```

## Previous execution result ##

```
{execution}
```

## Feedback ##

{feedback}

Please rewrite the query to address the feedback.""",
        ),
    ]
)


class State(MessagesState):
    question: str
    query: str
    execution: str
    answer: str
    feedback: str
    num_turns: int
    messages: list[AnyMessage]


class SQLAgent:

    def __init__(
        self,
        db: str,
        max_turns: int = 5,
        debug: bool = False,
        db_schema: str | None = None,
        endpoint: str | None = None,
        verl_replacement: Dict[str, Any] | None = None,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
        feedback_mode: str = "raw",
    ):
        if not db.startswith("sqlite:///"):
            raise ValueError("Only manifest-backed SQLite databases are supported")
        self.database_path = db[len("sqlite:///") :]
        engine = create_engine(
            "sqlite://", creator=lambda: read_only_connection(self.database_path), poolclass=NullPool
        )
        self.db = SQLiteSchemaDatabase(engine)
        self.feedback_mode = feedback_mode
        self.candidates: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.stop_reason = "unfinished"
        self.budget_events: list[dict[str, Any]] = []
        self.schema_truncated = False
        self.visible_schema: str | None = None
        self.db_schema = db_schema
        self.debug = debug
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate
        if verl_replacement is not None:
            self.model_name: str = verl_replacement["model"]  # type: ignore
            assert endpoint is not None
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                temperature=verl_replacement["temperature"],
                top_p=verl_replacement.get("top_p", 1.0),
                extra_body={"repetition_penalty": 1.0, "top_k": -1},
                max_retries=0,
                timeout=120,
                max_tokens=2048,
            )
        else:
            self.model_name: str = os.environ.get("MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint or os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"],
                temperature=0,
                top_p=1.0,
                extra_body={"repetition_penalty": 1.0, "top_k": -1},
                max_retries=1,
                timeout=120,
                max_tokens=2048,
            )

    def get_table_info(self) -> str:
        """Get the table information in a human-readable format."""
        if self.visible_schema is not None:
            return self.visible_schema
        try:
            table_info = self.db.get_table_info()
            if len(table_info) > self.table_info_truncate:
                self.schema_truncated = True
                table_info = table_info[: self.table_info_truncate] + "\n... (truncated)"
            self.visible_schema = table_info
            return table_info
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            raise RuntimeError("No usable schema available") from e

    def prompt_fits(self, prompt: Any, stage: str) -> bool:
        tokenizer_path = os.environ.get("SQL_AGENT_TOKENIZER_PATH")
        if not tokenizer_path:
            return True
        messages = [
            {"role": {"human": "user", "ai": "assistant"}.get(m.type, m.type), "content": m.content}
            for m in prompt.messages
        ]
        length = len(
            _tokenizer(tokenizer_path).apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        )
        if length > 4096:
            self.budget_events.append({"stage": stage, "prompt_tokens": length, "limit": 4096})
            return False
        return True

    def invoke_prompt(self, prompt: Any) -> AnyMessage:
        if self.debug:
            for message in prompt.messages:
                termcolor.cprint(message.pretty_repr(), "blue")

        if not self.prompt_fits(prompt, "request"):
            raise ValueError("Initial prompt budget exceeded; check common schema/question protocol")

        started = time.monotonic()
        for attempt in range(3):
            try:
                result = self.llm.invoke(prompt)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (2**attempt))
        self.llm_calls.append(
            {
                "messages": [{"role": m.type, "content": m.content} for m in prompt.messages],
                "response": result.content,
                "usage": getattr(result, "usage_metadata", None),
                "finish_reason": getattr(result, "response_metadata", {}).get("finish_reason"),
                "latency_seconds": time.monotonic() - started,
                "request_retries": attempt,
            }
        )

        if self.debug:
            termcolor.cprint(result.pretty_repr(), "green")

        return result  # type: ignore

    def truncate_execution(self, execution: str) -> str:
        """Truncate the execution result to a reasonable length."""
        if len(execution) > self.execution_truncate:
            return execution[: self.execution_truncate] + "\n... (truncated)"
        return execution

    def parse_query(self, message: AnyMessage) -> str | None:
        result: str | None = None
        for match in re.finditer(r".*```\w*\n(.*?)\n```.*", message.content, re.DOTALL):  # type: ignore
            result = match.group(1).strip()  # type: ignore
        return result  # type: ignore

    def write_query(self, state: State) -> State:
        """Generate SQL query to fetch information."""
        prompt: Any = WRITE_QUERY_PROMPT.invoke(  # type: ignore
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)  # type: ignore

        query = self.parse_query(result) or result.content  # type: ignore

        return {  # type: ignore
            **state,
            "query": query,  # type: ignore
            "num_turns": 1,
            "messages": [*prompt.messages, result],
        }

    def execute_query(self, state: State) -> State:
        """Execute SQL query."""
        started = time.monotonic()
        observation = execute_observation(self.database_path, state["query"], preview_chars=self.execution_truncate)
        execution_result = render_feedback(observation, self.feedback_mode)
        self.candidates.append(
            {
                "sql": state["query"],
                "observation": asdict(observation),
                "feedback_text": execution_result,
                "checker": None,
                "execution_seconds": time.monotonic() - started,
            }
        )
        if self.debug:
            termcolor.cprint(execution_result, "yellow")
        return {**state, "execution": execution_result}

    def check_query(self, state: State) -> State:
        """Check the SQL query for correctness."""
        prompt: Any = CHECK_QUERY_PROMPT.invoke(  # type: ignore
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "query": state["query"],
                "execution": state["execution"],
                "table_info": self.get_table_info(),
            }
        )
        if not self.prompt_fits(prompt, "check_query"):
            self.stop_reason = "prompt_budget_exhausted"
            return {**state, "feedback": ""}
        result = self.invoke_prompt(prompt)  # type: ignore

        res = {  # type: ignore
            **state,
            "feedback": result.content,  # type: ignore
            "messages": [*state.get("messages", []), *prompt.messages, result],
        }
        self.candidates[-1]["checker"] = result.content
        return res  # type: ignore

    def rewrite_prompt(self, state: State) -> Any:
        return REWRITE_QUERY_PROMPT.invoke(  # type: ignore
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "query": state["query"],
                "execution": state["execution"],
                "feedback": state["feedback"],
                "table_info": self.get_table_info(),
            }
        )

    def rewrite_query(self, state: State) -> State:
        """Rewrite SQL query if necessary."""
        prompt = self.rewrite_prompt(state)
        result = self.invoke_prompt(prompt)  # type: ignore

        rewritten_query = self.parse_query(result)  # type: ignore

        return {
            **state,
            "query": rewritten_query or state["query"],
            "num_turns": state.get("num_turns", 0) + 1,
            "messages": [*prompt.messages, result],  # clear previous prompts
        }

    def should_continue(self, state: State) -> Literal[END, "rewrite_query"]:  # type: ignore
        """Determine if the agent should continue based on the result."""
        if self.stop_reason == "prompt_budget_exhausted":
            return END
        if state["messages"] and isinstance(state["messages"][-1], BaseMessage):  # type: ignore
            last_message = state["messages"][-1]
            if "THE QUERY IS CORRECT" in last_message.content:  # type: ignore
                if "THE QUERY IS INCORRECT" in last_message.content:  # type: ignore
                    # Both correct and incorrect messages found
                    # See which is the last one
                    correct_index = last_message.content.rfind("THE QUERY IS CORRECT")  # type: ignore
                    incorrect_index = last_message.content.rfind("THE QUERY IS INCORRECT")  # type: ignore
                    if correct_index > incorrect_index:
                        self.stop_reason = "checker_accept"
                        return END
                else:
                    self.stop_reason = "checker_accept"
                    return END

        if state.get("num_turns", 0) >= self.max_turns:
            self.stop_reason = "sql_attempt_limit"
            return END

        if not self.prompt_fits(self.rewrite_prompt(state), "rewrite_query"):
            self.stop_reason = "prompt_budget_exhausted"
            return END

        return "rewrite_query"

    def graph(self, start_from_query: bool = False) -> CompiledStateGraph[State]:
        builder = StateGraph(State)
        builder.add_node(self.write_query)  # type: ignore
        builder.add_node(self.execute_query)  # type: ignore
        builder.add_node(self.check_query)  # type: ignore
        builder.add_node(self.rewrite_query)  # type: ignore

        builder.add_edge(START, "execute_query" if start_from_query else "write_query")
        builder.add_edge("write_query", "execute_query")
        builder.add_edge("execute_query", "check_query")
        builder.add_conditional_edges(
            "check_query",
            self.should_continue,  # type: ignore
        )
        builder.add_edge("rewrite_query", "execute_query")

        return builder.compile()  # type: ignore


def evaluate_query(query: str, ground_truth: str, database: str, raise_on_error: bool = True) -> float:
    from .evaluation_guard import evaluate_query as guarded_evaluate

    return guarded_evaluate(query, ground_truth, database, raise_on_error=raise_on_error)


@lru_cache(maxsize=2)
def _tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


@lru_cache(maxsize=2)
def _gold_index(path: str) -> dict[str, str]:
    """Evaluator-only lookup. Never pass this mapping into the graph or resources."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_record(directory: str, name: str, value: dict[str, Any]) -> None:
    """Write one independently auditable artifact per rollout attempt."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class LitSQLAgent(agl.LitAgent[Dict[str, Any]]):

    def __init__(
        self,
        trained_agents: Optional[str] = r"write",
        val_temperature: Optional[float] = None,
        max_turns: int = 3,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
        feedback_mode: str = "raw",
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.spider_dir = os.environ.get("VERL_SPIDER_DATA_DIR", "data")
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate
        self.feedback_mode = feedback_mode

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        question = task["question"]
        start_time = time.time()
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])

        if "query" in task or "gold" in task:
            raise ValueError("Gold SQL must not be present in the agent task payload")
        original_db_path = str(resolve_database(self.spider_dir, task))

        schema_path = os.path.join(os.path.dirname(original_db_path), "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                schema = f.read()
        else:
            logger.error("Schema file not found: %s", schema_path)
            schema = "No schema available."

        rollout_id = rollout.rollout_id
        artifact_name = f"{rollout_id}_{rollout.attempt.attempt_id}"
        artifact_dir = os.environ["SQL_AGENT_ARTIFACT_DIR"]

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
            shutil.copyfile(original_db_path, db_path)
            logger.info(f"[Rollout {rollout_id}] Question: {question}")

            # Run the agent
            sql_agent = SQLAgent(
                "sqlite:///" + db_path,
                max_turns=self.max_turns,
                table_info_truncate=self.table_info_truncate,
                execution_truncate=self.execution_truncate,
                feedback_mode=self.feedback_mode,
                debug=False,
                db_schema=schema,
                endpoint=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),  # type: ignore
                verl_replacement=(
                    {"model": llm.model, **llm.sampling_parameters}
                    if rollout.mode == "train"
                    else {
                        "model": llm.model,
                        "temperature": (
                            self.val_temperature
                            if self.val_temperature is not None
                            else llm.sampling_parameters.get("temperature", 0.0)
                        ),
                    }
                ),
            )
            agent = sql_agent.graph()
            try:
                # Required to make the langchain tracing work
                handler = self.tracer.get_langchain_handler()
                result = agent.invoke(  # type: ignore
                    {"question": question},  # type: ignore
                    {"callbacks": [handler] if handler else [], "recursion_limit": 100},
                )
            except Exception as e:
                save_record(
                    artifact_dir + "/model_view",
                    artifact_name,
                    {
                        "source_id": task["source_id"],
                        "db_id": task["db_id"],
                        "feedback_mode": self.feedback_mode,
                        "candidates": sql_agent.candidates,
                        "llm_calls": sql_agent.llm_calls,
                        "stop_reason": "infrastructure_failure",
                        "error_type": type(e).__name__,
                        "elapsed_seconds": time.time() - start_time,
                    },
                )
                raise

            save_record(
                artifact_dir + "/model_view",
                artifact_name,
                {
                    "source_id": task["source_id"],
                    "db_id": task["db_id"],
                    "split": task["split"],
                    "feedback_mode": self.feedback_mode,
                    "candidates": sql_agent.candidates,
                    "llm_calls": sql_agent.llm_calls,
                    "stop_reason": sql_agent.stop_reason,
                    "schema_truncated": sql_agent.schema_truncated,
                    "visible_schema": sql_agent.get_table_info(),
                    "sql_attempts": len(sql_agent.candidates),
                    "elapsed_seconds": time.time() - start_time,
                    "budget_events": sql_agent.budget_events,
                    "schema_fallback_reason": sql_agent.db.schema_fallback_reason,
                },
            )

            logger.info(f"[Rollout {rollout_id}] Generated Query: {result['query']}")

        end_time_rollout = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
            shutil.copyfile(original_db_path, db_path)

            ground_truth = _gold_index(os.environ["SQL_AGENT_GOLD_PATH"])[task["source_id"]]
            reward = evaluate_query(result["query"], ground_truth, db_path)
            save_record(
                artifact_dir + "/evaluator",
                artifact_name,
                {
                    "source_id": task["source_id"],
                    "final_reward": reward,
                    "rollout_id": rollout_id,
                    "attempt_id": rollout.attempt.attempt_id,
                },
            )
            logger.info("[Rollout %s] Reward: %s", rollout_id, reward)

        end_time_eval = time.time()

        logger.info("[Rollout %s] Time taken for rollout: %.2f seconds", rollout_id, end_time_rollout - start_time)
        logger.info(
            "[Rollout %s] Time taken for evaluation: %.2f seconds", rollout_id, end_time_eval - end_time_rollout
        )

        return reward


