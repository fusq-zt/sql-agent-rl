# Third-party notices

## Agent Lightning

- Source: https://github.com/microsoft/agent-lightning
- Version: 0.3.1; commit `6db3c73dc0af2e50fe80e56873a57c1703f1b2a7`
- Copyright: Microsoft and contributors; MIT license, reproduced in `licenses/AGENT_LIGHTNING_MIT.txt`.

The Agent graph, prompt templates, model-call integration and training configuration are adapted from the upstream Spider example. `sql_agent_rl/agent.py` preserves the original attribution to the LangChain SQL QA and LangGraph SQL Agent tutorials. Local adaptations include task/data integration, bounded read-only SQL observations, selected training nodes, runtime instrumentation and portable entrypoints. These are not claims of inventing Agent Lightning, LangGraph or GRPO.

`runtime/agent-lightning.patch` applies only to the specified upstream commit. It retains the original file headers and the changes required by the selected runtime. The complete framework is fetched on demand and is not vendored in this repository.

## SQL execution comparison

- Original source: https://github.com/taoyds/test-suite-sql-eval
- Original license: Apache License 2.0, reproduced in `licenses/SQL_EVALUATOR_APACHE_2_0.txt`.
- Copied here through the pinned Agent Lightning Spider example, which retains Microsoft headers and the original repository attribution.

Files in `sql_agent_rl/spider_eval/` retain the upstream code and notices. `evaluation_guard.py` supplies read-only, deadline-limited database connections while reusing the comparison implementation. This project does not claim to implement the official evaluator from scratch, nor to run the full test-suite benchmark across multiple database instances.

## Model and data

- Model: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
- Spider 1.0: https://yale-lily.github.io/spider ; dataset license CC BY-SA 4.0 as stated by the dataset authors.

Model weights, Spider databases and annotations are not distributed in this repository. Obtain them from the linked sources under their respective terms. `examples/demo.sql` is synthetic teaching data created for this project.

Original project additions use the repository's MIT license; third-party portions retain their respective licenses and notices.
