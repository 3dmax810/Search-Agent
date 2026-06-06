from openai.types.responses import container_network_policy_allowlist_param
from openai.types.responses import container_network_policy_allowlist_param
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

import json
import re

from search_agent.agent.llm import OllamaModel


class AnswerJudge:
    def __init__(self, model=None):
        self.model = model or OllamaModel()

    def judge(self, question: str, answer: str, observe_text: str) -> tuple[bool, str]:
        prompt = f"""You are an answer judge.
Decide whether the answer directly answers the user's original question and is supported by the observations.

Return only JSON:
{{"accepted": true or false, "reason": "short reason"}}

Rules:
- Reject if the answer only states an intermediate fact.
- Reject if the question asks for a final attribute but the answer only identifies an intermediate entity.
- Reject if the answer is not supported by the observations.
- Accept only if the answer directly answers the original question.

Question:
{question}

Observations:
{observe_text}

Answer:
{answer}

JSON
"""
        raw = self.model.generate(prompt)

        try:
            data = json.loads(raw)
            return bool(data.get("accepted")), str(data.get("reason", ""))
        except json.JSONDecodeError:
            pass

        return False, f"judge returned invalid JSON: {raw[:120]}"
