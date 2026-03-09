from typing import List, Dict, Any
import re
from ..core.extractor import TaskExtractor


class DBBenchExtractor(TaskExtractor):
    """
    Extractor specific for DBBench tasks.
    Implements the TaskExtractor interface.
    """

    def extract_user_description(self, history: List[Dict[str, Any]]) -> str:
        """
        In DBBench, the task description is always the third turn:
            1) user -> system prompt
            2) agent -> "Ok."
            3) user -> actual task description
        """
        for i in range(len(history) - 1):
            if (
                history[i]["role"] == "agent"
                and history[i].get("content", "").strip().lower() == "ok."
                and history[i + 1]["role"] == "user"
            ):
                return history[i + 1].get("content", "").strip()
        return ""

    def extract_reasoning_steps(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts reasoning steps from DBBench task history.
        Tool calls = SQL Operation, Termination = Action: Answer
        """
        steps = []
        step_index = 1
        TASK_START_INDEX = 2  # DBBench task starts at third turn

        for i, turn in enumerate(history):

            if i < TASK_START_INDEX:
                continue

            if turn["role"] != "agent" or not turn.get("reasoning_content"):
                continue

            step_data = {
                "step_id": step_index,
                "reasoning": turn["reasoning_content"].strip(),
                "tool_call": None,
                "tool_input": None,
                "tool_output": None,
                "termination": False,
            }

            content = turn.get("content", "")

            # SQL tool call
            sql_match = re.search(
                r"Action:\s*Operation.*?```sql\s*([\s\S]*?)\s*```",
                content,
                re.DOTALL,
            )

            if sql_match:
                step_data["tool_call"] = "SQL"
                step_data["tool_input"] = sql_match.group(1).strip()

                # DB execution result = next user turn
                if i + 1 < len(history):
                    next_turn = history[i + 1]
                    if next_turn["role"] == "user":
                        step_data["tool_output"] = next_turn.get("content", "").strip()

            # Termination detection
            if re.search(r"Action:\s*Answer", content):
                step_data["termination"] = True

            steps.append(step_data)
            step_index += 1

        return steps