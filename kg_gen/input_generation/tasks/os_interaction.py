from typing import List, Dict, Any
from ..core.extractor import TaskExtractor


class OSInteractionExtractor(TaskExtractor):
    """
    Extractor specific for OS interaction tasks.
    Implements the TaskExtractor interface.
    """

    def extract_user_description(self, history: List[Dict[str, Any]]) -> str:
        """
        In OS tasks, the user description is detected by the specific start string:
            "Now, I will start a new problem in a new OS"
        """
        for turn in history:
            if (
                turn["role"] == "user"
                and "Now, I will start a new problem in a new OS" in turn.get("content", "")
            ):
                return turn["content"].strip()
        return ""

    def extract_reasoning_steps(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts reasoning steps for OS interaction tasks.
        Tool calls = Act: bash, Termination = Act: finish/answer
        """
        steps = []
        step_index = 1
        task_started = False

        for i, turn in enumerate(history):

            # detect start of task
            if (
                turn["role"] == "user"
                and "Now, I will start a new problem in a new OS" in turn.get("content", "")
            ):
                task_started = True
                continue

            if not task_started or turn["role"] != "agent" or not turn.get("reasoning_content"):
                continue

            step_data = {
                "step_id": step_index,
                "reasoning": turn["reasoning_content"].strip(),
                "tool_call": None,
                "tool_output": None,
                "termination": False,
            }

            content = turn.get("content", "")

            # Tool call detection
            if "Act: bash" in content:
                step_data["tool_call"] = "OS"

                # tool output = next user turn
                if i + 1 < len(history):
                    next_turn = history[i + 1]
                    if next_turn["role"] == "user":
                        user_content = next_turn.get("content", "")
                        if user_content.startswith("The output of the OS:"):
                            step_data["tool_output"] = user_content.replace(
                                "The output of the OS:", ""
                            ).strip()

            # Termination
            if "Act: finish" in content or "Act: answer" in content:
                step_data["termination"] = True

            steps.append(step_data)
            step_index += 1

        return steps