import json
import yaml
import uuid
from pathlib import Path
from typing import List, Dict, Any


#  LOAD CONFIGURATION AND EXTRACT MODEL NAME
def extract_model_from_config(config_path: Path) -> str:
    """
    Extracts the model name used by the agent from a YAML configuration file.

    Args:
        config_path (Path): Path to the YAML configuration file.

    Returns:
        str: The model name configured for the agent.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    agent_name = config["assignments"][0]["agent"]
    model = config["definition"]["agent"][agent_name]["parameters"]["body"]["model"]
    return model


#  LOAD JSONL AND FILTER FAILED SAMPLES
def load_failed_samples(jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    Loads a JSONL file of agent runs and filters for failed samples.

    A sample is considered failed if:
      - status is "agent validation failed" or "agent invalid action"
      - OR status is "completed" but the result is False

    Args:
        jsonl_path (Path): Path to the JSONL file containing runs.

    Returns:
        List[Dict[str, Any]]: List of dictionaries representing failed runs.
    """
    failed = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            status = sample.get("output", {}).get("status")
            result = sample.get("output", {}).get("result", {}).get("result")
            if (status in {"agent validation failed", "agent invalid action"} or (status == "completed" and result is False)):
               failed.append(sample)

    return failed


#  SAVE FAILED JSONL
def save_failed_jsonl(failed_samples: List[Dict[str, Any]], output_path: Path):
    """
    Saves failed samples into a JSONL file, one sample per line.

    Args:
        failed_samples (List[Dict[str, Any]]): List of failed sample dictionaries.
        output_path (Path): File path where the failed JSONL will be saved.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in failed_samples:
            f.write(json.dumps(sample) + "\n")


#  EXTRACT USER DESCRIPTION
def extract_user_description(history: List[Dict[str, Any]]) -> str:
    """
    Extracts the user-provided problem description from the conversation history.

    Args:
        history (List[Dict[str, Any]]): List of conversation turns.

    Returns:
        str: User description content, or empty string if not found.
    """
    for turn in history:
        if (
            turn["role"] == "user"
            and "Now, I will start a new problem in a new OS" in turn.get("content", "")
        ):
            return turn["content"].strip()
    return ""


#  EXTRACT REASONING STEPS
def extract_reasoning_steps(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts reasoning steps performed by the agent from conversation history.

    Each step may include:
      - reasoning content
      - tool calls (e.g., OS commands)
      - tool outputs
      - termination flag

    Args:
        history (List[Dict[str, Any]]): Conversation history with user and agent turns.

    Returns:
        List[Dict[str, Any]]: List of reasoning step dictionaries.
    """
    steps = []
    step_index = 1
    task_started = False

    for i, turn in enumerate(history):

        if (
            turn["role"] == "user"
            and "Now, I will start a new problem in a new OS" in turn.get("content", "")
        ):
            task_started = True
            continue

        if not task_started:
            continue

        if turn["role"] == "agent" and turn.get("reasoning_content"):

            step_data = {
                "step_id": step_index,
                "reasoning": turn["reasoning_content"].strip(),
                "tool_call": None,
                "tool_output": None,
                "termination": False
            }

            content = turn.get("content", "")

            # TOOL CALL
            if "Act: bash" in content:
                step_data["tool_call"] = "OS"

                if i + 1 < len(history):
                    next_turn = history[i + 1]
                    if next_turn["role"] == "user":
                        user_content = next_turn.get("content", "")
                        if user_content.startswith("The output of the OS:"):
                            step_data["tool_output"] = user_content.replace(
                                "The output of the OS:", ""
                            ).strip()

            # TERMINATION
            if "Act: finish" in content:
                step_data["termination"] = True

            steps.append(step_data)
            step_index += 1

    return steps


# BUILD KGGEN INPUT STRUCTURE
def build_reasonings_structure(sample: Dict[str, Any], model_name: str) -> Dict[int, str]:
    """
    Converts a failed sample into KGGen input format.

    Args:
        sample (Dict[str, Any]): Failed sample dictionary.
        model_name (str): Name of the model used in the run.

    Returns:
        Dict[int, str]: Mapping from step_id to reasoning block string for KGGen.
    """
    history = sample["output"]["history"]
    sample_index = sample["index"]

    agent_id = str(uuid.uuid4().int)[:4]
    user_id = str(uuid.uuid4().int)[:4]
    run_id = str(uuid.uuid4().int)[:4]

    task_description = extract_user_description(history)
    safe_task = task_description.replace('"', '\\"').replace("'", "\\'")

    steps = extract_reasoning_steps(history)
    result = {}

    for step in steps:

        step_id = step["step_id"]
        block = []

        termination_flag_value = "true" if step["termination"] else "false"

        # HEADER (only for first step)
        if step_id == 1:

            block.append(f'@agent(id="{agent_id}", model="{model_name}")')

            block.append("@state(")
            block.append(f'    id="{sample_index}",')
            block.append(f'    current_step_id="{step_id}",')
            block.append('    calculator_ready="true",')
            block.append('    search_ready="true",')
            block.append('    database_connected="true",')
            block.append(f'    last_user_input="{safe_task}",')
            block.append(f'    current_step_index="{step_id}",')
            block.append('    intentions_pending="[]",')
            block.append('    last_action_intent="None",')
            block.append('    last_tool_call="None",')
            block.append('    last_tool_output="None",')
            block.append(f'    termination_flag="{termination_flag_value}"')
            block.append(")")

            block.append(f'@user(id="{user_id}")')
            block.append(f'@task(id="T{sample_index}", description="{safe_task}")')
            block.append(f'@run(id="{run_id}")')

        # STEP
        block.append(f'@step(id="{step_id}", index="{step_id}")')

        safe_reasoning = step["reasoning"].replace('"', '\\"').replace("'", "\\'")
        block.append(f'@start_reasoning(step_id="{step_id}")')
        block.append(f'"{safe_reasoning}"')
        block.append(f'@end_reasoning(step_id="{step_id}")')

        # TOOL CALL
        if step["tool_call"]:
            block.append(f'@call(tool="{step["tool_call"]}")')

            safe_output = str(step["tool_output"]).replace('"', '\\"').replace("'", "\\'")
            block.append("@observation(")
            block.append(f'    tool="{step["tool_call"]}",')
            block.append(f'    output="{safe_output}"')
            block.append(")")

        # TERMINATION
        if step["termination"]:
            block.append("@termination")

        result[step_id] = "\n".join(block)

    return result


#  MAIN FUNCTION: RETURN LIST OF KG INPUT DICTIONARIES
def get_failed_samples_inputs(jsonl_path: Path, config_path: Path) -> List[Dict[int, str]]:
    """
    Generates KGGen-ready inputs for all failed samples in a JSONL file.

    Steps:
      1. Extract model name from config
      2. Load failed samples from JSONL
      3. Save failed samples to a local JSONL file for record
      4. Convert each failed sample into KGGen input structure

    Args:
        jsonl_path (Path): Path to the JSONL file containing runs.
        config_path (Path): Path to the YAML config file.

    Returns:
        List[Dict[int, str]]: List of dictionaries, each mapping step_id -> reasoning block string.
    """
    print("Loading model from config...")
    model_name = extract_model_from_config(config_path)
    print(f"Model name: {model_name}")

    print("Loading failed samples...")
    failed_samples = load_failed_samples(jsonl_path)
    print(f"Found {len(failed_samples)} failed samples.")
    
    base_dir = Path(__file__).parent
    failed_dir = base_dir / "failed_tasks_results"
    failed_dir.mkdir(exist_ok=True)
    failed_jsonl_path = failed_dir / "failed_runs.jsonl"
    save_failed_jsonl(failed_samples, failed_jsonl_path)
    print(f"Saved failed runs to {failed_jsonl_path}")

    kg_inputs_list = [build_reasonings_structure(sample, model_name) for sample in failed_samples]
    return kg_inputs_list

