import json
import yaml
import uuid
from pathlib import Path
from typing import List, Dict, Any
from .extractor import TaskExtractor


# load configuration and extract model name
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


# save failed jsonl
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


# BUILD KGGEN INPUT STRUCTURE
def build_reasonings_structure(sample: Dict[str, Any], model_name: str, extractor: TaskExtractor) -> Dict[int, str]:
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

    task_description = extractor.extract_user_description(history)
    safe_task = task_description.replace('"', '\\"').replace("'", "\\'")

    steps = extractor.extract_reasoning_steps(history)
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
