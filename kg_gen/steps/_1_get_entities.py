import litellm
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional


class EntitiesResponse(BaseModel):
    """
    Schema for entity extraction responses.

    This model is used to validate JSON output produced by the LLM and to
    provide a typed interface for downstream code.
    """

    entities: List[str]


def _load_entities_prompt() -> str:
    """
    Load the entities prompt template from disk.

    Returns:
        str: Prompt template text loaded from `kg_gen/prompts/entities.txt`.
    """
    # Resolve the prompt path relative to this module.
    prompt_path = Path(__file__).parent.parent / "prompts" / "entities.txt"
    # Read the full prompt text as UTF-8.
    return prompt_path.read_text()


def _get_entities_litellm(
    input_data: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    usage_history: Optional[list[dict]] = None,
) -> List[str]:
    """
    Extract entities using LiteLLM with a JSON schema response.

    Args:
        - input_data: Raw text to analyze for entity extraction.
        - model: LLM model identifier to use for extraction.
        - api_key: Optional API key override for the provider.
        - api_base: Optional API base URL override for the provider.
        - temperature: Sampling temperature for the LLM call.
        - usage_history: Optional list to accumulate LiteLLM usage dicts.

    Returns:
        List[str]: List of extracted entity strings.
    """
    # Load the system prompt template defining extraction rules.
    prompt_template = _load_entities_prompt()
    # Build a user prompt containing the text to analyze.
    user_prompt = f"""
Here is the text to extract entities from:

<reasoning>
{input_data}
</reasoning>
    """

    # Build schema with additionalProperties: false (required by OpenAI).
    schema = EntitiesResponse.model_json_schema()
    schema["additionalProperties"] = False

    # Assemble LiteLLM request payload with strict JSON schema output.
    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "entities_response",
                "schema": schema,
                "strict": True,
            }
        },
    }

    # Inject provider overrides only when explicitly supplied.
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    # Execute the LLM call.
    response = litellm.responses(**kwargs)
    if usage_history is not None:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage:
            usage_history.append(usage)
    # Parse the structured JSON response into the Pydantic model.
    parsed = EntitiesResponse.model_validate_json(response.output[-1].content[0].text)
    # Return the extracted entity list.
    return parsed.entities


def get_entities(
    input_data: str,
    is_conversation: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    usage_history: Optional[list[dict]] = None,
) -> List[str]:
    """
    Public API to extract entities from input text.

    Args:
        - input_data: Raw text to analyze.
        - is_conversation: Whether the input represents a dialogue. Currently
            unused but reserved for future routing.
        - model: LLM model identifier to use.
        - api_key: Optional API key override.
        - api_base: Optional API base URL override.
        - temperature: Sampling temperature for the LLM call.
        - usage_history: Optional list to accumulate LiteLLM usage dicts.

    Returns:
        List[str]: Extracted entity strings.
    """
    # Delegate to the LiteLLM-based implementation.
    return _get_entities_litellm(
        input_data,
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        usage_history=usage_history,
    )
