import litellm
from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import BaseModel


class EntityItem(BaseModel):
    """
    Pydantic model representing a single entity extracted from text.
    
    Attributes:
        name (str): The name/identifier of the entity.
        reasoning (str): The snippet of text from which the entity was extracted.
    """
    name: str
    reasoning: str


class EntitiesResponse(BaseModel):
    """
    Schema for the entities extraction response.
    
    Attributes:
        entities (List[EntityItem]): List of entities extracted from the text,
                                     each including its reasoning snippet.
    """
    entities: List[EntityItem]


def _load_entities_prompt() -> str:
    """
    Load the prompt template used for entity extraction.
    
    Returns:
        str: The content of the entities prompt file.
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "entities.txt"
    return prompt_path.read_text(encoding="utf-8")


def _get_entities_litellm(
    input_data: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    usage_history: Optional[list[dict]] = None,
) -> List[Tuple[str, str]]:
    """
    Extract entities from the input text using LiteLLM with strict JSON schema validation.
    
    Args:
        input_data (str): Text from which to extract entities.
        model (str): LiteLLM model name to use for extraction.
        api_key (Optional[str]): API key for LiteLLM/OpenAI.
        api_base (Optional[str]): API base URL if using a custom endpoint.
        temperature (float): Sampling temperature for the model (default 0.0).
        usage_history (Optional[list]): Optional list to record API usage statistics.
        
    Returns:
        List[Tuple[str, str]]: List of tuples containing (entity_name, reasoning_snippet)
    """
    # Load the system prompt template for entity extraction
    prompt_template = _load_entities_prompt()
    
    # Build the user-facing prompt with the input text
    user_prompt = f"""
    Here is the text to extract entities from:

    <reasoning>
    {input_data}
    </reasoning>

    Rules:
    - Include for each entity the snippet of text <reasoning> from which it is extracted.
    - Output JSON like:
      {{
        "entities": [
            {{"name": "trace:Step_1", "reasoning": "text snippet"}}
        ]
      }}
    - Only instantiate entities from the ontology.
    - Output JSON only, no explanations.
    """

    # Define strict JSON schema for LiteLLM/OpenAI output validation
    schema = {
        "title": "entities_response",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["name", "reasoning"]
                }
            }
        },
        "required": ["entities"]
    }
    
    # Construct the parameters for LiteLLM API call
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
    
    # Optionally set API key and base
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base
    
    # Call LiteLLM to get the response
    response = litellm.responses(**kwargs)
    
    # Optionally store usage information
    if usage_history is not None:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage:
            usage_history.append(usage)

    # Parse the JSON output from LiteLLM using Pydantic
    parsed = EntitiesResponse.model_validate_json(response.output[-1].content[0].text)

    # Convert to list of tuples (name, reasoning)
    return [(e.name, e.reasoning) for e in parsed.entities]


def get_entities(
    input_data: str,
    is_conversation: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    usage_history: Optional[list[dict]] = None,
) -> List[Tuple[str, str]]:
    """
    Public API to extract entities with reasoning from input text.
    
    Args:
        input_data (str): Text from which to extract entities.
        is_conversation (bool): Flag indicating if the input is a conversation (currently unused).
        model (Optional[str]): LiteLLM model to use.
        api_key (Optional[str]): API key for LiteLLM/OpenAI.
        api_base (Optional[str]): API base URL if using a custom endpoint.
        temperature (float): Sampling temperature for the model (default 0.0).
        usage_history (Optional[list]): Optional list to record API usage statistics.
        
    Returns:
        List[Tuple[str, str]]: List of (entity_name, reasoning_snippet) extracted from the text.
    """
    return _get_entities_litellm(
        input_data,
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        usage_history=usage_history,
    )
