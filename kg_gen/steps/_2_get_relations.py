import json
import litellm
from pathlib import Path
from typing import List, Tuple, Optional, Literal, Type
from pydantic import BaseModel, create_model, ValidationError


def parse_relations_response(
    raw_json: str,
    entities: List[str],
    response_model: Optional[Type[BaseModel]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Parse a relations JSON response with graceful fallback.

    The function first tries strict Pydantic validation. If it fails (e.g.,
    because entity literals do not match), it falls back to raw JSON parsing
    and filters out invalid items.

    Args:
        raw_json (str): Raw JSON string from the LLM response.
        entities (List[str]): List of valid entity strings used for validation.
        response_model (Optional[Type[BaseModel]]): Pydantic model for strict validation.

    Returns:
        List[Tuple[str, str, str]]: List of (subject, predicate, object)
        tuples with valid entities only.
    """

    # Strict mode requires a response model
    if response_model is None:
        raise ValueError("Strict parsing requires a response_model.")
    
    try:
        parsed = response_model.model_validate_json(raw_json)
    except ValidationError as e:
        # Hard fail: do NOT fellback
        print("Relation validation failed:", e)
        return []

    entities_set = set(entities)
    valid_relations: List[Tuple[str, str, str]] = []

    for r in parsed.relations:
        if r.subject not in entities_set:
            print(f"Skipping relation: subject not found -> {r.subject}")
            continue

        if r.object not in entities_set:
            print(f"Skipping relation: object not found -> {r.object}")
            continue

        valid_relations.append((r.subject, r.predicate, r.object))

    return valid_relations


def _load_relations_prompt() -> str:
    """
    Load the relations prompt template from disk.

    Returns:
        str: Prompt template text loaded from `kg_gen/prompts/relations.txt`.
    """
    # Resolve the prompt path relative to this module.
    prompt_path = Path(__file__).parent.parent / "prompts" / "relations.txt"
    # Read the full prompt text as UTF-8.
    return prompt_path.read_text(encoding="utf-8")


def _create_relations_model(entities: List[str]):
    """
    Dynamically create Pydantic models with entity-constrained subject/object fields.

    Args:
        entities (List[str]): List of valid entity strings to constrain subject/object fields.

    Returns:
        tuple: (RelationItem, RelationsResponse) Pydantic model classes.
    """
    # Convert entities list into a Literal type for strict validation
    # EntityLiteral = Literal[tuple(entities)]  # type: ignore

    # Create RelationItem with constrained subject/object.
    RelationItem = create_model(
        "RelationItem",
        subject=(str, ...),
        predicate=(str, ...),
        object=(str, ...),
    )

    # Create RelationsResponse containing list of RelationItem.
    RelationsResponse = create_model(
        "RelationsResponse",
        relations=(List[RelationItem], ...),
    )

    return RelationItem, RelationsResponse


def _get_relations_litellm(
    input_data: str,
    entities: List[str],
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    usage_history: Optional[list[dict]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Extract relations from text using LiteLLM with strict JSON schema validation.

    Args:
        input_data (str): Source text to analyze for relations.
        entities (List[str]): Entity list used to constrain subject/object values.
        model (str): LLM model identifier.
        api_key (Optional[str]): Optional API key override.
        api_base (Optional[str]): Optional API base URL override.
        temperature (float): Sampling temperature for the LLM call.
        usage_history (Optional[list[dict]]): Optional list to record LiteLLM usage.

    Returns:
        List[Tuple[str, str, str]]: Extracted (subject, predicate, object) triples.
    """

    # Return empty if no entities are provided
    if not entities:
        return []

    # Load the system prompt template defining relation extraction rules.
    prompt_template = _load_relations_prompt()

    # Format entities as a bullet list for the prompt.
    entities_str = "\n".join(f"- {e}" for e in entities)
    
    # Build the user prompt with entities and the source text.
    user_prompt = f"""
    Here is the list of entities that were previously extracted from the source text:

    <entities>
    {entities_str}
    </entities>

    Here is the source text to analyze:

    <text>
    {input_data}
    </text>
    """

    # Create dynamic model with entity constraints.
    _, RelationsResponse = _create_relations_model(entities)

    # Generate JSON schema from Pydantic model
    schema = RelationsResponse.model_json_schema()
    schema["additionalProperties"] = False
    # Ensure nested object schemas also disallow extra properties
    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            if def_schema.get("type") == "object":
                def_schema["additionalProperties"] = False

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
                "name": "relations_response",
                "schema": schema,
                "strict": True,
            }
        },
    }

    # Optional overrides for API key/base
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    # Execute the LLM call.
    response = litellm.responses(**kwargs)

    # Optionally log usage statistics
    if usage_history is not None:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage:
            usage_history.append(usage)

    # Extract raw JSON from response
    raw_json = response.output[-1].content[0].text

    # Parse and validate relations, fallback to safe filtering if needed
    return parse_relations_response(raw_json, entities, RelationsResponse)


def _filter_entities(entities: List[str]) -> List[str]:
    """
    Remove entities containing double quotes to prevent API parsing errors.

    Args:
        entities (List[str]): Raw entity strings.

    Returns:
        List[str]: Filtered list of safe entities for prompt usage.
    """
    return [e for e in entities if '"' not in e]


def get_relations(
    input_data: str,
    entities: list[str],
    is_conversation: bool = False,
    context: str = "",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    usage_history: Optional[list[dict]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Public API for extracting relations from a text source.

    Args:
        input_data (str): Source text for relation extraction.
        entities (list[str]): Entity list to constrain subject/object.
        is_conversation (bool): Whether the input is a conversation (currently unused).
        context (str): Optional context string (currently unused).
        model (Optional[str]): LLM model identifier.
        api_key (Optional[str]): Optional API key override.
        api_base (Optional[str]): Optional API base URL override.
        temperature (float): Sampling temperature for the LLM call.
        usage_history (Optional[list[dict]]): Optional list to record LiteLLM usage.

    Returns:
        List[Tuple[str, str, str]]: Extracted relation triples (subject, predicate, object).
    """
    # Remove entities that contain quotes to prevent API parsing issues
    entities = _filter_entities(entities)
    if not entities:
        return []

    # Delegate to the LiteLLM-based implementation.
    return _get_relations_litellm(
        input_data,
        entities,
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        usage_history=usage_history,
    )
