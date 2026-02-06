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
        - raw_json: Raw JSON string from the LLM response.
        - entities: List of valid entity strings used for validation.
        response_model: Optional Pydantic model used for strict validation.

    Returns:
        List[Tuple[str, str, str]]: List of (subject, predicate, object)
        tuples with valid entities only.
    """
    # Use a set for O(1) membership checks.
    entities_set = set(entities)

    # Try strict Pydantic validation first if a model is provided.
    if response_model is not None:
        try:
            parsed = response_model.model_validate_json(raw_json)
            return [(r.subject, r.predicate, r.object) for r in parsed.relations]
        except ValidationError:
            # Fall through to JSON parsing.
            pass

    # Fallback: parse as raw JSON and filter.
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return []

    # Handle both {"relations": [...]} and direct list formats.
    items = data.get("relations", data) if isinstance(data, dict) else data

    if not isinstance(items, list):
        return []

    relations = []
    for item in items:
        if not isinstance(item, dict):
            continue

        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")

        # Skip if missing required fields.
        if not all([subject, predicate, obj]):
            continue

        # Skip if subject or object not in valid entities.
        if subject not in entities_set or obj not in entities_set:
            continue

        relations.append((subject, predicate, obj))

    return relations


def _load_relations_prompt() -> str:
    """
    Load the relations prompt template from disk.

    Returns:
        str: Prompt template text loaded from `kg_gen/prompts/relations.txt`.
    """
    # Resolve the prompt path relative to this module.
    prompt_path = Path(__file__).parent.parent / "prompts" / "relations.txt"
    # Read the full prompt text as UTF-8.
    return prompt_path.read_text()


def _create_relations_model(entities: List[str]):
    """
    Create Pydantic models with entity-constrained subject/object fields.

    Args:
        - entities: List of valid entity strings used to constrain subject/object.

    Returns:
        tuple: (RelationItem, RelationsResponse) Pydantic model classes.
    """
    # Create a Literal type from the entities list.
    # Caller must ensure entities is non-empty to avoid Literal[()].
    EntityLiteral = Literal[tuple(entities)]  # type: ignore

    # Create RelationItem with constrained subject/object.
    RelationItem = create_model(
        "RelationItem",
        subject=(EntityLiteral, ...),
        predicate=(str, ...),
        object=(EntityLiteral, ...),
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
    Extract relations using LiteLLM with a JSON schema response.

    Args:
        - input_data: Source text to analyze for relations.
        - entities: Entity list used to constrain subject/object values.
        - model: LLM model identifier.
        - api_key: Optional API key override.
        - api_base: Optional API base URL override.
        - temperature: Sampling temperature for the LLM call.
        - usage_history: Optional list to accumulate LiteLLM usage dicts.

    Returns:
        List[Tuple[str, str, str]]: List of extracted (subject, predicate, object)
        relation triples.
    """
    # If there are no entities, there cannot be any relations.
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

    # Build schema with additionalProperties: false (required by OpenAI).
    schema = RelationsResponse.model_json_schema()
    schema["additionalProperties"] = False
    # Also set additionalProperties on nested objects.
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
    # Extract the raw JSON string from the response.
    raw_json = response.output[-1].content[0].text
    # Parse and validate relations with fallback filtering.
    return parse_relations_response(raw_json, entities, RelationsResponse)


def _filter_entities(entities: List[str]) -> List[str]:
    """
    Filter out entities that contain quotes to avoid API parsing issues.

    Args:
        - entities: Raw entity strings from extraction.

    Returns:
        List[str]: Filtered entity list safe for prompt inclusion.
    """
    # Exclude entities containing double quotes (not accepted by the API).
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
    Public API to extract relations from input text.

    Args:
        - input_data: Source text to analyze.
        - entities: Entity list to constrain relation extraction.
        - is_conversation: Whether the input represents dialogue. Currently unused.
        - context: Optional context string. Currently unused by this implementation.
        - model: LLM model identifier to use.
        - api_key: Optional API key override.
        - api_base: Optional API base URL override.
        - temperature: Sampling temperature for the LLM call.
        - usage_history: Optional list to accumulate LiteLLM usage dicts.

    Returns:
        List[Tuple[str, str, str]]: Extracted relation triples.
    """
    # Filter out entities that are problematic for the API.
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
