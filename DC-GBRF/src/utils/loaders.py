from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

import config


GPT5_REASONING_EFFORT_VALUES = {"low", "medium", "high", "xhigh"}


def validate_and_normalize_llm_params(
    *,
    temperature: float | None = 0.1,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    max_output_tokens: int | None = None,
) -> dict:
    normalized_reasoning_effort = None if reasoning_effort in {None, "", "none"} else str(reasoning_effort).strip().lower()
    if normalized_reasoning_effort is not None and normalized_reasoning_effort not in GPT5_REASONING_EFFORT_VALUES:
        raise ValueError("reasoning_effort must be one of: low, medium, high, xhigh")

    normalized_temperature = None if temperature is None else float(temperature)
    normalized_max_output_tokens = None if max_output_tokens is None else int(max_output_tokens)
    normalized_verbosity = None if verbosity in {None, ""} else str(verbosity).strip().lower()

    return {
        "temperature": normalized_temperature,
        "reasoning_effort": normalized_reasoning_effort,
        "verbosity": normalized_verbosity,
        "max_output_tokens": normalized_max_output_tokens,
    }


def build_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION_EMBEDDINGS,
        azure_deployment=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        max_retries=3,
    )


def build_llm(
    deployment_name: str,
    temperature: float | None = 0.1,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    max_output_tokens: int | None = None,
):
    llm_params = validate_and_normalize_llm_params(
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        max_output_tokens=max_output_tokens,
    )
    client_kwargs = {
        "azure_endpoint": config.AZURE_OPENAI_ENDPOINT,
        "api_key": config.AZURE_OPENAI_API_KEY,
        "api_version": config.AZURE_OPENAI_API_VERSION_CHAT,
        "azure_deployment": deployment_name,
        "max_retries": 3,
    }
    if llm_params["temperature"] is not None:
        client_kwargs["temperature"] = llm_params["temperature"]
    if llm_params["reasoning_effort"] is not None:
        client_kwargs["reasoning_effort"] = llm_params["reasoning_effort"]
    if llm_params["verbosity"] is not None:
        client_kwargs["verbosity"] = llm_params["verbosity"]
    if llm_params["max_output_tokens"] is not None:
        if str(deployment_name).lower().startswith("gpt-5"):
            client_kwargs["max_completion_tokens"] = llm_params["max_output_tokens"]
        else:
            client_kwargs["max_tokens"] = llm_params["max_output_tokens"]
    return AzureChatOpenAI(**client_kwargs)


def load_vector_store(path, embeddings):
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path_obj)
    return FAISS.load_local(str(path_obj), embeddings, allow_dangerous_deserialization=True)


def load_knowledge_base(path):
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path_obj)
    with open(path_obj, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)
