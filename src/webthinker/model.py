"""Model."""

from langchain_core.language_models import LanguageModelLike
from langchain_qwq import ChatQwen, ChatQwQ

from src.webthinker.config import (BASEURL, EVALUATION_MODEL, PLANNER_MODEL,
                                   REPETITION_PENALTY, SEED, SUPERVISOR_MODEL,
                                   TEMPERATURE, TOP_K, TOP_P, WRITER_MODEL)


def get_planner_model() -> LanguageModelLike:
    """Get planner model."""
    return ChatQwen(
        name=PLANNER_MODEL,
        base_url=BASEURL,
        # enable_thinking=False,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        extra_body={
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
        },
        seed=SEED,
    )


def get_supervisor_model() -> LanguageModelLike:
    """Get supervisor model."""
    if SUPERVISOR_MODEL.startswith("qwq"):
        return ChatQwQ(
            name=SUPERVISOR_MODEL,
            base_url=BASEURL,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            extra_body={
                "top_k": TOP_K,
                "repetition_penalty": REPETITION_PENALTY,
            },
            seed=SEED,
        )
    return ChatQwen(
        name=SUPERVISOR_MODEL,
        base_url=BASEURL,
        enable_thinking=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        extra_body={
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
        },
        seed=SEED,
    )


def get_writer_model() -> LanguageModelLike:
    """Get writer model."""
    return ChatQwen(
        name=WRITER_MODEL,
        base_url=BASEURL,
        # enable_thinking=False,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        extra_body={
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
        },
        seed=SEED,
    )


def get_evaluation_model() -> LanguageModelLike:
    """Get evaluation model."""
    return ChatQwen(
        name=EVALUATION_MODEL,
        base_url=BASEURL,
        # enable_thinking=False,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        extra_body={
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
        },
        seed=SEED,
    )
