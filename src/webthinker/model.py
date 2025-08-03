"""Model."""

from langchain_core.language_models import LanguageModelLike
from langchain_qwq import ChatQwen

from src.webthinker.config import BASEURL, PLANNER_MODEL, SEED, SUPERVISOR_MODEL, WRITER_MODEL


def get_planner_model() -> LanguageModelLike:
    """Get planner model."""
    return ChatQwen(
        name=PLANNER_MODEL,
        base_url=BASEURL,
        enable_thinking=False,
        temperature=0,
        seed=SEED,
    )


def get_supervisor_model() -> LanguageModelLike:
    """Get supervisor model."""
    return ChatQwen(
        name=SUPERVISOR_MODEL,
        base_url=BASEURL,
        enable_thinking=True,
        temperature=0,
        seed=SEED,
    )


def get_writer_model() -> LanguageModelLike:
    """Get writer model."""
    return ChatQwen(
        name=WRITER_MODEL,
        base_url=BASEURL,
        enable_thinking=False,
        temperature=0,
        seed=SEED,
    )
