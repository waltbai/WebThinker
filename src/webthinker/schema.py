"""Schema."""

from typing import Annotated, Any, Dict, Literal, Set, TypedDict

from langgraph.graph.message import add_messages


##################################
# State definition for report mode
##################################
class WebThinkerReportInputState(TypedDict):
    """Input state for WebThinker report mode."""
    research_question: str
    log_file: str


class WebThinkerReportOutputState(TypedDict):
    """Output state for WebThinker report mode."""
    article: str


class WebThinkerReportState(TypedDict):
    """State for WebThinker report mode."""
    # Input
    research_question: str
    log_file: str

    # Output
    article: str

    # Supervisor
    plan: str
    history: Annotated[list, add_messages]
    total_interactions: int
    research_complete_flag: bool

    # Search query
    url_cache: Dict[str, str]
    retriever: Any


####################################
# State definition for solution mode
####################################
class WebThinkerSolutionInputState(TypedDict):
    """Input state for WebThinker solution mode."""
    research_question: str
    log_file: str


class WebThinkerSolutionOutputState(TypedDict):
    """Output state for WebThinker solution mode."""
    solution: str


class WebThinkerSolutionState(TypedDict):
    """State for WebThinker solution mode."""
    # Input
    research_question: str
    log_file: str

    # Output
    solution: str

    # Supervisor
    history: Annotated[list, add_messages]
    total_interactions: int
    research_complete_flag: bool

    # Search query
    url_cache: Dict[str, str]
    executed_search_queries: Set[str]


###################
# Structured Output
###################
class EvaluationOutput(TypedDict):
    """Output of the evaluation result."""
    justification: Annotated[
        Literal["Correct", "Incorrect"],
        ...,
        "Justification for the prediction."
    ]
