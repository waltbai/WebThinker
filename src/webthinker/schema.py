"""Schema."""

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph.message import add_messages


##################
# State definition
##################
class WebThinkerReportInputState(TypedDict):
    """Input state for WebThinker."""
    research_question: str
    log_file: str


class WebThinkerReportOutputState(TypedDict):
    """Output state for WebThinker."""
    article: str


class WebThinkerReportState(TypedDict):
    """State for WebThinker report generation mode."""
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
    document_memory: List[str]
    retriever: Any
