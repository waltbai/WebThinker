"""State graph for solution mode."""

from typing import Annotated, Literal

from langchain_core.messages import (SystemMessage, ToolMessage,
                                     get_buffer_string)
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command

from src.webthinker.config import (MAX_INTERACTIONS, MAX_OUTPUT_RETRY,
                                   MAX_SEARCH_LIMIT, SEARCH_TOOL, SEARCH_TOP_K)
from src.webthinker.model import get_supervisor_model, get_writer_model
from src.webthinker.prompts import (EXTRACT_INFORMATION_PROMPT,
                                    SEARCH_INTENT_PROMPT,
                                    SUMMARIZE_SOLUTION_PROMPT,
                                    SUPERVISOR_PROMPT)
from src.webthinker.schema import (WebThinkerSolutionInputState,
                                   WebThinkerSolutionOutputState,
                                   WebThinkerSolutionState)
from src.webthinker.utils import (extract_context_by_snippet, fetch_content,
                                  format_search_results, get_logger,
                                  search_google_serper, search_tavily)


def webthinker():
    """WebThinker solution mode."""
    # Init state graph
    builder = StateGraph(
        WebThinkerSolutionState,
        input_schema=WebThinkerSolutionInputState,
        output_schema=WebThinkerSolutionOutputState,
    )

    # Nodes
    builder.add_node(
        "supervisor",
        supervisor,
        destinations=("supervisor_tool", "summarize_solution"),
    )
    builder.add_node(
        "supervisor_tool",
        ToolNode(
            [search_query, research_complete],
            messages_key="history",
            handle_tool_errors=False,
        ),
    )
    builder.add_node(
        "summarize_solution",
        summarize_solution,
    )
    # Edges
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor_tool", "supervisor")
    # builder.add_edge("supervisor", "supervisor_tool")
    # builder.add_edge("supervisor", "summarize_solution")
    builder.add_edge("summarize_solution", END)
    return builder.compile()


def supervisor(
    state: WebThinkerSolutionState,
    prompt: Annotated[str, InjectedToolArg] = SUPERVISOR_PROMPT,
) -> Command[Literal["supervisor_tool", "__end__"]]:
    """Supervisor."""
    research_question = state.get("research_question", "")
    history = state.get("history", [])
    research_complete_flag = state.get("research_complete_flag", False)
    total_interactions = state.get("total_interactions", 0)
    model = get_supervisor_model()
    logger = get_logger("webthinker.supervisor", state.get("log_file", None))
    logger.info("=== Supervisor ===")

    # Check if research complete
    if research_complete_flag:
        logger.info("Research complete.")
        return Command(goto=END)

    # Hint the supervisor to call tools
    if not history:
        init_prompt = prompt.format(
            search_limit=MAX_SEARCH_LIMIT,
            research_question=research_question,
        )
        history.append(SystemMessage(init_prompt))
    else:
        history.append(SystemMessage("Please generate next tool call."))

    # Call tools
    tools = [search_query, research_complete]
    model_with_tool = model.bind_tools(tools).with_retry(
        stop_after_attempt=MAX_OUTPUT_RETRY,
    )
    response = model_with_tool.invoke(history)

    # Log output
    logger.info(
        "Tool calls:\n"
        "%s\n",
        "\n".join([
            f"Name: {call['name']}, Args: {call['args']}"
            for call in response.tool_calls
        ])
    )

     # Check tool calls
    if not response.tool_calls or total_interactions >= MAX_INTERACTIONS:
        return Command(goto="summarize_solution")

    return Command(
        goto="supervisor_tool",
        update={
            "history": history + [response],
        }
    )


def summarize_solution(
    state: WebThinkerSolutionState,
    prompt: Annotated[str, InjectedToolArg] = SUMMARIZE_SOLUTION_PROMPT,
) -> Command[Literal["__end__"]]:
    """Summarize the solution."""
    research_question = state.get("research_question", "")
    history = state.get("history", [])
    model = get_writer_model()
    logger = get_logger("webthinker.summarize_solution", state.get("log_file", None))
    logger.info("=== Summarize Solution ===")

    previous_thoughts = get_buffer_string(history)
    content = prompt.format(
        research_question=research_question,
        previous_thoughts=previous_thoughts,
    )
    response = model.invoke([SystemMessage(content)])
    solution = response.content

    logger.info(
        "Final solution:\n"
        "%s\n",
        solution,
    )

    return {
        "solution": solution,
    }


########################
# Research Complete tool
########################
@tool
def research_complete(
    final_answer: Annotated[str, ..., "the final answer"],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Complete research and output final answer."""
    logger = get_logger("webthinker.research_complete", state.get("log_file", None))
    logger.info(
        "=== Research Complete ===\n"
        "Final answer: %s",
        final_answer
    )
    return Command(update={
        "research_complete_flag": True,
        "solution": final_answer,
        "history": [ToolMessage(
            f"Final answer: {final_answer}", tool_call_id=tool_call_id)
        ],
    })


###################
# Search query tool
###################
@tool
def search_query(
    query: Annotated[str, ..., "the query to search on the web."],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    intent_prompt: Annotated[str, InjectedToolArg] = SEARCH_INTENT_PROMPT,
    extract_information_prompt: Annotated[str, InjectedToolArg] = EXTRACT_INFORMATION_PROMPT,
) -> str:
    """Search query tool."""
    total_interactions = state.get("total_interactions", 0)
    history = state.get("history", [])
    url_cache = state.get("url_cache", {})
    executed_search_queries = state.get("executed_search_queries", set())
    model = get_writer_model()
    logger = get_logger("webthinker.search_query", state.get("log_file", None))
    logger.info("=== Search Query ===")

    # Check Query
    if query in executed_search_queries:
        logger.info("Query has been executed before.")
        return Command(update={
            "total_interactions": total_interactions + 1,
            "history": [ToolMessage(
                "You have already searched for this query.",
                tool_call_id=tool_call_id,
            )],
        })

    # Generate search intent
    previous_thoughts = get_buffer_string(history)
    content = intent_prompt.format(
        # research_question=research_question,
        previous_thoughts=previous_thoughts,
    )
    response = model.invoke([SystemMessage(content)])
    search_intent = response.content
    logger.info(
        "Search intent:\n"
        "%s\n",
        search_intent,
    )

    # Execute search
    if SEARCH_TOOL == "tavily":
        results = search_tavily(query=query, max_results=SEARCH_TOP_K)
    elif SEARCH_TOOL == "google":
        results = search_google_serper(query=query, max_results=SEARCH_TOP_K)
    else:
        logger.warning(
            "Unknown search tool: %s, default to use google.",
            SEARCH_TOOL,
        )
        results = search_google_serper(query=query, max_results=SEARCH_TOP_K)
    logger.info(
        "Totally %d results found:\n"
        "%s\n",
        len(results),
        "\n".join(r["url"] for r in results),
    )

    # Fetch webpages
    url_to_fetch = [result["url"] for result in results if result["url"] not in url_cache]
    for url in url_to_fetch:
        content = fetch_content(url)
        url_cache[url] = content

    # Truncate contents
    for i, result in enumerate(results):
        raw_content = url_cache[result["url"]]
        # Retain more chars for higher rank documents
        if i < 5:
            context_chars = 4000
        else:
            context_chars = 2000
        # Extract original contents according to snippet
        if raw_content != "Can not fetch the page content.":
            context = extract_context_by_snippet(
                raw_content=raw_content,
                snippet=result["snippet"],
                context_chars=context_chars,
            )
            result["content"] = context
        else:
            result["content"] = result["snippet"]

    # Extract relevant information
    formatted_results = format_search_results(results)
    content = extract_information_prompt.format(
        query=query,
        search_intent=search_intent,
        search_results=formatted_results,
    )
    response = model.invoke([SystemMessage(content)])
    final_information = response.content
    executed_search_queries.add(query)
    logger.info(
        "Relevant information extracted:\n"
        "%s\n",
        final_information
    )

    # Update state
    return Command(update={
        "url_cache": url_cache,
        "executed_search_queries": executed_search_queries,
        "total_interactions": total_interactions + 1,
        "history": [ToolMessage(final_information, tool_call_id=tool_call_id)],
    })
