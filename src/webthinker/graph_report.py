"""State graph for report mode."""

from typing import Annotated, Literal

from langchain_core.messages import (SystemMessage, ToolMessage,
                                     get_buffer_string)
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command

from src.webthinker.config import (MAX_INTERACTIONS, MAX_OUTPUT_RETRY,
                                   SEARCH_TOP_K)
from src.webthinker.model import (get_planner_model, get_supervisor_model,
                                  get_writer_model)
from src.webthinker.prompts_report import (EDIT_ARTICLE_PROMPT,
                                           EXTRACT_INFORMATION_PROMPT,
                                           FINAL_REFINEMENT_PROMPT,
                                           GENERATE_PLAN_PROMPT,
                                           SEARCH_INTENT_PROMPT,
                                           SUPERVISOR_PROMPT, TITLE_PROMPT,
                                           WRITE_SECTION_PROMPT)
from src.webthinker.schema import (WebThinkerReportInputState,
                                   WebThinkerReportOutputState,
                                   WebThinkerReportState)
from src.webthinker.utils import (BM25Retriever, extract_context_by_snippet,
                                  extract_outline, fetch_content_local,
                                  format_search_results, get_logger,
                                  search_google_serper)


############
# WebThinker
############
def webthinker_report():
    """WebThinker report mode."""
    # Init state graph
    builder = StateGraph(
        WebThinkerReportState,
        input_schema=WebThinkerReportInputState,
        output_schema=WebThinkerReportOutputState,
    )

    # Nodes
    builder.add_node(
        "generate_plan",
        generate_plan,
    )
    builder.add_node(
        "supervisor",
        supervisor,
        destinations=("supervisor_tool", "final_refinement"),
    )
    builder.add_node(
        "supervisor_tool",
        ToolNode(
            [search_query, write_section, check_article, edit_article, research_complete],
            messages_key="history",
        ),
    )
    builder.add_node(
        "final_refinement",
        final_refinement,
    )

    # Edges
    builder.add_edge(START, "generate_plan")
    builder.add_edge("generate_plan", "supervisor")
    builder.add_edge("supervisor_tool", "supervisor")
    # Comment destinations to hint the overall graph
    # builder.add_edge("supervisor", "supervisor_tool")
    # builder.add_edge("supervisor", "final_refinement")
    builder.add_edge("final_refinement", END)
    return builder.compile()


def generate_plan(
    state: WebThinkerReportState,
    prompt: str = GENERATE_PLAN_PROMPT,
) -> Command[Literal["supervisor"]]:
    """Generate a research plan for the research question."""
    research_question = state.get("research_question", "")
    model = get_planner_model()
    logger = get_logger("webthinker.generate_plan", state.get("log_file", None))
    logger.info("=== Generate Plan ===")

    # Get research plan
    content = prompt.format(research_question=research_question)
    response = model.invoke([SystemMessage(content)])
    plan = response.content

    # Log output
    logger.info(
        "Generated plan:\n"
        "%s\n",
        plan,
    )

    return {
        "plan": plan,
    }


def supervisor(
    state: WebThinkerReportState,
    prompt: str = SUPERVISOR_PROMPT,
) -> Command[Literal["supervisor_tool", "final_refinement"]]:
    """Supervisor."""
    research_question = state.get("research_question", "")
    plan = state.get("plan", "")
    history = state.get("history", [])
    research_complete_flag = state.get("research_complete_flag", False)
    total_interactions = state.get("total_interactions", 0)
    model = get_supervisor_model()
    logger = get_logger("webthinker.supervisor", state.get("log_file", None))
    logger.info("=== Supervisor ===")

    # Check if research complete
    if research_complete_flag or total_interactions >= MAX_INTERACTIONS:
        return Command(goto="final_refinement")

    # Hint the supervisor to call tools
    if not history:
        init_prompt = prompt.format(
            research_question=research_question,
            plan=plan,
        )
        history.append(SystemMessage(init_prompt))
    else:
        history.append(SystemMessage("Please generate next tool call."))

    # Call tools
    tools = [search_query, write_section, check_article, edit_article, research_complete]
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
    if not response.tool_calls:
        return Command(goto="final_refinement")

    return Command(
        goto="supervisor_tool",
        update={
            "history": history + [response],
        }
    )


def final_refinement(
    state: WebThinkerReportState,
    prompt: str = FINAL_REFINEMENT_PROMPT,
) -> Command[Literal["__end__"]]:
    """Final refinement."""
    research_question = state.get("research_question", "")
    article = state.get("article", "")
    model = get_writer_model()
    logger = get_logger("webthinker.final_refinement", state.get("log_file", None))
    logger.info("=== Final Refinement ===")

    # Final refinement
    content = prompt.format(
        research_question=research_question,
        article=article,
    )
    response = model.invoke([SystemMessage(content)])
    final_report = response.content

    # Log output
    logger.info(
        "Final report:\n"
        "%s\n",
        final_report,
    )

    return {
        "article": final_report,
    }


########################
# Research Complete tool
########################
@tool
def research_complete(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Research complete."""
    logger = get_logger("webthinker.research_complete", state.get("log_file", None))
    logger.info(
        "=== Research Complete ===\n"
    )
    return Command(update={
        "research_complete_flag": True,
        "history": [ToolMessage("Research complete.", tool_call_id=tool_call_id)],
    })


####################
# Write section tool
####################
@tool
def write_section(
    section_title: Annotated[str, ..., "the section title."],
    section_goal: Annotated[str, ..., "the goal of the section."],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    prompt: Annotated[str, InjectedToolArg] = WRITE_SECTION_PROMPT,
) -> str:
    """Write section tool."""
    research_question = state.get("research_question", "")
    total_interactions = state.get("total_interactions", 0)
    article_outline = state.get("article_outline", "")
    history = state.get("history", [])
    article = state.get("article", "")
    retriever = state.get("retriever", BM25Retriever())
    model = get_writer_model()
    logger = get_logger("webthinker.write_section", state.get("log_file", None))
    logger.info("=== Write Section ===")

    # Retrieve relevant documents
    query = f"{section_title} {section_goal}"
    relevant_documents = retriever.invoke(query)
    formatted_documents = "".join([
        f"Document {i}:\n{doc}\n\n"
        for i, doc in enumerate(relevant_documents)
    ])

    # Generate section content
    previous_thoughts = get_buffer_string(history)
    content = prompt.format(
        relevant_documents=formatted_documents,
        research_question=research_question,
        previous_thoughts=previous_thoughts,
        article_outline=article_outline,
        section_title=section_title,
        section_goal=section_goal,
    )
    response = model.invoke([SystemMessage(content)])
    section_content = response.content

    # Write section
    article += section_content
    article_outline = extract_outline(article)

    # Update state
    return Command(update={
        "article": article,
        "article_outline": article_outline,
        "total_interactions": total_interactions + 1,
        "history": [ToolMessage("Section written.", tool_call_id=tool_call_id)]
    })


####################
# Check article tool
####################
@tool
def check_article(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    prompt: Annotated[str, InjectedToolArg] = TITLE_PROMPT,
) -> str:
    """Check article tool."""
    research_question = state.get("research_question", "")
    total_interactions = state.get("total_interactions", 0)
    article = state.get("article", "")
    model = get_writer_model()
    logger = get_logger("webthinker.check_article", state.get("log_file", None))
    logger.info("=== Check Article ===")

    # Check title
    if not article.startswith("# "):
        content = prompt.format(
            research_question=research_question,
            article=article,
        )
        response = model.invoke([SystemMessage(content)])
        title = response.content
        article = f"# {title}\n" + article

    # Extract outline
    article_outline = extract_outline(article)

    # Update state
    return Command(update={
        "article_outline": article_outline,
        "total_interactions": total_interactions + 1,
        "history": [ToolMessage(article_outline, tool_call_id=tool_call_id)],
    })


####################
# Edit article tool
####################
@tool
def edit_article(
    instruction: Annotated[str, ..., "the instruction for editing the article."],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    prompt: Annotated[str, InjectedToolArg] = EDIT_ARTICLE_PROMPT,
) -> str:
    """Edit article tool."""
    article = state.get("article", "")
    total_interactions = state.get("total_interactions", 0)
    model = get_writer_model()
    logger = get_logger("webthinker.edit_article", state.get("log_file", None))
    logger.info("=== Edit Article ===")

    # Edit article according to instruction
    content = prompt.format(
        instruction=instruction,
        article=article,
    )
    response = model.invoke([SystemMessage(content)])
    edited_article = response.content

    # Update state
    return Command(update={
        "article": edited_article,
        "total_interactions": total_interactions + 1,
        "history": [ToolMessage("edit done", tool_call_id=tool_call_id)],
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
    research_question = state.get("research_question", "")
    total_interactions = state.get("total_interactions", 0)
    history = state.get("history", [])
    url_cache = state.get("url_cache", {})
    retriever = state.get("retriever", BM25Retriever())
    model = get_writer_model()
    logger = get_logger("webthinker.search_query", state.get("log_file", None))
    logger.info("=== Search Query ===")

    # Generate search intent
    previous_thoughts = get_buffer_string(history)
    content = intent_prompt.format(
        research_question=research_question,
        previous_thoughts=previous_thoughts,
    )
    response = model.invoke([SystemMessage(content)])
    search_intent = response.content

    # Execute search
    results = search_google_serper(query=query, max_results=SEARCH_TOP_K)
    # Fetch webpages
    url_to_fetch = [result["url"] for result in results if result["url"] not in url_cache]
    for url in url_to_fetch:
        content = fetch_content_local(url)
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

    # Update state
    new_docs = [result["content"] for result in results]
    retriever.add_documents(new_docs)
    return Command(update={
        "url_cache": url_cache,
        "retriever": retriever,
        "total_interactions": total_interactions + 1,
        "history": [ToolMessage(final_information, tool_call_id=tool_call_id)],
    })
