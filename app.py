"""AI Meeting Notes Analyzer | Streamlit + LangGraph + OpenAI."""

import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field


st.set_page_config(
    page_title="Meeting Intelligence | AI Notes Analyzer",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded",
)

TEAMS_PURPLE = "#6264A7"
DEEP_PURPLE = "#464775"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], [data-testid="stApp"] {font-family:'Inter',sans-serif;}
    [data-testid="stAppViewContainer"] {background:#F6F7FB;}
    [data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid #E7E8F1;}
    .block-container {max-width:1400px;padding-top:2rem;padding-bottom:3rem;}
    h1,h2,h3 {color:#24243A;letter-spacing:-.025em;}
    .hero {background:linear-gradient(115deg,#464775 0%,#6264A7 65%,#8587CB 100%);
        border-radius:20px;padding:34px 38px;color:#FFFFFF;margin-bottom:24px;
        box-shadow:0 12px 28px rgba(70,71,117,.16);}
    .hero .eyebrow {font-size:12px;font-weight:800;letter-spacing:2px;opacity:.83;}
    .hero h1 {color:#FFFFFF;font-size:36px;margin:12px 0 10px;font-weight:800;}
    .hero p {color:#F2F1FF;font-size:15px;max-width:780px;margin:0;line-height:1.65;}
    .surface {background:#FFFFFF;border:1px solid #E7E8F1;border-radius:16px;
        padding:20px 22px;margin-bottom:14px;box-shadow:0 3px 12px rgba(40,43,78,.035);}
    .metric {background:#FFFFFF;border:1px solid #E7E8F1;border-radius:16px;
        padding:19px 20px;min-height:115px;box-shadow:0 3px 12px rgba(40,43,78,.035);}
    .metric .label {color:#72768A;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;}
    .metric .value {color:#24243A;font-size:29px;font-weight:800;margin-top:10px;}
    .metric .hint {color:#777B8D;font-size:11px;margin-top:4px;}
    .section-title {font-size:18px;font-weight:800;color:#25263E;margin:12px 0 12px;}
    .topic-chip {display:inline-block;background:#EEEDFA;color:#464775;border:1px solid #DEDCF5;
        padding:9px 13px;border-radius:10px;margin:0 7px 9px 0;font-size:12px;font-weight:650;}
    .pill {display:inline-block;padding:5px 11px;border-radius:100px;font-size:11px;font-weight:800;}
    .pill-high {background:#FDE9E9;color:#B42332;}
    .pill-medium {background:#FFF1D6;color:#9A6300;}
    .pill-low {background:#E1F4EB;color:#176A49;}
    .action-card {background:#FFFFFF;border:1px solid #E8E8F0;border-radius:13px;
        padding:16px 18px;margin-bottom:11px;}
    .action-title {font-size:14px;font-weight:700;color:#282943;margin:8px 0 12px;line-height:1.55;}
    .action-meta {font-size:12px;color:#73768A;line-height:1.7;}
    .sidebar-brand {font-size:20px;font-weight:800;color:#464775;margin:5px 0 6px;}
    .sidebar-note {font-size:12px;color:#74778A;line-height:1.7;}
    div.stButton > button[kind="primary"] {background:#6264A7;border-color:#6264A7;
        border-radius:10px;font-weight:750;min-height:46px;}
    div.stButton > button[kind="primary"]:hover {background:#464775;border-color:#464775;}
    div.stDownloadButton > button {border-radius:10px;border-color:#DADBED;font-weight:650;}
    [data-testid="stTabs"] button {font-weight:700;}
    #MainMenu, footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Data contracts -----------------------------
class TopicOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topics: list[str] = Field(description="Distinct, meaningful discussion topics.")


class SummaryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(description="A factual meeting summary in 3 to 5 sentences.")


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str = Field(description="Specific actionable task, not a vague observation.")
    owner: str = Field(description="Explicit responsible speaker or 'Not specified'.")
    deadline: str | None = Field(description="Explicit deadline or time commitment; null if absent.")
    source_evidence: str = Field(description="Exact supporting quote from the transcript.")


class ActionsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_items: list[ActionItem] = Field(description="Distinct, supported action items.")


class PriorityItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str
    priority: Literal["High", "Medium", "Low"]


class PrioritiesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    priorities: list[PriorityItem]


class MeetingState(TypedDict):
    transcript: str
    topics: NotRequired[list[str]]
    summary: NotRequired[str]
    action_items: NotRequired[list[ActionItem]]
    priorities: NotRequired[list[PriorityItem]]
    final_report: NotRequired[str]


def clean_missing(value: object) -> str:
    if value is None or str(value).strip().lower() in {"", "null", "none", "n/a", "unknown"}:
        return "Not specified"
    return str(value).strip()


def safe_md(value: object) -> str:
    return clean_missing(value).replace("|", r"\|").replace("\n", " ")


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def build_graph(api_key: str, model: str):
    # Remove a stale local override without disabling certificate validation.
    os.environ.pop("SSL_CERT_FILE", None)
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
        timeout=90,
        max_retries=2,
        max_tokens=4000,
    )

    def ask(schema, system_prompt: str, user_prompt: str):
        return llm.with_structured_output(schema, method="json_schema").invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

    def topics_node(state: MeetingState) -> dict:
        response = ask(
            TopicOutput,
            "Identify distinct, meaningful topics in the transcript. Do not invent topics. "
            "Use concise labels; avoid duplicates.",
            state["transcript"],
        )
        return {"topics": response.topics}

    def summary_node(state: MeetingState) -> dict:
        response = ask(
            SummaryOutput,
            "Write a factual 3–5 sentence meeting summary. Cover the central problem, "
            "important findings, decisions, and follow-up. Distinguish hypotheses from "
            "confirmed facts. Do not invent commitments.",
            state["transcript"],
        )
        return {"summary": response.summary}

    def actions_node(state: MeetingState) -> dict:
        response = ask(
            ActionsOutput,
            "Extract all distinct commitments, assigned tasks, and clearly agreed follow-ups. "
            "Capture tasks mentioned anywhere, including testing AFTER a fix, release steps, "
            "customer updates, and meetings. Combine duplicate mentions of the SAME task "
            "but do not merge different tasks. Only assign an owner when the transcript "
            "explicitly supports that owner. Suggestions from one person do not make that "
            "person responsible. Attach timing only to the task it actually qualifies; "
            "do not transfer deadlines from nearby statements. Preserve conditional timing "
            "as conditional. Use null for unspecified deadlines. Each source_evidence must "
            "be an exact quote from the transcript. Return [] if no actionable tasks exist.",
            state["transcript"],
        )
        # Normalize common model formatting issues and remove exact duplicates.
        unique = {}
        for action in response.action_items:
            key = normalize(action.task)
            if key not in unique:
                unique[key] = action.model_copy(update={
                    "owner": clean_missing(action.owner),
                    "deadline": None if clean_missing(action.deadline) == "Not specified" else action.deadline,
                })
        return {"action_items": list(unique.values())}

    def priorities_node(state: MeetingState) -> dict:
        actions = state.get("action_items", [])
        if not actions:
            return {"priorities": []}
        payload = [{"task": a.task, "owner": a.owner, "deadline": a.deadline} for a in actions]
        response = ask(
            PrioritiesOutput,
            "Classify each action as High, Medium, or Low. High: urgent customer-impacting "
            "work, imminent deadlines, critical blockers. Medium: important supporting work "
            "and follow-up. Low: nonurgent routine work. Consider task dependencies and "
            "meeting context. Preserve every task string EXACTLY, with no additions, "
            "omissions, or duplicates.",
            "MEETING:\n" + state["transcript"] + "\n\nACTIONS:\n" + json.dumps(payload, ensure_ascii=False),
        )
        expected = Counter(a.task for a in actions)
        received = Counter(p.task for p in response.priorities)
        if expected != received:
            raise ValueError("Priority agent changed or omitted action items. Please rerun analysis.")
        return {"priorities": response.priorities}

    def route_actions(state: MeetingState) -> str:
        return "priority" if state.get("action_items") else "report"

    def report_node(state: MeetingState) -> dict:
        topics = state.get("topics", [])
        actions = state.get("action_items", [])
        priority_map = {p.task: p.priority for p in state.get("priorities", [])}
        counts = Counter(priority_map.values())
        lines = [
            "# AI Meeting Analysis Report", "", "## Meeting Summary", "",
            state.get("summary", "No summary available."), "", "## Key Discussion Topics", "",
        ]
        lines.extend(f"{i}. {topic}" for i, topic in enumerate(topics, 1))
        if not topics:
            lines.append("No topics identified.")
        lines += ["", "## Action Items", ""]
        if actions:
            lines += ["| # | Action Item | Owner | Deadline / Trigger | Priority |",
                      "|---|---|---|---|---|"]
            for i, action in enumerate(actions, 1):
                lines.append(
                    f"| {i} | {safe_md(action.task)} | {safe_md(action.owner)} | "
                    f"{safe_md(action.deadline)} | {priority_map.get(action.task, 'Not classified')} |"
                )
        else:
            lines.append("No action items identified.")
        lines += [
            "", "## Priority Distribution", "",
            f"- High: {counts['High']}", f"- Medium: {counts['Medium']}",
            f"- Low: {counts['Low']}", "", "## Meeting Statistics", "",
            f"- Discussion topics: {len(topics)}", f"- Action items: {len(actions)}",
            f"- Classified tasks: {len(state.get('priorities', []))}",
            f"- Unassigned tasks: {sum(clean_missing(a.owner) == 'Not specified' for a in actions)}",
        ]
        return {"final_report": "\n".join(lines)}

    workflow = StateGraph(MeetingState)
    workflow.add_node("topics", topics_node)
    workflow.add_node("summary", summary_node)
    workflow.add_node("actions", actions_node)
    workflow.add_node("priority", priorities_node)
    workflow.add_node("report", report_node)
    workflow.add_edge(START, "topics")
    workflow.add_edge("topics", "summary")
    workflow.add_edge("summary", "actions")
    workflow.add_conditional_edges("actions", route_actions, {"priority": "priority", "report": "report"})
    workflow.add_edge("priority", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def state_to_json(state: dict, elapsed: float) -> str:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_seconds": round(elapsed, 2),
        "summary": state.get("summary", ""),
        "topics": state.get("topics", []),
        "action_items": [
            {**a.model_dump(), "priority": next(
                (p.priority for p in state.get("priorities", []) if p.task == a.task),
                "Not classified",
            )}
            for a in state.get("action_items", [])
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def metric_card(label: str, value: object, hint: str = "") -> None:
    st.markdown(
        f'<div class="metric"><div class="label">{escape(str(label))}</div>'
        f'<div class="value">{escape(str(value))}</div>'
        f'<div class="hint">{escape(str(hint))}</div></div>',
        unsafe_allow_html=True,
    )


def priority_pill(priority: str) -> str:
    level = priority if priority in {"High", "Medium", "Low"} else "Low"
    return f'<span class="pill pill-{level.lower()}">{escape(priority)}</span>'


def get_api_key() -> str:
    # Streamlit Cloud Secrets take precedence; .env is for local development.
    try:
        secret = st.secrets.get("OPENAI_API_KEY", "")
    except (FileNotFoundError, OSError, KeyError):
        secret = ""
    return str(secret or os.getenv("OPENAI_API_KEY", "")).strip()


# ----------------------------- App interface -----------------------------
load_dotenv()
if "result" not in st.session_state:
    st.session_state.result = None
if "elapsed" not in st.session_state:
    st.session_state.elapsed = None
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

with st.sidebar:
    st.markdown('<div class="sidebar-brand">◈ Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-note">AI-powered meeting analysis built with LangGraph and OpenAI.</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**WORKSPACE**")
    st.caption("Transcript → Topics → Summary → Actions → Priorities → Report")
    st.divider()
    st.markdown("**MODEL SETTINGS**")
    model_name = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.caption("gpt-4o-mini is the default cost-efficient option.")
    api_key = get_api_key()
    if api_key:
        st.success("API connection configured", icon="✅")
    else:
        st.warning("OPENAI_API_KEY is not configured.", icon="⚠️")
    st.divider()
    st.markdown("**ABOUT**")
    st.markdown(
        '<div class="sidebar-note">Five LangGraph nodes · Structured Pydantic outputs · '
        'Conditional priority routing · Exportable reports</div>',
        unsafe_allow_html=True,
    )
    st.caption("Built for demonstration and productivity. Review AI-generated tasks before acting on them.")

st.markdown(
    '<div class="hero"><div class="eyebrow">AI-POWERED WORKSPACE</div>'
    '<h1>Meeting Notes Analyzer</h1>'
    '<p>Turn meeting transcripts into clear summaries, actionable follow-ups, '
    'assigned owners, and prioritized tasks — powered by a multi-agent LangGraph workflow.</p></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">01 / Meeting transcript</div>', unsafe_allow_html=True)
st.caption("Upload a plain-text transcript or paste meeting notes below. Speaker-labeled transcripts produce better ownership attribution.")

upload_col, clear_col = st.columns([4, 1], vertical_alignment="bottom")
with upload_col:
    uploaded_file = st.file_uploader("Upload transcript (.txt)", type=["txt"], label_visibility="visible")
with clear_col:
    clear_clicked = st.button("Clear workspace", use_container_width=True)

if clear_clicked:
    st.session_state.transcript = ""
    st.session_state.result = None
    st.session_state.elapsed = None
    st.session_state.pop("last_upload_signature", None)
    st.rerun()

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    signature = (uploaded_file.name, len(file_bytes), hash(file_bytes))
    if st.session_state.get("last_upload_signature") != signature:
        try:
            st.session_state.transcript = file_bytes.decode("utf-8-sig")
            st.session_state.last_upload_signature = signature
            st.session_state.result = None
        except UnicodeDecodeError:
            st.error("Please upload a UTF-8 encoded .txt file.")

transcript = st.text_area(
    "Meeting transcript",
    key="transcript",
    height=245,
    placeholder="Product Manager: Good morning, everyone...\nQA Tester: I'll prepare a bug report by 2 PM...",
)

word_count = len(transcript.split())
speaker_labels = set(re.findall(r"(?m)^\s*([^:\n]{2,70}):\s*\S", transcript))
info_col, button_col = st.columns([3, 1], vertical_alignment="center")
with info_col:
    st.caption(f"{word_count:,} words  •  {len(transcript):,} characters  •  {len(speaker_labels)} detected speaker labels")
with button_col:
    analyze = st.button("✦ Analyze meeting", type="primary", use_container_width=True)

if analyze:
    if not transcript.strip():
        st.warning("Please paste or upload a transcript before running analysis.")
    elif len(transcript.strip()) < 80:
        st.warning("The transcript is very short. Please provide more meeting context.")
    elif not api_key:
        st.error("Missing OPENAI_API_KEY. Add it to your local .env file or Streamlit Cloud Secrets.")
    else:
        st.session_state.result = None
        with st.spinner("AI agents are analyzing your meeting. This may take a few moments..."):
            try:
                graph = build_graph(api_key, model_name)
                started = time.perf_counter()
                result = graph.invoke({"transcript": transcript})
                st.session_state.elapsed = time.perf_counter() - started
                st.session_state.result = result
                st.session_state.analysis_model = model_name
                st.session_state.analysis_transcript = transcript
            except Exception as exc:
                st.error("The analysis could not be completed. Please check your API settings and try again.")
                with st.expander("Technical error details"):
                    st.code(f"{type(exc).__name__}: {str(exc)}")

result = st.session_state.result
if result:
    st.divider()
    st.markdown('<div class="section-title">02 / Analysis overview</div>', unsafe_allow_html=True)
    actions = result.get("action_items", [])
    topics = result.get("topics", [])
    priorities = {p.task: p.priority for p in result.get("priorities", [])}
    counts = Counter(priorities.values())

    if st.session_state.get("analysis_transcript") != transcript:
        st.info("The transcript has changed since this analysis. Click **Analyze meeting** to refresh the results.")

    cols = st.columns(5)
    metrics = [
        ("Discussion topics", len(topics), "Key themes"),
        ("Action items", len(actions), "Extracted follow-ups"),
        ("High priority", counts["High"], "Needs attention"),
        ("Unassigned", sum(clean_missing(a.owner) == "Not specified" for a in actions), "Owner not confirmed"),
        ("Processing time", f"{st.session_state.elapsed:.1f}s", "End-to-end"),
    ]
    for col, (label, value, hint) in zip(cols, metrics):
        with col:
            metric_card(label, value, hint)

    overview_tab, actions_tab, report_tab = st.tabs([
        "✦  Summary & topics", "☑  Action tracker", "▤  Full report & exports"
    ])

    with overview_tab:
        st.markdown('<div class="section-title">Executive summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="surface">' + escape(result.get("summary", "Not available.")) + '</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key discussion topics</div>', unsafe_allow_html=True)
        if topics:
            chips = "".join(f'<span class="topic-chip">{escape(topic)}</span>' for topic in topics)
            st.markdown('<div class="surface">' + chips + '</div>', unsafe_allow_html=True)
        else:
            st.info("No topics identified.")
        st.markdown('<div class="section-title">Priority breakdown</div>', unsafe_allow_html=True)
        pcols = st.columns(3)
        for col, level in zip(pcols, ["High", "Medium", "Low"]):
            with col:
                metric_card(f"{level} priority", counts[level], "Tasks")

    with actions_tab:
        st.markdown('<div class="section-title">Action tracker</div>', unsafe_allow_html=True)
        if not actions:
            st.info("No action items were identified. Priority classification was skipped automatically.")
        else:
            filter_col, owner_col = st.columns(2)
            with filter_col:
                selected_priority = st.selectbox("Priority", ["All", "High", "Medium", "Low"])
            with owner_col:
                owners = sorted({clean_missing(a.owner) for a in actions})
                selected_owner = st.selectbox("Owner", ["All"] + owners)
            visible = [
                (i, a) for i, a in enumerate(actions, 1)
                if (selected_priority == "All" or priorities.get(a.task) == selected_priority)
                and (selected_owner == "All" or clean_missing(a.owner) == selected_owner)
            ]
            st.caption(f"Showing {len(visible)} of {len(actions)} actions")
            for i, action in visible:
                priority = priorities.get(action.task, "Not classified")
                st.markdown(
                    '<div class="action-card">'
                    f'<span style="font-size:12px;color:#818398;font-weight:750">TASK {i:02d}</span>'
                    f'&nbsp;&nbsp;{priority_pill(priority)}'
                    f'<div class="action-title">{escape(action.task)}</div>'
                    f'<div class="action-meta"><b>Owner:</b> {escape(clean_missing(action.owner))}'
                    f' &nbsp;&nbsp;·&nbsp;&nbsp; <b>Deadline / trigger:</b> {escape(clean_missing(action.deadline))}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                with st.expander(f"Transcript evidence · Task {i:02d}"):
                    st.write(action.source_evidence or "No evidence supplied.")

    with report_tab:
        st.markdown('<div class="section-title">Complete meeting report</div>', unsafe_allow_html=True)
        st.markdown(result.get("final_report", "No report generated."))
        st.divider()
        st.markdown('<div class="section-title">Export your analysis</div>', unsafe_allow_html=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "↓ Download Markdown report",
                data=result.get("final_report", ""),
                file_name=f"meeting_analysis_{stamp}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "↓ Download structured JSON",
                data=state_to_json(result, st.session_state.elapsed),
                file_name=f"meeting_analysis_{stamp}.json",
                mime="application/json",
                use_container_width=True,
            )
    st.caption("AI-generated output may contain omissions or incorrect assignments. Verify decisions, owners, and deadlines against the transcript.")
else:
    st.markdown('<div class="section-title">02 / Your results will appear here</div>', unsafe_allow_html=True)
    st.info("Add a transcript and select **Analyze meeting** to generate your meeting intelligence dashboard.")
