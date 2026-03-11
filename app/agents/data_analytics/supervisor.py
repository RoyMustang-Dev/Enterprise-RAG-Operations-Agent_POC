import logging
from typing import Dict, Any, List, TypedDict, Annotated, Optional
import re
import pandas as pd
from pydantic import BaseModel, Field
import os
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from app.agents.data_analytics.pandas_engine import DeterministicPandasEngine
from app.agents.data_analytics.tools.rag_wrapper import enterprise_rag_tool
from app.agents.data_analytics.tools.third_party import google_analytics_4_tool, salesforce_soql_tool, google_sheets_tool, stripe_financial_tool, power_bi_analytics_tool
from app.infra.model_registry import get_phase_model

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Pydantic Output Generation Schema
# -----------------------------------------------------------------------------
class KPI(BaseModel):
    title: str = Field(description="The primary metric name (e.g., 'Total Revenue').")
    value: str = Field(description="The exact quantitative value calculated.")
    trend: str = Field(description="A short descriptive narrative of the trend (e.g. 'Up 15% WoW').")
    direction: str = Field(default="flat", description="Direction of change: up|down|flat.")
    delta_percent: float = Field(default=0.0, description="Percent delta for the trend.")
    period: str = Field(default="", description="Period for the trend (e.g., day_over_day).")


class StatisticalTest(BaseModel):
    test: str = Field(description="Name of the statistical test executed.")
    result: str = Field(description="Human-readable result string.")
    p_value: Optional[float] = Field(default=None, description="p-value if available.")
    significant: Optional[bool] = Field(default=None, description="Significance flag if available.")


class ForecastingSummary(BaseModel):
    model_used: str = Field(default="", description="Forecast model name (xgboost|prophet|linear_regression).")
    horizon_days: int = Field(default=0, description="Forecast horizon in days.")
    confidence_bounds: str = Field(default="", description="Confidence bounds summary if available.")
    forecast_csv_url: str = Field(default="", description="CSV URL from forecast tool output.")


class AnalyticsMetadata(BaseModel):
    rows_processed: int = Field(default=0, description="Number of rows processed.")
    columns_used: List[str] = Field(default_factory=list, description="Columns used in analysis.")
    date_range: str = Field(default="", description="Date range of the dataset.")
    missingness_percent: float = Field(default=0.0, description="Percent missing values in dataset.")
    merge_note: str = Field(default="", description="Notes about dataset merge/concat behavior.")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Data lineage sources and row counts.")

class ExecutiveDashboard(BaseModel):
    summary_paragraph: str = Field(description="A 3-4 sentence comprehensive business executive summary of the document/analysis.")
    kpi_cards: List[KPI] = Field(description="A rigid array of up to 4 core KPIs extracted from the data.")
    risk_alerts: List[Dict[str, Any]] = Field(description="Any critical anomalies in the data, missing dates, dropping trends, etc.")
    inferred_csv_payload: str = Field(
        default="", 
        description="A raw CSV formatted string (with headers separated by commas, entries separated by newlines) containing the explicit data points and underlying metrics deduced during the analysis."
    )
    csv_download_url: str = Field(
        default="",
        description="The physical URL endpoint where the user can click to download the generated CSV file."
    )
    suggested_filename: str = Field(
        default="analytics_export.csv",
        description="A short, URL-safe, lowercase string describing this data natively ending in .csv (e.g., 'marketing_roi_q3.csv')."
    )
    statistical_tests: List[StatisticalTest] = Field(default_factory=list, description="Deterministic statistical tests executed on the dataset.")
    forecasting: ForecastingSummary = Field(default_factory=ForecastingSummary, description="Forecasting summary if used.")
    metadata: AnalyticsMetadata = Field(default_factory=AnalyticsMetadata, description="Dataset metadata and processing stats.")
    segments: List[Dict[str, Any]] = Field(default_factory=list, description="Top-performing segments for cohort analysis.")
    time_windows: Dict[str, Any] = Field(default_factory=dict, description="Time-window summary (last 7/30 days) if available.")
    governance_checks: List[Dict[str, Any]] = Field(default_factory=list, description="Deterministic governance and data quality checks.")
    causal_insights: Dict[str, Any] = Field(default_factory=dict, description="Lightweight causal proxy results if available.")
    scenario_simulation: Dict[str, Any] = Field(default_factory=dict, description="Deterministic what-if scenario simulation results.")
    drift_alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Model drift monitoring alerts based on numeric distribution shifts.")
    xlsx_download_url: str = Field(default="", description="XLSX report download URL.")
    kpi_explanations: Dict[str, str] = Field(default_factory=dict, description="Deterministic explanations for KPI calculations.")
    forecast_explainability: Dict[str, Any] = Field(default_factory=dict, description="Explainability details for forecast/backtest.")

# -----------------------------------------------------------------------------
# 2. LangGraph State and Orchestrator
# -----------------------------------------------------------------------------
class AnalyticsState(TypedDict):
    session_id: str
    query: str
    df_schema: str
    persona: str
    messages: Annotated[list, add_messages]
    dashboard_json: Optional[Dict[str, Any]]
    deterministic_kpis: List[Dict[str, Any]]
    deterministic_risks: List[Dict[str, Any]]
    deterministic_tests: List[Dict[str, Any]]
    deterministic_metadata: Dict[str, Any]
    deterministic_segments: List[Dict[str, Any]]
    deterministic_time_windows: Dict[str, Any]
    deterministic_governance: List[Dict[str, Any]]
    deterministic_causal: Dict[str, Any]
    deterministic_scenario: Dict[str, Any]
    deterministic_summary: str
    deterministic_forecast_meta: Dict[str, Any]
    deterministic_drift: List[Dict[str, Any]]
    deterministic_sources: List[Dict[str, Any]]
    deterministic_kpi_explanations: Dict[str, str]
    deterministic_forecast_explain: Dict[str, Any]
    rewritten_query: str
    intent_payload: Dict[str, Any]
    early_exit: bool
    early_reason: str

class DataAnalyticsSupervisor:
    """
    The LangGraph orchestrator specifically built for the Business Analyst Persona.
    It takes an array of CSV files, generates schema context, and deterministically computes mathematical insights.
    """
    def __init__(self, dataframes: List[pd.DataFrame], sources: Optional[List[Dict[str, Any]]] = None):
        # Merge multiple datasets when schema overlaps; otherwise fallback to first
        self.merge_note = ""
        self.sources = sources or []
        self.samples = dataframes
        self.primary_df = self._merge_dataframes(dataframes)
        from app.agents.data_analytics.tools.deterministic_kpi import auto_date_cutoff
        self.primary_df = auto_date_cutoff(self.primary_df)
        self.pandas_sandbox = DeterministicPandasEngine(self.primary_df)
        
        cfg = get_phase_model("hallucination_verifier")
        
        # We hook LangChain's Groq wrapper to use the most powerful LLaMA model
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY", "dummy"),
            temperature=0.0,
            max_retries=1,
            timeout=30
        )
        
        # Bind the specific Master Toolkit
        # NOTE: LangChain requires tools to be bound for parallel execution capabilities
        # Map Pandas code execution as a direct callable Tool for the LLM
        pandas_tool = StructuredTool.from_function(
            func=self.pandas_sandbox.execute_deterministic_math,
            name="execute_pandas_math",
            description="Use explicit Python Pandas code as string to perform math on the 'df' DataFrame. ALWAYS use aggregations (like .groupby(), .sum(), .head()) to extract insights. NEVER try to print the entire dataframe."
        )
        
        # We define dynamic memory-bound forecasting wrappers so the LLM doesn't have to stringify 10,000 rows.
        # The LLM evaluates a dataframe variable inside the sandbox, and passes that variable's EXACT string name here.
        def _prophet_wrapper(variable_name: str, periods: int = 30) -> str:
            from app.agents.data_analytics.tools.predictive_tools import time_series_forecast_prophet
            try:
                target_df = self.pandas_sandbox.repl.locals.get(variable_name)
                if target_df is None or not isinstance(target_df, pd.DataFrame):
                    return f"Fatal Error: The variable '{variable_name}' does not exist as a Pandas DataFrame in the active sandbox. Please use execute_pandas_math to create it first."
                # Pass the physical Pandas DF down to the backend tool.
                # DO NOT run .to_dict() inside this LangChain StructuredTool wrapper. 
                # Doing so forces Langfuse and the ReAct Message History to log the massive 2MB array as tool args!
                return time_series_forecast_prophet(historical_data=target_df, periods=periods)
            except Exception as e:
                return f"Prophet Pipeline Error: {e}"
                
        prophet_tool = StructuredTool.from_function(
            func=_prophet_wrapper,
            name="time_series_forecast_prophet",
            description="Use for baseline time-series forecasting. First run execute_pandas_math to create a Pandas dataframe variable (e.g. `df_grouped`) containing 'ds' (dates) and 'y' (metrics). Then pass that EXACT variable name as a string (e.g. 'df_grouped') to this tool's `variable_name` argument."
        )
        
        def _xgboost_wrapper(variable_name: str, periods: int = 30) -> str:
            from app.agents.data_analytics.tools.predictive_tools import time_series_forecast_xgboost
            try:
                target_df = self.pandas_sandbox.repl.locals.get(variable_name)
                if target_df is None or not isinstance(target_df, pd.DataFrame):
                    return f"Fatal Error: The variable '{variable_name}' does not exist as a Pandas DataFrame in the active sandbox. Please use execute_pandas_math to create it first."
                # Pass the physical Pandas DF directly down to prevent ToolNode JSON Truncations
                return time_series_forecast_xgboost(historical_data=target_df, periods=periods)
            except Exception as e:
                return f"XGBoost Pipeline Error: {e}"
                
        xgboost_tool = StructuredTool.from_function(
            func=_xgboost_wrapper,
            name="time_series_forecast_xgboost",
            description="Use exclusively for advanced multi-variable (Panel Data) forecasting (e.g. predicting across multiple regions and categories simultaneously without collapsing them into a flat line). First run execute_pandas_math to create a Pandas dataframe variable containing 'ds' (dates), 'y' (metrics), and 'unique_id' (string labels). Then pass that EXACT variable name as a string (e.g. 'df_project') to this tool's `variable_name` argument."
        )
        
        def _regression_wrapper(historical_variable_name: str, target_column: str, future_feature_values: List[Dict[str, Any]]) -> str:
            from app.agents.data_analytics.tools.predictive_tools import linear_regression_projection
            try:
                target_df = self.pandas_sandbox.repl.locals.get(historical_variable_name)
                if target_df is None or not isinstance(target_df, pd.DataFrame):
                    return f"Fatal Error: Variable '{historical_variable_name}' does not exist."
                # Pass directly down to avoid ToolNode tracing
                result = linear_regression_projection(features_data=target_df, target_column=target_column, future_feature_values=future_feature_values)
                return result
            except Exception as e:
                return f"Regression Pipeline Error: {e}"
                
        regression_tool = StructuredTool.from_function(
            func=_regression_wrapper,
            name="linear_regression_projection",
            description="Use for multi-variate continuous prediction. Run execute_pandas_math to create your feature dataframe first, then pass its variable name here."
        )
        
        self.tools = [
            enterprise_rag_tool, 
            prophet_tool, 
            xgboost_tool,
            regression_tool, 
            pandas_tool,
            google_analytics_4_tool,
            salesforce_soql_tool,
            google_sheets_tool,
            power_bi_analytics_tool,
            stripe_financial_tool
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Pydantic Structural generator
        self.llm_structured = self.llm.with_structured_output(ExecutiveDashboard)
        
        self.graph = self._build_graph()

    def _merge_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        if not dataframes:
            return pd.DataFrame()
        if len(dataframes) == 1:
            return dataframes[0]
        try:
            # Only concat if schemas overlap significantly
            base_cols = set(dataframes[0].columns)
            compatible = [df for df in dataframes if len(base_cols.intersection(df.columns)) >= max(3, int(0.5 * len(base_cols)))]
            if len(compatible) >= 2:
                self.merge_note = f"Concatenated {len(compatible)} datasets with overlapping schema."
                return pd.concat(compatible, ignore_index=True)
            self.merge_note = "Multiple datasets provided, but schemas diverged; using the first dataset."
            return dataframes[0]
        except Exception:
            self.merge_note = "Dataset merge failed; using the first dataset."
            return dataframes[0]

    def _build_graph(self) -> Any:
        workflow = StateGraph(AnalyticsState)
        
        workflow.add_node("smalltalk_gate", self.smalltalk_gate)
        workflow.add_node("prompt_guard", self.prompt_guard)
        workflow.add_node("prompt_rewriter", self.prompt_rewriter)
        workflow.add_node("detect_schema", self.detect_schema)
        workflow.add_node("reasoning_agent", self.reasoning_agent)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("generate_dashboard", self.generate_dashboard)
        workflow.add_node("forecast_meta", self.forecast_meta)
        workflow.add_node("generate_early_exit", self.generate_early_exit)
        
        workflow.add_edge(START, "smalltalk_gate")
        workflow.add_conditional_edges(
            "smalltalk_gate",
            lambda state: "early" if state.get("early_exit") else "continue",
            {"early": "generate_early_exit", "continue": "prompt_guard"}
        )
        workflow.add_conditional_edges(
            "prompt_guard",
            lambda state: "early" if state.get("early_exit") else "continue",
            {"early": "generate_early_exit", "continue": "prompt_rewriter"}
        )
        workflow.add_edge("prompt_rewriter", "detect_schema")
        workflow.add_edge("detect_schema", "reasoning_agent")
        
        # The AI loops: Reason -> Tool -> Reason -> Tool until it stops calling tools
        workflow.add_conditional_edges(
            "reasoning_agent",
            tools_condition,
            {"tools": "tools", "__end__": "generate_dashboard"}
        )
        workflow.add_edge("tools", "forecast_meta")
        workflow.add_edge("forecast_meta", "reasoning_agent")
        workflow.add_edge("generate_dashboard", END)
        workflow.add_edge("generate_early_exit", END)
        
        return workflow.compile()

    def forecast_meta(self, state: AnalyticsState) -> Dict:
        """
        Extract deterministic forecast metadata from tool outputs.
        Looks for forecast tool success strings that include model name and csv path.
        """
        meta = {}
        explain = {}
        try:
            # find last tool message
            last_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, ToolMessage):
                    last_msg = msg
                    break
            if last_msg and isinstance(last_msg.content, str):
                content = last_msg.content.lower()
                if "xgboost" in content:
                    meta["model_used"] = "xgboost"
                elif "prophet" in content:
                    meta["model_used"] = "prophet"

                # extract csv path
                match = re.search(r"/api/v1/exports/[^\s']+", last_msg.content)
                if match:
                    meta["forecast_csv_url"] = match.group(0)

                # extract horizon days if mentioned
                horizon_match = re.search(r"Projecting\s+(\d+)\s+days", last_msg.content, re.IGNORECASE)
                if horizon_match:
                    meta["horizon_days"] = int(horizon_match.group(1))

                ci_match = re.search(r"CI=±([0-9\.]+)", last_msg.content)
                if ci_match:
                    meta["confidence_bounds"] = f"±{ci_match.group(1)}"

                mae_match = re.search(r"Backtest MAE=([0-9\.]+)", last_msg.content)
                if mae_match:
                    explain["mae"] = float(mae_match.group(1))
        except Exception:
            pass
        return {"deterministic_forecast_meta": meta, "deterministic_forecast_explain": explain}

    def _is_smalltalk(self, text: str) -> bool:
        greetings = [
            r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bhow are you\b", r"\bgood morning\b",
            r"\bgood evening\b", r"\bthanks\b", r"\bthank you\b", r"\bwho are you\b"
        ]
        for pattern in greetings:
            if re.search(pattern, text.lower()):
                return True
        return False

    def _guard_violations(self, text: str) -> Optional[str]:
        # Deterministic guardrails: block prompt injection / system override attempts
        disallowed = [
            "ignore previous instructions",
            "system prompt",
            "reveal your prompt",
            "bypass safety",
            "developer message",
            "api key",
            "password"
        ]
        lowered = text.lower()
        for term in disallowed:
            if term in lowered:
                return f"Unsafe request detected: {term}"
        return None

    def _rewrite_intent(self, text: str) -> Dict[str, Any]:
        # Deterministic intent parsing (no LLM)
        intent = {
            "intent": "analysis",
            "metrics": [],
            "filters": {},
            "time_horizon_days": None
        }

        for metric in ["revenue", "units_sold", "marketing_spend", "profit", "margin"]:
            if metric.replace("_", " ") in text.lower() or metric in text.lower():
                intent["metrics"].append(metric)

        # Simple horizon extraction (e.g. "next 30 days")
        match = re.search(r"next\s+(\d+)\s+days", text.lower())
        if match:
            intent["time_horizon_days"] = int(match.group(1))

        # Region / product keyword extraction
        if "region" in text.lower():
            intent["filters"]["group_by"] = intent["filters"].get("group_by", []) + ["Region"]
        if "product" in text.lower():
            intent["filters"]["group_by"] = intent["filters"].get("group_by", []) + ["Product_Category"]

        return intent

    def smalltalk_gate(self, state: AnalyticsState) -> Dict:
        query = state["query"]
        if self._is_smalltalk(query):
            return {
                "early_exit": True,
                "early_reason": "smalltalk",
                "rewritten_query": query,
                "intent_payload": {}
            }
        return {"early_exit": False, "rewritten_query": query, "intent_payload": {}}

    def prompt_guard(self, state: AnalyticsState) -> Dict:
        violation = self._guard_violations(state["query"])
        if violation:
            return {
                "early_exit": True,
                "early_reason": violation
            }
        return {"early_exit": False}

    def prompt_rewriter(self, state: AnalyticsState) -> Dict:
        intent = self._rewrite_intent(state["query"])
        rewritten = state["query"]
        if intent["metrics"]:
            rewritten = f"{state['query']}\n\n[INTENT] metrics={intent['metrics']} filters={intent.get('filters', {})} horizon={intent.get('time_horizon_days')}"
        return {
            "rewritten_query": rewritten,
            "intent_payload": intent,
            "messages": [HumanMessage(content=rewritten)]
        }

    def detect_schema(self, state: AnalyticsState) -> Dict:
        """Runs the deterministic df.describe() sandbox to extract a lightweight token map."""
        if self.primary_df.empty:
            return {
                "df_schema": "NO_DATASET_PROVIDED",
                "deterministic_kpis": [],
                "deterministic_risks": [],
                "deterministic_tests": [],
                "deterministic_segments": [],
                "deterministic_time_windows": {},
                "deterministic_governance": [],
                "deterministic_causal": {},
                "deterministic_scenario": {},
                "deterministic_metadata": {
                    "rows_processed": 0,
                    "columns_used": [],
                    "date_range": "",
                    "missingness_percent": 0.0,
                    "merge_note": self.merge_note
                }
            }
            
        logger.info("[ANALYTICS] Auto-detecting Schema from dataset...")
        schema_map = self.pandas_sandbox.generate_schema_context()
        from app.agents.data_analytics.tools.deterministic_kpi import (
            compute_basic_kpis,
            compute_risk_alerts,
            compute_statistical_tests,
            compute_auto_stat_tests,
            compute_metadata,
            compute_segments,
            compute_time_windows,
            compute_governance_checks,
            compute_causal_proxy,
            compute_scenario_simulation,
            compute_deterministic_summary,
            compute_drift_alerts,
            compute_kpi_explanations,
        )
        kpis = compute_basic_kpis(self.primary_df)
        return {
            "df_schema": schema_map,
            "deterministic_kpis": kpis,
            "deterministic_risks": compute_risk_alerts(self.primary_df),
            "deterministic_tests": compute_auto_stat_tests(self.primary_df) or compute_statistical_tests(self.primary_df),
            "deterministic_metadata": compute_metadata(self.primary_df, merge_note=self.merge_note),
            "deterministic_segments": compute_segments(self.primary_df),
            "deterministic_time_windows": compute_time_windows(self.primary_df),
            "deterministic_governance": compute_governance_checks(self.primary_df),
            "deterministic_causal": compute_causal_proxy(self.primary_df),
            "deterministic_scenario": compute_scenario_simulation(self.primary_df),
            "deterministic_summary": compute_deterministic_summary(self.primary_df),
            "deterministic_forecast_meta": {},
            "deterministic_drift": compute_drift_alerts(self.primary_df),
            "deterministic_kpi_explanations": compute_kpi_explanations(kpis),
            "deterministic_forecast_explain": {}
        }

    def reasoning_agent(self, state: AnalyticsState) -> Dict:
        """The core ReAct loop that decides whether to run Python Math, call RAG, or Forecast."""
        system_prompt = f"""You are a senior Business Analyst computing calculations perfectly. 
TARGET GOAL: {state.get('rewritten_query') or state['query']}
PERSONA OVERLAY: {state['persona']}

CRITICAL RULES:
1. ALWAYS use 'execute_pandas_math' to perform math on the 'df' DataFrame. 
2. Write plain python syntax. NO GUESSING.
3. If you need to forecast, run 'execute_pandas_math' FIRST and assign the resulting dataframe to a variable like 'df_project'. Do NOT print massive dicts. 
4. STRICT CHAIN OF THOUGHT: NEVER nest a tool call. You MUST call 'execute_pandas_math' first to calculate your table, wait for the observation, and ONLY THEN pass the exact string variable name of your table into the forecasting tools.
5. IF using XGBoost for multivariate panel forecasting (e.g. multiple regions), you MUST construct a string column named `unique_id` in your dataframe first (e.g., `df_project['unique_id'] = df_project['Region'] + '_' + df_project['Product_Category']`).
6. Do not stop until all KPI numbers are 100% computed.

SCHEMA MAP REVEALED:
{state['df_schema']}
"""
        messages = [SystemMessage(content=system_prompt)] + state['messages']
        
        logger.info("[ANALYTICS] Triggering Tool-Calling Analytical Reasoner.")
        response = self.llm_with_tools.invoke(messages)
        
        # LangGraph automatically intercepts this AIMessage. 
        # If it has tool_calls, tools_condition routes it to the ToolNode. 
        # If not, it routes to dashboard.
        return {"messages": [response]}

    def save_csv_and_generate_url(self, raw_csv_string: str, suggested_name: str, session_id: str) -> str:
        """Physically saves the inferred CSV to disk and returns a download route."""
        try:
            export_dir = os.path.join(os.getcwd(), "data", "exports")
            os.makedirs(export_dir, exist_ok=True)
            
            # Use the LLM's suggested name, but prefix with a short session hash to prevent collisions
            safe_id = str(session_id)[:8]
            clean_name = suggested_name.replace(" ", "_").lower()
            if not clean_name.endswith('.csv'):
                clean_name += '.csv'
                
            file_name = f"{safe_id}_{clean_name}"
            file_path = os.path.join(export_dir, file_name)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(raw_csv_string)
                
            return f"/api/v1/exports/{file_name}"
        except Exception as e:
            logger.error(f"[CSV EXPORT] Failed to write file: {e}")
            return ""

    def save_xlsx_report(self, session_id: str, dashboard: ExecutiveDashboard, samples: List[pd.DataFrame]) -> str:
        """Generate an XLSX report with metadata, samples, insights, and charts."""
        try:
            import xlsxwriter

            export_dir = os.path.join(os.getcwd(), "data", "exports")
            os.makedirs(export_dir, exist_ok=True)
            file_name = f"{str(session_id)[:8]}_analytics_report.xlsx"
            file_path = os.path.join(export_dir, file_name)

            workbook = xlsxwriter.Workbook(file_path)
            header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9E1F2"})

            # Sheet 1: Metadata
            ws_meta = workbook.add_worksheet("Metadata")
            ws_meta.write(0, 0, "Field", header_fmt)
            ws_meta.write(0, 1, "Value", header_fmt)
            meta = dashboard.metadata.model_dump() if isinstance(dashboard.metadata, AnalyticsMetadata) else {}
            row = 1
            for k, v in meta.items():
                ws_meta.write(row, 0, k)
                ws_meta.write(row, 1, str(v))
                row += 1

            # Sheet 2: Raw Samples
            from app.agents.data_analytics.tools.deterministic_kpi import redact_pii
            ws_samples = workbook.add_worksheet("Raw Samples")
            row = 0
            for i, df in enumerate(samples):
                df = redact_pii(df)
                ws_samples.write(row, 0, f"Dataset {i+1}", header_fmt)
                row += 1
                if not df.empty:
                    for col_idx, col in enumerate(df.columns):
                        ws_samples.write(row, col_idx, col, header_fmt)
                    row += 1
                    for _, r in df.head(50).iterrows():
                        for col_idx, col in enumerate(df.columns):
                            ws_samples.write(row, col_idx, str(r[col]))
                        row += 1
                row += 2

            # Sheet 3: Insights
            ws_insights = workbook.add_worksheet("Insights")
            ws_insights.write(0, 0, "KPI", header_fmt)
            ws_insights.write(0, 1, "Value", header_fmt)
            for idx, kpi in enumerate(dashboard.kpi_cards):
                ws_insights.write(idx + 1, 0, kpi.title)
                ws_insights.write(idx + 1, 1, kpi.value)

            # Sheet 4: Charts
            ws_charts = workbook.add_worksheet("Charts")
            if dashboard.kpi_cards:
                ws_charts.write(0, 0, "KPI", header_fmt)
                ws_charts.write(0, 1, "Value", header_fmt)
                for idx, kpi in enumerate(dashboard.kpi_cards):
                    ws_charts.write(idx + 1, 0, kpi.title)
                    try:
                        ws_charts.write_number(idx + 1, 1, float(str(kpi.value).replace(",", "")))
                    except Exception:
                        ws_charts.write(idx + 1, 1, 0)

                chart = workbook.add_chart({"type": "column"})
                chart.add_series({
                    "categories": ["Charts", 1, 0, len(dashboard.kpi_cards), 0],
                    "values": ["Charts", 1, 1, len(dashboard.kpi_cards), 1],
                    "name": "KPI Overview"
                })
                chart.set_title({"name": "KPI Overview"})
                ws_charts.insert_chart("D2", chart)

            workbook.close()
            return f"/api/v1/exports/{file_name}"
        except Exception:
            return ""

    def generate_dashboard(self, state: AnalyticsState) -> Dict:
        """Forces the raw text from the Agent into a strict Pydantic UI Dashboard mapping."""
        logger.info("[ANALYTICS] Synthesizing final structured dashboard payload.")
        
        synthesis_prompt = f"""Based on the current chat execution and analysis, extract the exact KPIs and business summaries.
Do not hallucinate data. Map the results into the output structured format securely.

CRITICALLY: 
If the Analysis Results explicitly gave you a CSV URL (e.g. `/api/v1/exports/...`), YOU MUST place that exact URL into the `csv_download_url` field, and leave `inferred_csv_payload` BLANK. Do not invent data!
If the analysis did NOT provide a file URL, you must construct the aggregated data into `inferred_csv_payload` (max 20 rows) and provide a `suggested_filename`.

DO NOT HALLUCINATE DATES OR METRICS. Rely entirely on the Analysis Results context!

RAW DATA SCHEMA:
{state['df_schema']}

ANALYSIS RESULTS:
{state['messages'][-1].content}
"""
        try:
            dashboard = self.llm_structured.invoke([
                SystemMessage(content="You are a strict JSON mapping architect."),
                HumanMessage(content=synthesis_prompt)
            ])

            # Inject deterministic results (authoritative)
            if not dashboard.kpi_cards and state.get("deterministic_kpis"):
                dashboard.kpi_cards = [KPI(**k) for k in state["deterministic_kpis"]]

            if not dashboard.risk_alerts and state.get("deterministic_risks"):
                dashboard.risk_alerts = state["deterministic_risks"]

            if state.get("deterministic_tests"):
                dashboard.statistical_tests = [StatisticalTest(**t) for t in state["deterministic_tests"]]

            if state.get("deterministic_metadata"):
                dashboard.metadata = AnalyticsMetadata(**state["deterministic_metadata"])
                if state.get("deterministic_sources"):
                    dashboard.metadata.sources = state["deterministic_sources"]

            if state.get("deterministic_segments"):
                dashboard.segments = state["deterministic_segments"]

            if state.get("deterministic_time_windows"):
                dashboard.time_windows = state["deterministic_time_windows"]

            if state.get("deterministic_governance"):
                dashboard.governance_checks = state["deterministic_governance"]

            if state.get("deterministic_causal"):
                dashboard.causal_insights = state["deterministic_causal"]

            if state.get("deterministic_scenario"):
                dashboard.scenario_simulation = state["deterministic_scenario"]

            if state.get("deterministic_drift"):
                dashboard.drift_alerts = state["deterministic_drift"]

            if state.get("deterministic_kpi_explanations"):
                dashboard.kpi_explanations = state["deterministic_kpi_explanations"]

            if state.get("deterministic_forecast_explain"):
                dashboard.forecast_explainability = state["deterministic_forecast_explain"]

            if state.get("deterministic_summary"):
                dashboard.summary_paragraph = state["deterministic_summary"]

            if state.get("deterministic_forecast_meta"):
                try:
                    for k, v in state["deterministic_forecast_meta"].items():
                        if hasattr(dashboard.forecasting, k):
                            setattr(dashboard.forecasting, k, v)
                except Exception:
                    pass

            if dashboard.csv_download_url and dashboard.csv_download_url.startswith("/api/v1/exports/"):
                # The LLM captured the physical URL from a forensic/predictive tool. Do not overwrite!
                pass
            elif dashboard.inferred_csv_payload:
                dashboard.csv_download_url = self.save_csv_and_generate_url(
                    raw_csv_string=dashboard.inferred_csv_payload,
                    suggested_name=dashboard.suggested_filename,
                    session_id=state["session_id"]
                )

            # XLSX export
            xlsx_url = self.save_xlsx_report(state["session_id"], dashboard, self.samples)
            if xlsx_url:
                dashboard.xlsx_download_url = xlsx_url
            
            return {"dashboard_json": dashboard.model_dump()}
        except Exception as e:
            logger.error(f"[ANALYTICS] Dashboard Pydantic Format Failure: {e}")
            # Deterministic fallback without LLM schema
            fallback = ExecutiveDashboard(
                summary_paragraph=state.get("deterministic_summary") or "Analytics fallback executed due to schema error.",
                kpi_cards=[KPI(**k) for k in state.get("deterministic_kpis", [])][:4],
                risk_alerts=state.get("deterministic_risks", []),
                inferred_csv_payload="",
                csv_download_url=state.get("deterministic_forecast_meta", {}).get("forecast_csv_url", ""),
                suggested_filename="analytics_export.csv",
                statistical_tests=[StatisticalTest(**t) for t in state.get("deterministic_tests", [])],
                forecasting=ForecastingSummary(**state.get("deterministic_forecast_meta", {})) if state.get("deterministic_forecast_meta") else ForecastingSummary(),
                metadata=AnalyticsMetadata(**state.get("deterministic_metadata", {})),
                segments=state.get("deterministic_segments", []),
                time_windows=state.get("deterministic_time_windows", {}),
                governance_checks=state.get("deterministic_governance", []),
                causal_insights=state.get("deterministic_causal", {}),
                scenario_simulation=state.get("deterministic_scenario", {}),
                drift_alerts=state.get("deterministic_drift", []),
            )
            if state.get("deterministic_sources"):
                fallback.metadata.sources = state["deterministic_sources"]
            if state.get("deterministic_kpi_explanations"):
                fallback.kpi_explanations = state["deterministic_kpi_explanations"]
            if state.get("deterministic_forecast_explain"):
                fallback.forecast_explainability = state["deterministic_forecast_explain"]
            xlsx_url = self.save_xlsx_report(state["session_id"], fallback, self.samples)
            if xlsx_url:
                fallback.xlsx_download_url = xlsx_url
            return {"dashboard_json": fallback.model_dump()}

    def generate_early_exit(self, state: AnalyticsState) -> Dict:
        """Creates a deterministic response for smalltalk or blocked requests."""
        reason = state.get("early_reason") or "smalltalk"
        if reason == "smalltalk":
            summary = "Hello! I'm your Business Analytics Agent. Upload a CSV or Excel file and tell me the analysis you want."
        else:
            summary = f"Request blocked by Prompt Guard: {reason}"

        fallback = ExecutiveDashboard(
            summary_paragraph=summary,
            kpi_cards=[],
            risk_alerts=[],
            inferred_csv_payload="",
            csv_download_url="",
            suggested_filename="analytics_export.csv",
            statistical_tests=[],
            forecasting=ForecastingSummary(),
            metadata=AnalyticsMetadata()
        )
        return {"dashboard_json": fallback.model_dump()}

    async def run(self, query: str, persona: str, session_id: str) -> Dict[str, Any]:
        """Entrypoint for the API Route execution."""
        from app.infra.database import init_analytics_memory_db, fetch_analytics_memory, insert_analytics_memory
        init_analytics_memory_db()
        # Load prior memory for multi-turn context
        memory_rows = fetch_analytics_memory(session_id=session_id, limit=6)
        memory_messages = []
        for role, content, kpi_json in memory_rows:
            if role == "user":
                memory_messages.append(HumanMessage(content=content))
            else:
                memory_messages.append(AIMessage(content=content))
        initial_state = {
            "session_id": session_id,
            "query": query,
            "df_schema": "",
            "persona": persona,
            "messages": memory_messages + [HumanMessage(content=query)],
            "dashboard_json": None,
            "deterministic_kpis": [],
            "deterministic_risks": [],
            "deterministic_tests": [],
            "deterministic_metadata": {},
            "deterministic_segments": [],
            "deterministic_time_windows": {},
            "deterministic_governance": [],
            "deterministic_causal": {},
            "deterministic_scenario": {},
            "deterministic_summary": "",
            "deterministic_forecast_meta": {},
            "deterministic_drift": [],
            "deterministic_sources": self.sources,
            "deterministic_kpi_explanations": {},
            "deterministic_forecast_explain": {},
            "rewritten_query": "",
            "intent_payload": {},
            "early_exit": False,
            "early_reason": ""
        }
        
        config = {}
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse.callback import CallbackHandler
                langfuse_handler = CallbackHandler(
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    host=os.getenv("LANGFUSE_HOST", "https://us.langfuse.com"),
                    tags=["BUSINESS_ANALYST"]
                )
                config["callbacks"] = [langfuse_handler]
                logger.info(f"[LANGFUSE] Tracing enabled for Business Analyst session: {session_id}")
            except Exception as e:
                logger.warning(f"[LANGFUSE] Failed to initialize Analytics callback: {e}")

        try:
            import asyncio
            # Hard 120-second timeout on the entire analytical reasoning pipeline
            final_state = await asyncio.wait_for(
                self.graph.ainvoke(initial_state, config=config), 
                timeout=120.0
            )
            # Persist memory
            insert_analytics_memory(session_id, "user", query, "")
            if final_state.get("dashboard_json"):
                insert_analytics_memory(session_id, "assistant", json.dumps(final_state["dashboard_json"])[:5000], json.dumps(final_state["dashboard_json"].get("kpi_cards", []))[:2000])
            return final_state["dashboard_json"]
            
        except asyncio.TimeoutError:
            logger.error(f"[ANALYTICS] Supervisor timed out after 120 seconds. Session: {session_id}")
            fallback = ExecutiveDashboard(
                summary_paragraph="Analytics Agent Engine Timeout. The data logic or LLM inference loops exceeded the standard 2-minute API limit.",
                kpi_cards=[KPI(title="System Status", value="TIMEOUT", trend="Execution Halted")],
                risk_alerts=[{"type": "timeout", "message": "The dataset complexity caused a timeout during reasoning.", "severity": "high"}],
                inferred_csv_payload="Error\nSystem Timeout",
                csv_download_url="",
                suggested_filename="timeout_error.csv"
            )
            return fallback.model_dump()
            
        except Exception as e:
            logger.error(f"[ANALYTICS] Fatal crash during Graph execution. Session: {session_id} Error: {e}")
            fallback = ExecutiveDashboard(
                summary_paragraph="Analytics System Breakdown. A fatal exception crashed the reasoning supervisor.",
                kpi_cards=[KPI(title="System Status", value="ERROR", trend="Exception Caught")],
                risk_alerts=[{"type": "system_error", "message": f"Pipeline fault: {str(e)[:150]}", "severity": "high"}],
                inferred_csv_payload="Error\nSystem Exception",
                csv_download_url="",
                suggested_filename="crash_error.csv"
            )
            return fallback.model_dump()
