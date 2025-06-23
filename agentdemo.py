#!/usr/bin/env python3
"""
Educational Insurance Case Evaluation with Anthropic MCP Protocol
================================================================

An interactive Streamlit application demonstrating different levels of AI agents
for insurance case evaluation using the MCP (Model Context Protocol).

Requirements:
    pip install streamlit anthropic mcp typing-extensions pydantic plotly

To run:
    streamlit run insurance_mcp_education.py

Environment:
    export ANTHROPIC_API_KEY="your-api-key"
"""

import os
import json
import asyncio
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
import time
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

# Page config
st.set_page_config(
    page_title="AI Agent Levels: Insurance MCP Demo",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []
if 'agent_conversations' not in st.session_state:
    st.session_state.agent_conversations = {}
if 'tool_calls' not in st.session_state:
    st.session_state.tool_calls = []
if 'tutorial_step' not in st.session_state:
    st.session_state.tutorial_step = 0
if 'live_thoughts' not in st.session_state:
    st.session_state.live_thoughts = []
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = {}

# Try to get API key from environment first, then from Streamlit secrets
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# If not in environment, try Streamlit secrets (but don't error if secrets file doesn't exist)
if not ANTHROPIC_API_KEY:
    try:
        ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY")
    except FileNotFoundError:
        pass

if not ANTHROPIC_API_KEY:
    st.error("⚠️ Please set ANTHROPIC_API_KEY environment variable or add it to Streamlit secrets")
    st.info("""
    **Option 1: Environment Variable**
    ```bash
    export ANTHROPIC_API_KEY="your-api-key"
    streamlit run insurance_mcp_education.py
    ```
    
    **Option 2: Streamlit Secrets**
    Create `.streamlit/secrets.toml` in your project directory:
    ```toml
    ANTHROPIC_API_KEY = "your-api-key"
    ```
    """)
    st.stop()

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ============================================================================
# Scenario Presets (Improvement #4)
# ============================================================================

SCENARIO_PRESETS = {
    "Suspicious Fraud Pattern": {
        "claim_type": "auto",
        "description": "Vehicle suddenly caught fire immediately after purchasing comprehensive coverage",
        "amount": 45000,
        "documents": ["photos"],
        "risk_indicators": ["suspicious_timing", "total_loss", "recent_policy_change"],
        "previous_claims": 3
    },
    "Legitimate High-Value": {
        "claim_type": "property",
        "description": "Tree fell on house during documented storm",
        "amount": 35000,
        "documents": ["photos", "police_report", "repair_estimate"],
        "risk_indicators": [],
        "previous_claims": 0
    },
    "Edge Case": {
        "claim_type": "auto",
        "description": "Minor fender bender with conflicting witness statements",
        "amount": 8000,
        "documents": ["police_report", "photos"],
        "risk_indicators": ["conflicting_reports"],
        "previous_claims": 1
    },
    "Medical Emergency": {
        "claim_type": "health",
        "description": "Emergency surgery following accident, all procedures documented",
        "amount": 25000,
        "documents": ["medical_records", "bills", "prescription"],
        "risk_indicators": [],
        "previous_claims": 0
    },
    "Repeat Claimant": {
        "claim_type": "auto",
        "description": "Third claim this year, similar circumstances each time",
        "amount": 12000,
        "documents": ["police_report", "photos", "repair_estimate"],
        "risk_indicators": ["multiple_claims", "pattern_behavior"],
        "previous_claims": 5
    }
}


# ============================================================================
# Insurance Domain Models (Same as before)
# ============================================================================

class ClaimType(Enum):
    AUTO = "auto"
    PROPERTY = "property"
    HEALTH = "health"
    LIABILITY = "liability"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InsuranceClaim:
    """Insurance claim data"""
    claim_id: str
    claim_type: ClaimType
    amount: float
    date_filed: datetime
    description: str
    policy_number: str
    claimant_history: List[Dict[str, Any]] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)
    risk_indicators: List[str] = field(default_factory=list)
    status: str = "pending"


@dataclass
class EvaluationResult:
    """Evaluation result from an agent"""
    claim_id: str
    recommendation: str  # "approve", "deny", "investigate"
    confidence: float
    risk_level: RiskLevel
    reasons: List[str]
    suggested_payout: Optional[float] = None
    red_flags: List[str] = field(default_factory=list)
    evaluator: str = "unknown"
    processing_time: float = 0.0
    tool_usage: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Enhanced MCP Tools with Logging
# ============================================================================

class InsuranceMCPTools:
    """MCP-compatible tools for insurance evaluation with educational logging"""
    
    @staticmethod
    def get_tools() -> List[Tool]:
        """Return MCP tool definitions"""
        return [
            Tool(
                name="check_policy_coverage",
                description="Check if a claim type is covered by the policy",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "policy_number": {"type": "string"},
                        "claim_type": {"type": "string"}
                    },
                    "required": ["policy_number", "claim_type"]
                }
            ),
            Tool(
                name="calculate_risk_score",
                description="Calculate risk score for a claim based on multiple factors",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_amount": {"type": "number"},
                        "claim_history_count": {"type": "integer"},
                        "days_since_incident": {"type": "integer"},
                        "risk_indicators": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["claim_amount"]
                }
            ),
            Tool(
                name="verify_documents",
                description="Verify if all required documents are present",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_type": {"type": "string"},
                        "provided_documents": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["claim_type", "provided_documents"]
                }
            ),
            Tool(
                name="check_fraud_patterns",
                description="Check for common fraud patterns in the claim",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_description": {"type": "string"},
                        "claim_amount": {"type": "number"},
                        "claimant_history": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "date": {"type": "string"},
                                    "amount": {"type": "number"}
                                }
                            }
                        }
                    },
                    "required": ["claim_description", "claim_amount"]
                }
            ),
            Tool(
                name="investigate_claim",
                description="Perform detailed investigation of a claim",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "investigation_type": {
                            "type": "string",
                            "enum": ["social_media", "witnesses", "expert_assessment", "site_inspection"]
                        }
                    },
                    "required": ["claim_id", "investigation_type"]
                }
            )
        ]
    
    @staticmethod
    def get_anthropic_tools() -> List[Dict[str, Any]]:
        """Convert MCP tools to Anthropic format"""
        mcp_tools = InsuranceMCPTools.get_tools()
        anthropic_tools = []
        
        for tool in mcp_tools:
            anthropic_tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            })
        
        return anthropic_tools
    
    @staticmethod
    async def execute_tool(name: str, arguments: Dict[str, Any], log_callback=None, live_log_callback=None) -> Dict[str, Any]:
        """Execute a tool and return results with logging"""
        
        # Log tool execution
        if log_callback:
            log_callback({
                "timestamp": datetime.now().isoformat(),
                "tool": name,
                "input": arguments,
                "status": "executing"
            })
        
        # Live logging for real-time display
        if live_log_callback:
            live_log_callback('tool', f"Executing {name}", {"tool": name, "input": arguments})
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        result = {}
        
        if name == "check_policy_coverage":
            # Simulate policy coverage check
            result = {
                "covered": True,
                "coverage_limit": 100000,
                "deductible": 1000,
                "exclusions": []
            }
        
        elif name == "calculate_risk_score":
            # Calculate risk based on inputs
            score = 0.0
            factors = []
            
            if arguments.get("claim_amount", 0) > 20000:
                score += 0.3
                factors.append("High claim amount")
            if arguments.get("claim_history_count", 0) > 2:
                score += 0.2
                factors.append("Multiple previous claims")
            if arguments.get("days_since_incident", 30) < 7:
                score += 0.1
                factors.append("Recent incident")
            
            risk_indicators = arguments.get("risk_indicators", [])
            score += len(risk_indicators) * 0.1
            if risk_indicators:
                factors.append(f"{len(risk_indicators)} risk indicators present")
            
            result = {
                "risk_score": min(score, 1.0),
                "risk_factors": factors,
                "calculation_details": {
                    "claim_amount_factor": 0.3 if arguments.get("claim_amount", 0) > 20000 else 0,
                    "history_factor": 0.2 if arguments.get("claim_history_count", 0) > 2 else 0,
                    "recency_factor": 0.1 if arguments.get("days_since_incident", 30) < 7 else 0,
                    "indicators_factor": len(risk_indicators) * 0.1
                }
            }
        
        elif name == "verify_documents":
            # Check document completeness
            required = {
                "auto": ["police_report", "photos", "repair_estimate"],
                "property": ["photos", "repair_estimate", "proof_of_ownership"],
                "health": ["medical_records", "bills", "prescription"],
                "liability": ["incident_report", "witness_statements"]
            }
            
            claim_type = arguments.get("claim_type", "auto")
            provided = arguments.get("provided_documents", [])
            required_docs = required.get(claim_type, [])
            missing = [doc for doc in required_docs if doc not in provided]
            
            result = {
                "complete": len(missing) == 0,
                "missing_documents": missing,
                "verification_status": "verified" if len(missing) == 0 else "incomplete",
                "required_documents": required_docs,
                "provided_documents": provided
            }
        
        elif name == "check_fraud_patterns":
            # Analyze for fraud patterns
            description = arguments.get("claim_description", "").lower()
            suspicious_terms = ["suddenly", "immediately", "total loss", "stolen", "disappeared"]
            found_terms = [term for term in suspicious_terms if term in description]
            
            history = arguments.get("claimant_history", [])
            frequent_claims = len(history) > 3
            
            fraud_score = len(found_terms) * 0.2 + (0.3 if frequent_claims else 0)
            
            result = {
                "fraud_score": min(fraud_score, 1.0),
                "suspicious_patterns": found_terms,
                "frequent_claimant": frequent_claims,
                "recommendation": "investigate" if fraud_score > 0.5 else "proceed",
                "analysis_details": {
                    "suspicious_terms_found": len(found_terms),
                    "total_previous_claims": len(history),
                    "fraud_score_calculation": f"{len(found_terms)} * 0.2 + {'0.3' if frequent_claims else '0'} = {fraud_score}"
                }
            }
        
        elif name == "investigate_claim":
            # Simulate investigation
            investigation_results = {
                "social_media": {
                    "contradictions_found": False,
                    "relevant_posts": 0,
                    "summary": "No contradictory social media activity found"
                },
                "witnesses": {
                    "witnesses_found": 2,
                    "statements_consistent": True,
                    "summary": "2 witnesses corroborate the claim"
                },
                "expert_assessment": {
                    "assessment_complete": True,
                    "findings": "Damage consistent with claimed incident",
                    "expert_confidence": 0.85
                },
                "site_inspection": {
                    "inspection_complete": True,
                    "evidence_found": True,
                    "physical_evidence": ["Skid marks", "Vehicle damage patterns"]
                }
            }
            
            result = investigation_results.get(
                arguments.get("investigation_type", "witnesses"),
                {"error": "Unknown investigation type"}
            )
        
        # Log tool result
        if log_callback:
            log_callback({
                "timestamp": datetime.now().isoformat(),
                "tool": name,
                "output": result,
                "status": "completed"
            })
        
        # Live logging for result
        if live_log_callback:
            live_log_callback('result', f"Tool {name} completed", result)
        
        return result


# ============================================================================
# Enhanced Agent Classes with Live Logging (Improvement #2)
# ============================================================================

class ReactiveInsuranceAgent:
    """
    Level 1: Reactive Agent
    - No LLM usage
    - Simple rule-based responses
    - No memory or learning
    """
    
    def __init__(self):
        self.name = "Reactive Agent"
        self.description = "Simple rule-based agent with no AI capabilities"
        self.capabilities = [
            "❌ No LLM usage",
            "❌ No tool usage",
            "✅ Fast response time",
            "✅ Deterministic outcomes",
            "❌ No contextual understanding"
        ]
    
    def evaluate_claim(self, claim: InsuranceClaim, live_log_callback=None) -> EvaluationResult:
        """Simple rule-based evaluation"""
        start_time = time.time()
        
        if live_log_callback:
            live_log_callback('agent', f"{self.name} starting evaluation", {})
        
        # Simple rules
        rules_applied = []
        
        if claim.amount > 10000:
            recommendation = "investigate"
            risk_level = RiskLevel.HIGH
            rules_applied.append(f"Rule: Amount > $10,000 → Investigate")
            reasons = [f"High claim amount: ${claim.amount:,.2f}"]
            confidence = 0.3
            
            if live_log_callback:
                live_log_callback('agent', f"Applied rule: High amount threshold exceeded", {"amount": claim.amount})
        else:
            recommendation = "approve"
            risk_level = RiskLevel.LOW
            rules_applied.append(f"Rule: Amount ≤ $10,000 → Approve")
            reasons = ["Low claim amount"]
            confidence = 0.3
            
            if live_log_callback:
                live_log_callback('agent', f"Applied rule: Low amount, auto-approve", {"amount": claim.amount})
            
        # Log decision process
        st.session_state.agent_conversations[self.name] = {
            "thought_process": rules_applied,
            "decision_tree": {
                "claim_amount": claim.amount,
                "threshold": 10000,
                "decision": recommendation
            }
        }
        
        if live_log_callback:
            live_log_callback('result', f"Decision: {recommendation}", {"confidence": confidence})
        
        return EvaluationResult(
            claim_id=claim.claim_id,
            recommendation=recommendation,
            confidence=confidence,
            risk_level=risk_level,
            reasons=reasons,
            suggested_payout=claim.amount * 0.8 if recommendation == "approve" else None,
            evaluator=self.name,
            processing_time=time.time() - start_time
        )


class AssistantInsuranceAgent:
    """
    Level 2: Assistant Agent
    - Uses LLM with MCP tools
    - Responds to requests
    - No autonomous behavior
    """
    
    def __init__(self):
        self.name = "Assistant Agent (Claude with MCP)"
        self.description = "LLM-powered agent that uses tools when asked"
        self.capabilities = [
            "✅ LLM reasoning",
            "✅ MCP tool usage",
            "✅ Structured responses",
            "❌ No proactive investigation",
            "❌ No memory"
        ]
        self.tools = InsuranceMCPTools()
    
    async def evaluate_claim(self, claim: InsuranceClaim, live_log_callback=None) -> EvaluationResult:
        """Evaluate using Claude with MCP tools"""
        start_time = time.time()
        tool_usage_log = []
        
        if live_log_callback:
            live_log_callback('agent', f"{self.name} starting evaluation", {})
            live_log_callback('agent', "Preparing claim summary for LLM analysis", {})
        
        # Prepare claim summary
        claim_summary = f"""
        Claim ID: {claim.claim_id}
        Type: {claim.claim_type.value}
        Amount: ${claim.amount:,.2f}
        Description: {claim.description}
        Filed: {claim.date_filed.strftime('%Y-%m-%d')}
        Previous claims: {len(claim.claimant_history)}
        Documents provided: {', '.join(claim.documents)}
        Risk indicators: {', '.join(claim.risk_indicators)}
        """
        
        # Track conversation
        conversation_log = [{
            "role": "system",
            "content": "Insurance claim evaluator using MCP tools"
        }, {
            "role": "user",
            "content": claim_summary
        }]
        
        if live_log_callback:
            live_log_callback('agent', "Sending claim to Claude for analysis", {})
        
        # Create message with tools
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            temperature=0.3,
            tools=self.tools.get_anthropic_tools(),
            messages=[
                {
                    "role": "user",
                    "content": f"""You are an insurance claim evaluator. Evaluate this claim and provide a recommendation.
                    
{claim_summary}

Use the available tools to:
1. Check if the claim is covered by the policy
2. Calculate the risk score
3. Verify documents are complete
4. Check for fraud patterns

Based on your analysis, provide:
- Recommendation: approve, deny, or investigate
- Confidence level (0-1)
- Risk level: low, medium, high, or critical
- Reasons for your decision
- Red flags if any

Think step by step and explain your reasoning."""
                }
            ]
        )
        
        # Process tool calls
        risk_score = 0.5
        red_flags = []
        reasons = []
        
        # Log Claude's response
        conversation_log.append({
            "role": "assistant",
            "content": "Processing claim with MCP tools..."
        })
        
        if message.content:
            for content in message.content:
                if hasattr(content, 'type') and content.type == 'tool_use':
                    # Execute tool
                    def log_tool_call(log_entry):
                        tool_usage_log.append(log_entry)
                    
                    result = await self.tools.execute_tool(
                        content.name,
                        content.input,
                        log_callback=log_tool_call,
                        live_log_callback=live_log_callback
                    )
                    
                    # Process results
                    if content.name == "calculate_risk_score":
                        risk_score = result.get("risk_score", 0.5)
                        reasons.extend(result.get("risk_factors", []))
                    
                    elif content.name == "check_fraud_patterns":
                        if result.get("fraud_score", 0) > 0.5:
                            red_flags.append("High fraud score")
                            red_flags.extend(result.get("suspicious_patterns", []))
                    
                    elif content.name == "verify_documents":
                        if not result.get("complete"):
                            reasons.append(f"Missing documents: {', '.join(result.get('missing_documents', []))}")
        
        # Extract final recommendation from Claude's response
        response_text = message.content[0].text if message.content else ""
        conversation_log.append({
            "role": "assistant",
            "content": response_text
        })
        
        if live_log_callback:
            live_log_callback('agent', "Claude's analysis complete", {"response_preview": response_text[:100]})
        
        # Parse recommendation
        recommendation = "investigate"
        if "approve" in response_text.lower():
            recommendation = "approve"
        elif "deny" in response_text.lower():
            recommendation = "deny"
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = RiskLevel.CRITICAL
        elif risk_score > 0.5:
            risk_level = RiskLevel.HIGH
        elif risk_score > 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        if live_log_callback:
            live_log_callback('result', f"Final decision: {recommendation}", {
                "risk_level": risk_level.value,
                "confidence": 0.7
            })
        
        # Store conversation for display
        st.session_state.agent_conversations[self.name] = {
            "conversation": conversation_log,
            "tool_usage": tool_usage_log,
            "reasoning": response_text
        }
        
        return EvaluationResult(
            claim_id=claim.claim_id,
            recommendation=recommendation,
            confidence=0.7,
            risk_level=risk_level,
            reasons=reasons[:3] if reasons else ["Evaluated using MCP tools"],
            red_flags=red_flags,
            evaluator=self.name,
            processing_time=time.time() - start_time,
            tool_usage=tool_usage_log
        )


class AutonomousInsuranceAgent:
    """
    Level 3: Autonomous Agent
    - Proactive investigation
    - Maintains conversation memory
    - Learns from patterns
    """
    
    def __init__(self):
        self.name = "Autonomous Agent (Claude with Memory)"
        self.description = "Proactive agent with memory and pattern recognition"
        self.capabilities = [
            "✅ LLM reasoning",
            "✅ MCP tool usage",
            "✅ Proactive investigation",
            "✅ Pattern learning",
            "✅ Memory across claims"
        ]
        self.tools = InsuranceMCPTools()
        self.memory = []
        self.pattern_insights = {}
    
    async def evaluate_claim(self, claim: InsuranceClaim, live_log_callback=None) -> Tuple[EvaluationResult, List[str]]:
        """Autonomously evaluate with memory and learning"""
        start_time = time.time()
        actions_taken = []
        tool_usage_log = []
        
        if live_log_callback:
            live_log_callback('agent', f"{self.name} starting autonomous evaluation", {})
            live_log_callback('agent', "Accessing memory and pattern database", {"memory_size": len(self.memory)})
        
        # Add to memory
        self.memory.append({
            "timestamp": datetime.now(),
            "claim_id": claim.claim_id,
            "type": claim.claim_type.value,
            "amount": claim.amount
        })
        
        # Prepare context with memory
        memory_context = ""
        if len(self.memory) > 1:
            recent_claims = self.memory[-5:]
            memory_context = f"""
Previous evaluations:
{chr(10).join([f"- {m['type']} claim for ${m['amount']:,.2f}" for m in recent_claims[:-1]])}

Patterns observed:
- Average claim amount: ${sum(m['amount'] for m in recent_claims) / len(recent_claims):,.2f}
- Claim frequency: {len(recent_claims)} in recent history
- Pattern insights: {json.dumps(self.pattern_insights, indent=2)}
"""
            if live_log_callback:
                live_log_callback('agent', "Retrieved historical patterns", {"patterns": len(self.pattern_insights)})
        
        claim_summary = f"""
        Claim ID: {claim.claim_id}
        Type: {claim.claim_type.value}
        Amount: ${claim.amount:,.2f}
        Description: {claim.description}
        Filed: {claim.date_filed.strftime('%Y-%m-%d')}
        Previous claims: {len(claim.claimant_history)}
        Documents: {', '.join(claim.documents)}
        Risk indicators: {', '.join(claim.risk_indicators)}
        """
        
        if live_log_callback:
            live_log_callback('agent', "Initiating proactive investigation", {})
        
        # Create autonomous evaluation prompt
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            temperature=0.4,
            tools=self.tools.get_anthropic_tools(),
            system="""You are an autonomous insurance claim evaluator with the ability to:
1. Proactively investigate claims
2. Identify patterns across multiple claims
3. Make decisions based on accumulated knowledge
4. Escalate complex cases when confidence is low

You should think step-by-step and use tools proactively to gather all necessary information.
Explain your reasoning and what patterns you notice.""",
            messages=[
                {
                    "role": "user",
                    "content": f"""Evaluate this insurance claim autonomously. 

{memory_context}

Current claim:
{claim_summary}

Proactively investigate using available tools. Consider:
1. Is this claim consistent with patterns you've seen?
2. What additional investigation might reveal important information?
3. Are there any red flags that warrant deeper investigation?
4. Should this be escalated to a senior evaluator?

Think through your evaluation process step by step, using tools as needed.
Explain what you're doing and why."""
                }
            ]
        )
        
        # Track actions and extract insights
        confidence = 0.8
        reasons = []
        red_flags = []
        risk_level = RiskLevel.MEDIUM
        recommendation = "investigate"
        reasoning_log = []
        
        # Process response and tool usage
        for content in message.content:
            if hasattr(content, 'type') and content.type == 'tool_use':
                actions_taken.append(f"Used tool: {content.name}")
                
                if live_log_callback:
                    live_log_callback('agent', f"Proactively using tool: {content.name}", {})
                
                # Execute tool
                def log_tool_call(log_entry):
                    tool_usage_log.append(log_entry)
                
                result = await self.tools.execute_tool(
                    content.name, 
                    content.input,
                    log_callback=log_tool_call,
                    live_log_callback=live_log_callback
                )
                
                # Process based on tool type
                if content.name == "investigate_claim":
                    actions_taken.append("Proactive investigation initiated")
                    if any("contradict" in str(v).lower() for v in result.values()):
                        red_flags.append("Investigation found contradictions")
                        confidence -= 0.2
                
                elif content.name == "check_fraud_patterns":
                    fraud_score = result.get("fraud_score", 0)
                    if fraud_score > 0.5:
                        red_flags.extend(result.get("suspicious_patterns", []))
                        confidence -= 0.1
                        risk_level = RiskLevel.HIGH
            
            elif hasattr(content, 'text'):
                reasoning_log.append(content.text)
                
                # Extract insights from Claude's reasoning
                text = content.text.lower()
                if "high risk" in text:
                    risk_level = RiskLevel.HIGH
                elif "low risk" in text:
                    risk_level = RiskLevel.LOW
                
                if "approve" in text and "recommend" in text:
                    recommendation = "approve"
                elif "deny" in text and "recommend" in text:
                    recommendation = "deny"
                
                # Extract reasons
                if "because" in text:
                    reason_start = text.find("because")
                    reason_text = text[reason_start:reason_start+100].split('.')[0]
                    reasons.append(reason_text)
        
        # Update pattern insights
        pattern_key = f"{claim.claim_type.value}_{risk_level.value}"
        if pattern_key not in self.pattern_insights:
            self.pattern_insights[pattern_key] = {"count": 0, "outcomes": {}}
        
        self.pattern_insights[pattern_key]["count"] += 1
        self.pattern_insights[pattern_key]["outcomes"][recommendation] = \
            self.pattern_insights[pattern_key]["outcomes"].get(recommendation, 0) + 1
        
        actions_taken.append(f"Updated pattern insights for {pattern_key}")
        
        if live_log_callback:
            live_log_callback('agent', "Pattern database updated", {"pattern": pattern_key})
        
        # Low confidence triggers escalation
        if confidence < 0.6:
            actions_taken.append("Escalating to senior evaluator due to low confidence")
            reasons.append("Requires senior review")
            if live_log_callback:
                live_log_callback('agent', "⚠️ Escalating to senior review", {"confidence": confidence})
        
        if live_log_callback:
            live_log_callback('result', f"Autonomous evaluation complete: {recommendation}", {
                "confidence": confidence,
                "actions_taken": len(actions_taken)
            })
        
        # Store conversation for display
        st.session_state.agent_conversations[self.name] = {
            "memory_context": memory_context,
            "reasoning": reasoning_log,
            "tool_usage": tool_usage_log,
            "patterns": self.pattern_insights,
            "actions": actions_taken
        }
        
        return EvaluationResult(
            claim_id=claim.claim_id,
            recommendation=recommendation,
            confidence=confidence,
            risk_level=risk_level,
            reasons=reasons[:3] if reasons else ["Autonomous evaluation completed"],
            red_flags=red_flags,
            evaluator=self.name,
            processing_time=time.time() - start_time,
            tool_usage=tool_usage_log
        ), actions_taken


class MultiAgentInsuranceSystem:
    """
    Level 4: Multi-Agent System
    - Multiple specialized Claude instances
    - Collaborative decision making
    - Emergent consensus
    """
    
    def __init__(self):
        self.name = "Multi-Agent System (Claude Specialists)"
        self.description = "Multiple specialized agents working together"
        self.capabilities = [
            "✅ Multiple perspectives",
            "✅ Specialist knowledge",
            "✅ Consensus building",
            "✅ Senior arbitration",
            "✅ Comprehensive analysis"
        ]
        self.tools = InsuranceMCPTools()
    
    async def evaluate_claim(self, claim: InsuranceClaim, live_log_callback=None) -> Tuple[EvaluationResult, Dict[str, Any]]:
        """Multi-agent collaborative evaluation"""
        start_time = time.time()
        evaluation_log = {
            "agents": ["Fraud Specialist", "Risk Analyst", "Customer Advocate"],
            "consensus": False,
            "deliberation_rounds": 0,
            "agent_conversations": {}
        }
        
        if live_log_callback:
            live_log_callback('agent', f"{self.name} initiating multi-agent evaluation", {
                "specialists": evaluation_log["agents"]
            })
        
        claim_summary = f"""
        Claim ID: {claim.claim_id}
        Type: {claim.claim_type.value}
        Amount: ${claim.amount:,.2f}
        Description: {claim.description}
        Documents: {', '.join(claim.documents)}
        Risk indicators: {', '.join(claim.risk_indicators)}
        """
        
        # Get evaluations from different specialist perspectives
        evaluations = {}
        
        if live_log_callback:
            live_log_callback('agent', "Specialists beginning parallel evaluation", {})
        
        # 1. Fraud Specialist
        if live_log_callback:
            live_log_callback('agent', "Fraud Specialist analyzing claim", {})
        fraud_eval, fraud_conv = await self._get_specialist_evaluation(
            claim_summary,
            "fraud detection specialist",
            "Focus on identifying potential fraud patterns, suspicious timing, and inconsistencies.",
            live_log_callback
        )
        evaluations["Fraud Specialist"] = fraud_eval
        evaluation_log["agent_conversations"]["Fraud Specialist"] = fraud_conv
        
        # 2. Risk Analyst
        if live_log_callback:
            live_log_callback('agent', "Risk Analyst evaluating overall risk", {})
        risk_eval, risk_conv = await self._get_specialist_evaluation(
            claim_summary,
            "risk analysis expert",
            "Focus on calculating overall risk, considering claim amount, history, and documentation.",
            live_log_callback
        )
        evaluations["Risk Analyst"] = risk_eval
        evaluation_log["agent_conversations"]["Risk Analyst"] = risk_conv
        
        # 3. Customer Advocate
        if live_log_callback:
            live_log_callback('agent', "Customer Advocate reviewing fairness", {})
        customer_eval, customer_conv = await self._get_specialist_evaluation(
            claim_summary,
            "customer advocate",
            "Focus on fair treatment, customer history, and giving benefit of the doubt where reasonable.",
            live_log_callback
        )
        evaluations["Customer Advocate"] = customer_eval
        evaluation_log["agent_conversations"]["Customer Advocate"] = customer_conv
        
        # Check for consensus
        recommendations = [e["recommendation"] for e in evaluations.values()]
        consensus_recommendation = max(set(recommendations), key=recommendations.count)
        
        if live_log_callback:
            live_log_callback('agent', "Checking for consensus among specialists", {
                "recommendations": recommendations
            })
        
        if recommendations.count(consensus_recommendation) >= 2:
            evaluation_log["consensus"] = True
            
            if live_log_callback:
                live_log_callback('agent', "✅ Consensus reached!", {
                    "decision": consensus_recommendation,
                    "vote": f"{recommendations.count(consensus_recommendation)}/3"
                })
            
            # Average confidence
            avg_confidence = sum(e["confidence"] for e in evaluations.values()) / 3
            
            # Combine insights
            all_reasons = []
            all_red_flags = []
            for agent, eval_data in evaluations.items():
                all_reasons.extend([f"[{agent}] {r}" for r in eval_data.get("reasons", [])])
                all_red_flags.extend(eval_data.get("red_flags", []))
            
            # Determine risk level (take highest)
            risk_levels = [e["risk_level"] for e in evaluations.values()]
            risk_order = [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]
            final_risk = RiskLevel.MEDIUM
            for risk in risk_order:
                if risk.value in risk_levels:
                    final_risk = risk
                    break
            
            result = EvaluationResult(
                claim_id=claim.claim_id,
                recommendation=consensus_recommendation,
                confidence=avg_confidence,
                risk_level=final_risk,
                reasons=all_reasons[:5],
                red_flags=list(set(all_red_flags)),
                evaluator=self.name,
                processing_time=time.time() - start_time
            )
        else:
            # No consensus - senior arbitration needed
            evaluation_log["consensus"] = False
            
            if live_log_callback:
                live_log_callback('agent', "❌ No consensus - escalating to senior arbitrator", {
                    "recommendations": recommendations
                })
            
            result, arbitration_conv = await self._senior_arbitration(claim_summary, evaluations, live_log_callback)
            evaluation_log["agent_conversations"]["Senior Arbitrator"] = arbitration_conv
        
        evaluation_log["evaluations"] = evaluations
        
        if live_log_callback:
            live_log_callback('result', f"Multi-agent evaluation complete: {result.recommendation}", {
                "consensus": evaluation_log["consensus"],
                "final_confidence": result.confidence
            })
        
        # Store for display
        st.session_state.agent_conversations[self.name] = evaluation_log
        
        return result, evaluation_log
    
    async def _get_specialist_evaluation(self, claim_summary: str, role: str, focus: str, live_log_callback=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get evaluation from a specialist perspective"""
        tool_usage_log = []
        
        def log_tool_call(log_entry):
            tool_usage_log.append(log_entry)
        
        # MUCH MORE EXTREME specialist personalities
        specialist_prompts = {
            "fraud detection specialist": {
                "system": """You are a HARDLINE fraud detection specialist. You've caught thousands of fraudsters over 20 years.

YOUR MINDSET:
- 30% of claims have some element of fraud or exaggeration
- Fraudsters are getting more sophisticated every year
- Your job is to PROTECT THE COMPANY'S MONEY
- You'd rather investigate 10 legitimate claims than let 1 fraud through
- You've seen every trick in the book

YOUR DECISION RULES:
- Any red flags → INVESTIGATE or DENY
- Conflicting statements → INVESTIGATE minimum, likely DENY
- Suspicious timing → INVESTIGATE or DENY
- Multiple previous claims → INVESTIGATE
- High amounts with minimal documentation → DENY
- Only APPROVE if: low amount, perfect documentation, zero red flags, good history

BE SUSPICIOUS. BE TOUGH. The company hired you to stop fraud.""",
                "temperature": 0.6,
                "decision_bias": "deny"
            },
            
            "risk analysis expert": {
                "system": """You are a NUMBERS-ONLY risk analyst. You make decisions purely on financial calculations.

YOUR APPROACH:
- Emotion is irrelevant. Only data matters.
- Calculate expected value: (probability of fraud × claim amount) vs investigation cost
- Investigation costs the company $500-1000 in time and resources

YOUR DECISION MATRIX:
- Claim < $5,000 + low risk indicators → APPROVE (investigation cost exceeds benefit)
- Claim < $5,000 + high risk indicators → DENY (likely fraud, not worth investigating)
- Claim $5,000-$20,000 + low risk → APPROVE 
- Claim $5,000-$20,000 + medium/high risk → INVESTIGATE
- Claim > $20,000 + any risk → INVESTIGATE
- Claim > $50,000 → Always INVESTIGATE

You only care about ROI. Make the financially optimal decision.""",
                "temperature": 0.1,
                "decision_bias": "calculated"
            },
            
            "customer advocate": {
                "system": """You are a PASSIONATE customer advocate. You fight for policyholders who pay premiums faithfully.

YOUR BELIEFS:
- Insurance companies make billions - they can afford to pay legitimate claims
- Most people are honest and deserve the benefit of the doubt
- Denying legitimate claims destroys lives and families
- Over-investigation frustrates and insults good customers
- Customer retention is worth more than catching small fraud

YOUR DECISION APPROACH:
- Documentation provided → APPROVE
- Long-time customer → APPROVE
- First claim in years → APPROVE
- Reasonable explanation → APPROVE
- Only INVESTIGATE if: massive red flags AND high amount
- Only DENY if: absolutely clear fraud with proof
- When in doubt → APPROVE

FIGHT FOR THE CUSTOMER. They've paid premiums for years - pay their claim!""",
                "temperature": 0.5,
                "decision_bias": "approve"
            }
        }
        
        # Get specialist configuration
        specialist_config = specialist_prompts.get(role, {
            "system": f"You are a {role} evaluating insurance claims. {focus}",
            "temperature": 0.3,
            "decision_bias": "neutral"
        })
        
        # Force them to make a STRONG decision
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            temperature=specialist_config["temperature"],
            tools=self.tools.get_anthropic_tools(),
            system=specialist_config["system"],
            messages=[
                {
                    "role": "user",
                    "content": f"""Evaluate this claim from YOUR SPECIALIST PERSPECTIVE:

{claim_summary}

IMPORTANT: You must make a STRONG DECISION based on your role:
- Fraud Specialist: You PROTECT the company. Be SUSPICIOUS.
- Risk Analyst: You calculate NUMBERS ONLY. Ignore emotions.
- Customer Advocate: You FIGHT for customers. Give benefit of doubt.

1. Your immediate reaction based on YOUR ROLE
2. Use tools that support YOUR PERSPECTIVE
3. Interpret findings through YOUR LENS
4. Make a FIRM DECISION:

**I STRONGLY RECOMMEND: [APPROVE/DENY/INVESTIGATE]**

State this clearly. Don't hedge. Don't be neutral. Take a position!

5. Confidence: [0.0-1.0]
6. Risk from YOUR view: [low/medium/high/critical]
7. Your TOP THREE reasons (from YOUR perspective)
8. Red flags YOU care about

BE TRUE TO YOUR ROLE. Don't try to be balanced. That's what the other specialists are for!"""
                }
            ]
        )
        
        # Process response and tool calls
        response_text = ""
        actual_tool_calls = []
        
        if message.content:
            for content in message.content:
                if hasattr(content, 'type') and content.type == 'tool_use':
                    # Execute tool
                    result = await self.tools.execute_tool(
                        content.name,
                        content.input,
                        log_callback=log_tool_call,
                        live_log_callback=live_log_callback
                    )
                    actual_tool_calls.append({
                        "tool": content.name,
                        "input": content.input,
                        "result": result
                    })
                elif hasattr(content, 'text'):
                    response_text += content.text + "\n"
        
        # Parse recommendation with fallback to role bias
        import re
        
        recommendation = "investigate"
        
        # Look for strong recommendation statement
        strong_rec = re.search(r'STRONGLY RECOMMEND:\s*\**\s*(\w+)', response_text, re.IGNORECASE)
        if strong_rec:
            rec_word = strong_rec.group(1).lower()
            if 'approve' in rec_word:
                recommendation = "approve"
            elif 'deny' in rec_word or 'reject' in rec_word:
                recommendation = "deny"
            elif 'investigate' in rec_word:
                recommendation = "investigate"
        else:
            # Fallback parsing
            text_lower = response_text.lower()
            if any(phrase in text_lower for phrase in ["recommend approval", "should be approved", "approve the claim", "i approve"]):
                recommendation = "approve"
            elif any(phrase in text_lower for phrase in ["recommend denial", "should be denied", "deny the claim", "reject the claim", "i deny"]):
                recommendation = "deny"
            elif any(phrase in text_lower for phrase in ["needs investigation", "should investigate", "must investigate"]):
                recommendation = "investigate"
            
            # Apply role-specific bias if still unclear
            if recommendation == "investigate":
                if specialist_config["decision_bias"] == "approve":
                    # Customer advocate defaults to approve
                    if "no major red flags" in text_lower or "benefit of doubt" in text_lower:
                        recommendation = "approve"
                elif specialist_config["decision_bias"] == "deny":
                    # Fraud specialist defaults to investigate/deny
                    if "suspicious" in text_lower or "red flag" in text_lower:
                        recommendation = "deny"
                elif specialist_config["decision_bias"] == "calculated":
                    # Risk analyst based on amount
                    amount_match = re.search(r'\$(\d+(?:,\d+)*(?:\.\d+)?)', claim_summary)
                    if amount_match:
                        amount = float(amount_match.group(1).replace(',', ''))
                        if amount < 5000:
                            recommendation = "approve"
                        elif amount > 50000:
                            recommendation = "investigate"
        
        # Extract confidence (specialists should be confident in their bias)
        confidence = 0.8  # Higher default
        conf_match = re.search(r'(?:confidence|confident)[\s:]+(\d*\.?\d+)', response_text, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                if confidence > 1:
                    confidence = confidence / 100
            except:
                pass
        
        # Risk level based on specialist perspective
        risk_level = "medium"
        text_lower = response_text.lower()
        
        if role == "fraud detection specialist":
            # Fraud specialist sees more risk
            if any(term in text_lower for term in ["high risk", "suspicious", "red flag"]):
                risk_level = "high"
            elif "critical" in text_lower or "clear fraud" in text_lower:
                risk_level = "critical"
            elif "low risk" in text_lower and "no red flags" in text_lower:
                risk_level = "low"
        elif role == "customer advocate":
            # Customer advocate sees less risk
            if "critical" in text_lower or "obvious fraud" in text_lower:
                risk_level = "high"
            elif "some risk" in text_lower:
                risk_level = "medium"
            else:
                risk_level = "low"
        else:
            # Standard risk parsing
            if "critical risk" in text_lower or "very high risk" in text_lower:
                risk_level = "critical"
            elif "high risk" in text_lower:
                risk_level = "high"
            elif "low risk" in text_lower:
                risk_level = "low"
        
        # Extract reasons
        reasons = []
        reason_patterns = [
            r'\d+\.\s*([^.!?\n]+)',
            r'(?:because|due to|reason:)\s*([^.!?\n]+)',
            r'[-•]\s*([^.!?\n]+)'
        ]
        
        for pattern in reason_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            reasons.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        reasons = reasons[:3] if reasons else [f"{role} perspective applied"]
        
        # Extract red flags (different specialists care about different flags)
        red_flags = []
        if role == "fraud detection specialist":
            flag_patterns = [
                r'(?:red flag|suspicious|concerning):\s*([^.!?\n]+)',
                r'(?:fraud indicator|warning sign):\s*([^.!?\n]+)'
            ]
        else:
            flag_patterns = [
                r'(?:concern|issue|problem):\s*([^.!?\n]+)',
            ]
        
        for pattern in flag_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            red_flags.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        conversation = {
            "role": role,
            "focus": focus,
            "response": response_text,
            "tool_usage": tool_usage_log,
            "actual_tools_called": actual_tool_calls,
            "parsed_recommendation": recommendation,
            "specialist_bias": specialist_config["decision_bias"]
        }
        
        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "risk_level": risk_level,
            "reasons": reasons,
            "red_flags": red_flags[:3] if red_flags else []
        }, conversation
    
    async def _senior_arbitration(self, claim_summary: str, evaluations: Dict[str, Dict], live_log_callback=None) -> Tuple[EvaluationResult, Dict[str, Any]]:
        """Senior evaluator makes final decision when no consensus"""
        
        if live_log_callback:
            live_log_callback('agent', "Senior arbitrator reviewing all specialist opinions", {})
        
        # Prepare evaluation summary
        eval_summary = "\n".join([
            f"{agent}: {data['recommendation']} (confidence: {data['confidence']})"
            for agent, data in evaluations.items()
        ])
        
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.2,
            system="You are a senior insurance evaluator making final decisions when specialist agents disagree.",
            messages=[
                {
                    "role": "user",
                    "content": f"""The specialist agents could not reach consensus on this claim:

{claim_summary}

Agent evaluations:
{eval_summary}

As the senior evaluator, make the final decision. Consider all perspectives and provide:
1. Final recommendation
2. Confidence in your decision
3. Risk level assessment
4. Key reasons for your decision

Explain your arbitration process."""
                }
            ]
        )
        
        # Extract decision
        response_text = message.content[0].text if message.content else ""
        
        recommendation = "investigate"
        if "approve" in response_text.lower():
            recommendation = "approve"
        elif "deny" in response_text.lower():
            recommendation = "deny"
        
        if live_log_callback:
            live_log_callback('agent', f"Senior arbitrator decision: {recommendation}", {})
        
        conversation = {
            "role": "Senior Arbitrator",
            "evaluations_summary": eval_summary,
            "decision_process": response_text
        }
        
        return EvaluationResult(
            claim_id="CLM-2024-001",  # Fix hardcoded ID
            recommendation=recommendation,
            confidence=0.85,
            risk_level=RiskLevel.MEDIUM,
            reasons=["Senior arbitration: no consensus among specialists"],
            evaluator="Senior Evaluator"
        ), conversation


# ============================================================================
# Tutorial Functions (Improvement #1)
# ============================================================================

def display_tutorial_mode():
    """Interactive tutorial mode for learning about agent levels"""
    steps = [
        {
            "title": "Welcome to AI Agent Levels!",
            "content": """Let's explore how AI agents evolve from simple rules to complex systems.
            
            In this tutorial, you'll see:
            - **Level 1**: Rule-based reactive agents
            - **Level 2**: LLM-powered assistants with tools
            - **Level 3**: Autonomous agents with memory
            - **Level 4**: Multi-agent collaborative systems
            
            Click 'Next' to begin exploring each level!""",
            "highlight": None
        },
        {
            "title": "Level 1: Reactive Agents",
            "content": """**Reactive agents** use simple if-then rules with no AI.
            
            **Characteristics:**
            - ⚡ Lightning fast (milliseconds)
            - 📏 Deterministic outcomes
            - ❌ No understanding of context
            - 💰 Zero AI costs
            
            **Best for:** High-volume, simple decisions with clear rules.
            
            Try creating a claim and see how it only looks at the amount!""",
            "highlight": "reactive"
        },
        {
            "title": "Level 2: Assistant Agents",
            "content": """**Assistant agents** add LLM reasoning and MCP tools.
            
            **New capabilities:**
            - 🧠 Natural language understanding
            - 🔧 Uses tools to gather information
            - 📊 Provides reasoned explanations
            - ⚖️ Balances multiple factors
            
            **MCP Protocol:** Structured tool calling with type safety.
            
            Notice how this agent actively uses tools to analyze the claim!""",
            "highlight": "assistant"
        },
        {
            "title": "Level 3: Autonomous Agents",
            "content": """**Autonomous agents** add memory and proactive behavior.
            
            **Advanced features:**
            - 💾 Remembers previous claims
            - 🔍 Proactively investigates
            - 📈 Learns patterns over time
            - 🚨 Self-escalates when uncertain
            
            **Key insight:** These agents get smarter with each evaluation!
            
            Watch how it references past patterns and takes initiative.""",
            "highlight": "autonomous"
        },
        {
            "title": "Level 4: Multi-Agent Systems",
            "content": """**Multi-agent systems** use specialist collaboration.
            
            **Collaborative features:**
            - 👥 Multiple specialist perspectives
            - 🗳️ Consensus building
            - ⚖️ Senior arbitration
            - 🎯 Highest accuracy
            
            **Trade-off:** More complex and expensive, but best for critical decisions.
            
            See how specialists debate and reach consensus!""",
            "highlight": "multiagent"
        }
    ]
    
    current_step = st.session_state.tutorial_step
    
    # Tutorial UI
    tutorial_container = st.container()
    with tutorial_container:
        col1, col2, col3 = st.columns([1, 6, 1])
        
        with col1:
            if st.button("◀ Previous", disabled=current_step == 0):
                st.session_state.tutorial_step = max(0, current_step - 1)
                st.rerun()
        
        with col2:
            st.markdown(f"### Step {current_step + 1} of {len(steps)}: {steps[current_step]['title']}")
            st.info(steps[current_step]['content'])
            
            # Progress bar
            progress = (current_step + 1) / len(steps)
            st.progress(progress)
        
        with col3:
            if st.button("Next ▶", disabled=current_step == len(steps) - 1):
                st.session_state.tutorial_step = min(len(steps) - 1, current_step + 1)
                st.rerun()
        
        # Tutorial completion
        if current_step == len(steps) - 1:
            st.success("🎉 Tutorial complete! Try evaluating a claim to see all agents in action.")
            if st.button("Exit Tutorial", type="primary"):
                st.session_state.tutorial_mode = False
                st.rerun()


# ============================================================================
# Live Agent Thinking Display (Improvement #2)
# ============================================================================

def display_live_agent_thinking():
    """Show agent's thought process as it happens"""
    if 'live_thoughts' not in st.session_state or not st.session_state.live_thoughts:
        return
    
    st.subheader("🧠 Live Agent Thinking")
    
    # Create a scrollable container for live thoughts
    thought_container = st.container()
    
    with thought_container:
        for thought in st.session_state.live_thoughts[-10:]:  # Show last 10 thoughts
            if thought['type'] == 'agent':
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(f"**{thought.get('agent', 'Agent')}:** {thought['content']}")
                    if thought.get('details'):
                        with st.expander("Details"):
                            st.json(thought['details'])
            
            elif thought['type'] == 'tool':
                with st.chat_message("function", avatar="🔧"):
                    st.code(f"Calling: {thought['content']}\nInput: {json.dumps(thought.get('details', {}), indent=2)}")
            
            elif thought['type'] == 'result':
                with st.chat_message("system", avatar="✅"):
                    st.success(thought['content'])
                    if thought.get('details'):
                        with st.expander("Result Details"):
                            st.json(thought['details'])


def add_live_thought(thought_type: str, content: str, details: Dict = None):
    """Add a thought to the live display"""
    if 'live_thoughts' not in st.session_state:
        st.session_state.live_thoughts = []
    
    st.session_state.live_thoughts.append({
        'type': thought_type,
        'content': content,
        'details': details,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# Comparison Mode (Improvement #3)
# ============================================================================

def display_comparison_mode():
    """Side-by-side agent comparison"""
    st.subheader("🔍 Agent Comparison Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        agent1 = st.selectbox(
            "First Agent",
            ["Level 1: Reactive", "Level 2: Assistant", "Level 3: Autonomous", "Level 4: Multi-Agent"],
            key="compare_agent1"
        )
    
    with col2:
        agent2 = st.selectbox(
            "Second Agent",
            ["Level 1: Reactive", "Level 2: Assistant", "Level 3: Autonomous", "Level 4: Multi-Agent"],
            key="compare_agent2"
        )
    
    if st.button("Compare Agents", type="primary"):
        st.session_state.comparison_mode = True
        st.session_state.comparison_agents = [agent1, agent2]
    
    # Display comparison results
    if st.session_state.comparison_mode and 'comparison_results' in st.session_state:
        display_comparison_results()


def display_comparison_results():
    """Display side-by-side comparison results"""
    results = st.session_state.comparison_results
    
    if not results:
        return
    
    # Create comparison metrics
    col1, col2 = st.columns(2)
    
    for i, (agent_name, result) in enumerate(results.items()):
        with col1 if i == 0 else col2:
            st.markdown(f"### {agent_name}")
            
            # Metrics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Decision", result.recommendation.upper())
                st.metric("Confidence", f"{result.confidence:.2%}")
            
            with metric_col2:
                st.metric("Risk Level", result.risk_level.value.upper())
                st.metric("Time", f"{result.processing_time:.2f}s")
            
            # Reasons
            st.markdown("**Reasoning:**")
            for reason in result.reasons[:3]:
                st.write(f"• {reason}")
            
            # Red flags
            if result.red_flags:
                st.markdown("**🚩 Red Flags:**")
                for flag in result.red_flags:
                    st.write(f"• {flag}")
            
            # Tool usage
            if hasattr(result, 'tool_usage') and result.tool_usage:
                st.markdown(f"**Tools Used:** {len(result.tool_usage)}")
    
    # Comparison insights
    st.markdown("---")
    st.markdown("### 📊 Comparison Insights")
    
    # Agreement check
    decisions = [r.recommendation for r in results.values()]
    if len(set(decisions)) == 1:
        st.success(f"✅ Both agents agree: **{decisions[0].upper()}**")
    else:
        st.warning(f"❌ Agents disagree: {' vs '.join([d.upper() for d in decisions])}")
    
    # Performance comparison
    times = [r.processing_time for r in results.values()]
    faster_agent = list(results.keys())[0] if times[0] < times[1] else list(results.keys())[1]
    speed_diff = abs(times[0] - times[1])
    st.info(f"⚡ {faster_agent} was {speed_diff:.2f}s faster")


# ============================================================================
# Enhanced Visualizations (Improvement #5)
# ============================================================================

def create_decision_flow_animation(agent_level: int, claim: InsuranceClaim):
    """Animated flow showing how each agent processes the claim"""
    fig = go.Figure()
    
    if agent_level == 1:
        # Simple decision tree animation
        steps = [
            {"x": [0], "y": [0], "text": ["Claim Input"], "color": ["lightblue"]},
            {"x": [0, 0], "y": [0, -1], "text": ["Claim Input", "Check Amount"], "color": ["lightblue", "yellow"]},
            {"x": [0, 0, -1, 1], "y": [0, -1, -2, -2], "text": ["Claim Input", "Check Amount", "≤$10k", ">$10k"], "color": ["lightblue", "yellow", "green", "red"]},
            {"x": [0, 0, -1, 1, -1, 1], "y": [0, -1, -2, -2, -3, -3], "text": ["Claim Input", "Check Amount", "≤$10k", ">$10k", "Approve", "Investigate"], "color": ["lightblue", "yellow", "green", "red", "lightgreen", "orange"]}
        ]
        
        frames = []
        for i, step in enumerate(steps):
            frame = go.Frame(
                data=[go.Scatter(
                    x=step["x"],
                    y=step["y"],
                    mode='markers+text',
                    text=step["text"],
                    textposition="top center",
                    marker=dict(size=40, color=step["color"], line=dict(width=2, color='black')),
                    showlegend=False
                )],
                name=str(i)
            )
            frames.append(frame)
        
        fig = go.Figure(
            data=[frames[0].data[0]],
            frames=frames,
            layout=go.Layout(
                title=f"Level 1: Decision Flow for ${claim.amount:,.2f} claim",
                updatemenus=[{
                    "type": "buttons",
                    "showactive": False,
                    "y": 0,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "bottom",
                    "buttons": [
                        {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}]},
                        {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                    ]
                }],
                xaxis=dict(range=[-2, 2], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[-4, 1], showgrid=False, zeroline=False, showticklabels=False),
                height=400
            )
        )
    
    elif agent_level == 2:
        # Tool usage flow
        tools = ["Policy Check", "Risk Score", "Document Verify", "Fraud Check", "Decision"]
        x_pos = list(range(len(tools)))
        
        # Create frames showing sequential tool execution
        frames = []
        for i in range(len(tools) + 1):
            active = ["lightgray"] * len(tools)
            for j in range(i):
                active[j] = "lightgreen"
            if i < len(tools):
                active[i] = "yellow"
            
            frame = go.Frame(
                data=[go.Bar(
                    x=x_pos[:i+1] if i < len(tools) else x_pos,
                    y=[1] * min(i+1, len(tools)),
                    text=tools[:i+1] if i < len(tools) else tools,
                    textposition='inside',
                    marker_color=active[:i+1] if i < len(tools) else active,
                    showlegend=False
                )],
                name=str(i)
            )
            frames.append(frame)
        
        fig = go.Figure(
            data=[frames[0].data[0]],
            frames=frames,
            layout=go.Layout(
                title="Level 2: MCP Tool Execution Flow",
                xaxis=dict(ticktext=tools, tickvals=x_pos),
                yaxis=dict(range=[0, 1.5], showticklabels=False),
                updatemenus=[{
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 800}}]},
                        {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
                    ]
                }],
                height=400
            )
        )
    
    return fig


def display_cost_benefit_analysis():
    """Show ROI for different agent levels based on claim volume"""
    st.subheader("💰 Cost-Benefit Analysis by Volume")
    
    # Calculate costs for different volumes
    volumes = [100, 1000, 10000, 100000]
    
    data = []
    for volume in volumes:
        for level in range(1, 5):
            # Cost calculation
            if level == 1:
                cost = volume * 0.001  # $0.001 per claim
            elif level == 2:
                cost = volume * 0.01   # $0.01 per claim
            elif level == 3:
                cost = volume * 0.03   # $0.03 per claim
            else:
                cost = volume * 0.08   # $0.08 per claim
            
            # Accuracy benefit (prevented losses)
            accuracy = [0.6, 0.75, 0.85, 0.95][level-1]
            avg_claim = 15000
            fraud_rate = 0.05
            prevented_loss = volume * avg_claim * fraud_rate * accuracy
            
            data.append({
                'Volume': volume,
                'Level': f'Level {level}',
                'Total Cost': cost,
                'Prevented Losses': prevented_loss,
                'Net Benefit': prevented_loss - cost,
                'ROI': ((prevented_loss - cost) / cost * 100) if cost > 0 else 0
            })
    
    df = pd.DataFrame(data)
    
    # Net benefit chart
    fig_benefit = px.line(
        df, 
        x='Volume', 
        y='Net Benefit', 
        color='Level',
        title='Net Benefit by Claim Volume',
        log_x=True,
        markers=True,
        labels={'Net Benefit': 'Net Benefit ($)'}
    )
    fig_benefit.update_layout(height=400)
    st.plotly_chart(fig_benefit, use_container_width=True)
    
    # ROI chart
    fig_roi = px.bar(
        df,
        x='Volume',
        y='ROI',
        color='Level',
        title='Return on Investment (ROI) by Volume',
        barmode='group',
        labels={'ROI': 'ROI (%)'}
    )
    fig_roi.update_layout(height=400)
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # Insights
    st.info("""
    **Key Insights:**
    - Level 1 agents are most cost-effective for very high volumes
    - Level 4 agents provide best ROI for lower volumes with high-value claims
    - Break-even points shift based on average claim value and fraud rate
    """)


# ============================================================================
# Streamlit UI with Improvements
# ============================================================================

def main():
    st.title("🤖 AI Agent Levels: Insurance Claims with MCP Protocol")
    st.markdown("""
    This educational demo shows how different levels of AI agents evaluate insurance claims using the Model Context Protocol (MCP).
    Watch how agents progress from simple rules to complex multi-agent systems!
    """)
    
    # Tutorial mode toggle
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.checkbox("🎓 Tutorial Mode", value=st.session_state.get('tutorial_mode', False)):
            st.session_state.tutorial_mode = True
        else:
            st.session_state.tutorial_mode = False
    
    with col2:
        if st.checkbox("🔍 Compare Mode", value=st.session_state.get('comparison_mode', False)):
            st.session_state.comparison_mode = True
        else:
            st.session_state.comparison_mode = False
    
    # Display tutorial if active
    if st.session_state.get('tutorial_mode', False):
        display_tutorial_mode()
        st.markdown("---")
    
    # Display comparison mode if active
    if st.session_state.get('comparison_mode', False):
        display_comparison_mode()
        st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("📋 Create Test Claim")
        
        # Scenario presets (Improvement #4)
        preset = st.selectbox(
            "📌 Load Scenario Preset",
            ["Custom"] + list(SCENARIO_PRESETS.keys()),
            help="Choose a pre-configured scenario or create your own"
        )
        
        # Load preset values or use defaults
        if preset != "Custom" and preset in SCENARIO_PRESETS:
            preset_data = SCENARIO_PRESETS[preset]
            claim_type = preset_data["claim_type"]
            claim_amount = preset_data["amount"]
            claim_description = preset_data["description"]
            selected_docs = preset_data["documents"]
            risk_indicators = preset_data["risk_indicators"]
            num_previous_claims = preset_data["previous_claims"]
            
            st.success(f"✅ Loaded preset: {preset}")
        else:
            claim_type = st.selectbox(
                "Claim Type",
                options=[ct.value for ct in ClaimType],
                format_func=lambda x: x.title()
            )
            
            claim_amount = st.number_input(
                "Claim Amount ($)",
                min_value=1000,
                max_value=100000,
                value=15000,
                step=1000
            )
            
            claim_description = st.text_area(
                "Claim Description",
                value="Vehicle collision at intersection. Sudden brake failure caused accident.",
                height=100
            )
            
            # Document selection
            st.subheader("📄 Documents Provided")
            doc_options = {
                "auto": ["police_report", "photos", "repair_estimate"],
                "property": ["photos", "repair_estimate", "proof_of_ownership"],
                "health": ["medical_records", "bills", "prescription"],
                "liability": ["incident_report", "witness_statements"]
            }
            
            selected_docs = st.multiselect(
                "Select documents",
                options=doc_options.get(claim_type, []),
                default=["police_report", "photos"] if claim_type == "auto" else []
            )
            
            # Risk indicators
            risk_indicators = st.multiselect(
                "Risk Indicators",
                options=["multiple_claims", "high_amount", "recent_incident", "suspicious_timing"],
                default=["multiple_claims", "high_amount"]
            )
            
            # Previous claims
            num_previous_claims = st.slider("Number of Previous Claims", 0, 5, 2)
        
        # Show claim preview
        with st.expander("👁️ Claim Preview", expanded=True):
            st.markdown(f"""
            **Type:** {claim_type.title()}  
            **Amount:** ${claim_amount:,}  
            **Description:** {claim_description[:50]}...  
            **Documents:** {len(selected_docs)}  
            **Risk Indicators:** {len(risk_indicators)}  
            **Previous Claims:** {num_previous_claims}
            """)
        
        # Evaluation buttons
        st.markdown("---")
        
        if st.session_state.get('comparison_mode', False):
            if st.button("🔄 Compare Selected Agents", type="primary", use_container_width=True):
                # Create claim
                claim = InsuranceClaim(
                    claim_id=f"CLM-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    claim_type=ClaimType(claim_type),
                    amount=claim_amount,
                    date_filed=datetime.now() - timedelta(days=3),
                    description=claim_description,
                    policy_number="POL-123456",
                    claimant_history=[
                        {"date": f"2023-{i:02d}-01", "amount": 5000 + i * 1000}
                        for i in range(1, num_previous_claims + 1)
                    ],
                    documents=selected_docs,
                    risk_indicators=risk_indicators
                )
                
                # Run comparison
                asyncio.run(run_comparison(claim))
        else:
            if st.button("🚀 Evaluate Claim", type="primary", use_container_width=True):
                # Create claim
                claim = InsuranceClaim(
                    claim_id=f"CLM-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    claim_type=ClaimType(claim_type),
                    amount=claim_amount,
                    date_filed=datetime.now() - timedelta(days=3),
                    description=claim_description,
                    policy_number="POL-123456",
                    claimant_history=[
                        {"date": f"2023-{i:02d}-01", "amount": 5000 + i * 1000}
                        for i in range(1, num_previous_claims + 1)
                    ],
                    documents=selected_docs,
                    risk_indicators=risk_indicators
                )
                
                # Clear previous live thoughts
                st.session_state.live_thoughts = []
                
                # Run evaluations
                asyncio.run(evaluate_all_agents(claim))
        
        # Add cost-benefit analysis button
        st.markdown("---")
        if st.button("📊 Show Cost-Benefit Analysis", use_container_width=True):
            st.session_state.show_cost_benefit = True
    
    # Main content area
    if st.session_state.get('show_cost_benefit', False):
        display_cost_benefit_analysis()
        if st.button("Close Analysis"):
            st.session_state.show_cost_benefit = False
            st.rerun()
    
    # Display live thoughts if evaluation is running
    if 'live_thoughts' in st.session_state and st.session_state.live_thoughts:
        display_live_agent_thinking()
    
    # Display results
    if st.session_state.evaluation_history and not st.session_state.get('comparison_mode', False):
        display_results()


async def run_comparison(claim: InsuranceClaim):
    """Run comparison between selected agents"""
    st.session_state.comparison_results = {}
    
    # Get selected agents from comparison UI
    agent_mapping = {
        "Level 1: Reactive": 1,
        "Level 2: Assistant": 2,
        "Level 3: Autonomous": 3,
        "Level 4: Multi-Agent": 4
    }
    
    selected_agents = st.session_state.get('comparison_agents', ["Level 1: Reactive", "Level 2: Assistant"])
    
    # Progress placeholder
    progress_placeholder = st.empty()
    
    for agent_name in selected_agents:
        level = agent_mapping[agent_name]
        
        with progress_placeholder.container():
            st.info(f"🔄 Evaluating with {agent_name}...")
        
        # Clear live thoughts for this agent
        st.session_state.live_thoughts = []
        
        # Create live log callback
        def live_log_callback(log_type, content, details=None):
            add_live_thought(log_type, content, details)
            add_live_thought('agent', agent_name, details)
        
        # Run appropriate agent
        if level == 1:
            agent = ReactiveInsuranceAgent()
            result = agent.evaluate_claim(claim, live_log_callback)
        elif level == 2:
            agent = AssistantInsuranceAgent()
            result = await agent.evaluate_claim(claim, live_log_callback)
        elif level == 3:
            agent = AutonomousInsuranceAgent()
            result, _ = await agent.evaluate_claim(claim, live_log_callback)
        else:  # level == 4
            agent = MultiAgentInsuranceSystem()
            result, _ = await agent.evaluate_claim(claim, live_log_callback)
        
        st.session_state.comparison_results[agent_name] = result
    
    # Clear progress
    progress_placeholder.empty()
    st.success("✅ Comparison complete!")
    st.rerun()


async def evaluate_all_agents(claim: InsuranceClaim):
    """Run all agent evaluations with live logging"""
    st.session_state.agent_conversations = {}
    results = []
    
    # Create progress placeholder
    progress_placeholder = st.empty()
    
    # Create live log callback
    def live_log_callback(log_type, content, details=None):
        add_live_thought(log_type, content, details)
    
    # Level 1: Reactive Agent
    with progress_placeholder.container():
        st.info("🔄 Running Level 1: Reactive Agent...")
    reactive = ReactiveInsuranceAgent()
    result1 = reactive.evaluate_claim(claim, live_log_callback)
    results.append(result1)
    
    # Level 2: Assistant Agent
    with progress_placeholder.container():
        st.info("🔄 Running Level 2: Assistant Agent...")
    assistant = AssistantInsuranceAgent()
    result2 = await assistant.evaluate_claim(claim, live_log_callback)
    results.append(result2)
    
    # Level 3: Autonomous Agent
    with progress_placeholder.container():
        st.info("🔄 Running Level 3: Autonomous Agent...")
    autonomous = AutonomousInsuranceAgent()
    result3, actions = await autonomous.evaluate_claim(claim, live_log_callback)
    results.append(result3)
    
    # Level 4: Multi-Agent System
    with progress_placeholder.container():
        st.info("🔄 Running Level 4: Multi-Agent System...")
    multi_agent = MultiAgentInsuranceSystem()
    result4, log = await multi_agent.evaluate_claim(claim, live_log_callback)
    results.append(result4)
    
    # Clear progress
    progress_placeholder.empty()
    
    # Store results
    st.session_state.evaluation_history = results
    st.success("✅ All evaluations complete!")


def display_results():
    """Display evaluation results with educational insights and animations"""
    st.header("📊 Evaluation Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Overview", 
        "🤖 Agent Details", 
        "🔧 Tool Usage", 
        "💭 Reasoning Process",
        "🎬 Decision Animation",
        "📚 Learning Points",
        "🏗️ Architecture"
    ])
    
    with tab1:
        display_overview()
    
    with tab2:
        display_agent_details()
    
    with tab3:
        display_tool_usage()
    
    with tab4:
        display_reasoning()
    
    with tab5:
        display_decision_animations()
    
    with tab6:
        display_learning_points()
    
    with tab7:
        display_architecture()


def display_decision_animations():
    """Display animated decision flows (Improvement #5)"""
    st.subheader("🎬 Decision Flow Animations")
    
    # Get the claim from the first result
    if st.session_state.evaluation_history:
        first_result = st.session_state.evaluation_history[0]
        
        # Mock claim for animation
        claim = type('obj', (object,), {
            'amount': 15000,  # Use a default or extract from session
            'claim_type': ClaimType.AUTO,
            'description': 'Sample claim for animation'
        })
        
        # Agent selector for animation
        selected_agent = st.selectbox(
            "Select Agent to Animate",
            ["Level 1: Reactive", "Level 2: Assistant", "Level 3: Autonomous", "Level 4: Multi-Agent"]
        )
        
        agent_level = int(selected_agent.split(":")[0].split()[-1])
        
        # Create and display animation
        fig = create_decision_flow_animation(agent_level, claim)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional context
        if agent_level == 1:
            st.info("**Animation shows:** Simple threshold-based decision making. The claim amount directly determines the outcome.")
        elif agent_level == 2:
            st.info("**Animation shows:** Sequential tool execution. Each MCP tool is called to gather information before making a decision.")
        elif agent_level == 3:
            st.info("**Animation shows:** Pattern recognition and memory access. The agent considers historical data and proactively investigates.")
        elif agent_level == 4:
            st.info("**Animation shows:** Parallel specialist evaluation followed by consensus building or arbitration.")


def display_overview():
    """Display overview of all agent results"""
    results = st.session_state.evaluation_history
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", len(results))
    
    with col2:
        recommendations = [r.recommendation for r in results]
        most_common = max(set(recommendations), key=recommendations.count)
        st.metric("Consensus", most_common.title())
    
    with col3:
        avg_confidence = sum(r.confidence for r in results) / len(results)
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    with col4:
        avg_time = sum(r.processing_time for r in results) / len(results)
        st.metric("Avg Time", f"{avg_time:.2f}s")
    
    # Comparison chart
    df_results = pd.DataFrame([{
        'Agent': r.evaluator,
        'Recommendation': r.recommendation,
        'Confidence': r.confidence,
        'Risk Level': r.risk_level.value,
        'Processing Time': r.processing_time
    } for r in results])
    
    # Confidence comparison
    fig_confidence = px.bar(
        df_results, 
        x='Agent', 
        y='Confidence',
        title='Agent Confidence Levels',
        color='Recommendation',
        text='Confidence'
    )
    fig_confidence.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    st.plotly_chart(fig_confidence, use_container_width=True)
    
    # Processing time comparison
    fig_time = px.bar(
        df_results,
        x='Agent',
        y='Processing Time',
        title='Processing Time by Agent Level',
        text='Processing Time'
    )
    fig_time.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
    st.plotly_chart(fig_time, use_container_width=True)


def display_agent_details():
    """Display detailed information about each agent"""
    agents = [
        ReactiveInsuranceAgent(),
        AssistantInsuranceAgent(),
        AutonomousInsuranceAgent(),
        MultiAgentInsuranceSystem()
    ]
    
    for i, (agent, result) in enumerate(zip(agents, st.session_state.evaluation_history)):
        with st.expander(f"Level {i+1}: {agent.name}", expanded=i==0):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**Description:** {agent.description}")
                st.markdown("**Capabilities:**")
                for cap in agent.capabilities:
                    st.write(cap)
            
            with col2:
                st.markdown("**Evaluation Result:**")
                st.write(f"✅ Recommendation: **{result.recommendation.upper()}**")
                st.write(f"📊 Confidence: **{result.confidence:.2%}**")
                st.write(f"⚠️ Risk Level: **{result.risk_level.value.upper()}**")
                
                if result.reasons:
                    st.markdown("**Reasons:**")
                    for reason in result.reasons:
                        st.write(f"- {reason}")
                
                if result.red_flags:
                    st.markdown("**🚩 Red Flags:**")
                    for flag in result.red_flags:
                        st.write(f"- {flag}")


def display_tool_usage():
    """Display MCP tool usage across agents"""
    st.subheader("🔧 MCP Tool Usage Analysis")
    
    # Collect all tool usage
    all_tools = []
    for result in st.session_state.evaluation_history:
        if hasattr(result, 'tool_usage') and result.tool_usage:
            for tool_call in result.tool_usage:
                all_tools.append({
                    'Agent': result.evaluator,
                    'Tool': tool_call.get('tool', 'Unknown'),
                    'Status': tool_call.get('status', 'Unknown')
                })
    
    if all_tools:
        df_tools = pd.DataFrame(all_tools)
        
        # Tool usage by agent
        fig_tools = px.histogram(
            df_tools,
            x='Agent',
            color='Tool',
            title='Tool Usage by Agent',
            barmode='group'
        )
        st.plotly_chart(fig_tools, use_container_width=True)
        
        # Detailed tool calls
        for agent in df_tools['Agent'].unique():
            agent_tools = df_tools[df_tools['Agent'] == agent]
            if not agent_tools.empty:
                st.markdown(f"**{agent} Tool Calls:**")
                for _, tool in agent_tools.iterrows():
                    st.write(f"- {tool['Tool']} ({tool['Status']})")
    else:
        st.info("No tool usage recorded (Level 1 agent doesn't use tools)")


def display_reasoning():
    """Display agent reasoning and conversations"""
    st.subheader("💭 Agent Reasoning Process")
    
    conversations = st.session_state.agent_conversations
    
    for agent_name, conv_data in conversations.items():
        if 'agent_conversations' in conv_data:
            # Level 4: Multi-agent - handle differently to avoid nested expanders
            with st.expander(f"{agent_name} Reasoning", expanded=False):
                st.markdown("**Multi-Agent Deliberation:**")
                
                # Use tabs for different specialists instead of nested expanders
                specialist_names = list(conv_data['agent_conversations'].keys())
                tabs = st.tabs(specialist_names)
                
                for i, (specialist, spec_conv) in enumerate(conv_data['agent_conversations'].items()):
                    with tabs[i]:
                        if 'role' in spec_conv:
                            st.info(f"**Role:** {spec_conv['role']}")
                        if 'focus' in spec_conv:
                            st.info(f"**Focus:** {spec_conv['focus']}")
                        
                        if 'response' in spec_conv:
                            st.markdown("**Analysis:**")
                            # Use markdown instead of text_area for better display
                            st.markdown(
                                f"""<div style="max-height: 400px; overflow-y: auto; padding: 10px; 
                                background-color: #1e1e1e; border-radius: 5px; border: 1px solid #333;">
                                <pre style="white-space: pre-wrap; word-wrap: break-word; margin: 0; color: #e0e0e0;">
{spec_conv['response']}</pre>
                                </div>""", 
                                unsafe_allow_html=True
                            )
                        
                        if 'tool_usage' in spec_conv and spec_conv['tool_usage']:
                            st.markdown("**Tools Used:**")
                            for tool in spec_conv['tool_usage']:
                                if isinstance(tool, dict):
                                    tool_name = tool.get('tool', 'Unknown')
                                    tool_status = tool.get('status', 'Unknown')
                                    st.write(f"- {tool_name} ({tool_status})")
                        
                        if 'decision_process' in spec_conv:
                            st.markdown("**Decision Process:**")
                            st.markdown(
                                f"""<div style="max-height: 300px; overflow-y: auto; padding: 10px; 
                                background-color: #1e1e1e; border-radius: 5px; border: 1px solid #333;">
                                <pre style="white-space: pre-wrap; word-wrap: break-word; margin: 0; color: #e0e0e0;">
{spec_conv['decision_process']}</pre>
                                </div>""", 
                                unsafe_allow_html=True
                            )
                
                st.markdown(f"**Consensus Reached:** {'✅ Yes' if conv_data.get('consensus') else '❌ No'}")
                
                # Add evaluation summary
                if 'evaluations' in conv_data:
                    st.markdown("### 📊 Specialist Decisions Summary")
                    summary_data = []
                    for agent, eval_data in conv_data['evaluations'].items():
                        summary_data.append({
                            'Specialist': agent,
                            'Recommendation': eval_data.get('recommendation', 'N/A').upper(),
                            'Confidence': f"{eval_data.get('confidence', 0):.2%}",
                            'Risk Level': eval_data.get('risk_level', 'N/A').upper()
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, hide_index=True, use_container_width=True)
        
        else:
            # Handle other agent types (Level 1, 2, 3)
            with st.expander(f"{agent_name} Reasoning", expanded=False):
                if 'thought_process' in conv_data:
                    # Level 1: Simple rules
                    st.markdown("**Decision Process:**")
                    for rule in conv_data['thought_process']:
                        st.write(f"- {rule}")
                    
                    if 'decision_tree' in conv_data:
                        st.json(conv_data['decision_tree'])
                
                elif 'reasoning' in conv_data:
                    # Level 2 & 3: LLM reasoning
                    st.markdown("**LLM Reasoning:**")
                    if isinstance(conv_data['reasoning'], list):
                        for reasoning in conv_data['reasoning']:
                            st.text_area("Thought Process", reasoning, height=200)
                    else:
                        st.text_area("Thought Process", conv_data['reasoning'], height=200)
                    
                    if 'memory_context' in conv_data:
                        st.markdown("**Memory Context:**")
                        st.text(conv_data['memory_context'])
                    
                    if 'patterns' in conv_data:
                        st.markdown("**Learned Patterns:**")
                        st.json(conv_data['patterns'])
                    
                    if 'actions' in conv_data:
                        st.markdown("**Autonomous Actions Taken:**")
                        for action in conv_data['actions']:
                            st.write(f"- {action}")


def display_learning_points():
    """Display educational insights about agent levels"""
    st.subheader("📚 Key Learning Points")
    
    learning_points_dict = {
        "Level 1 - Reactive Agents": [
            "✅ **Pros:** Fast, deterministic, predictable",
            "❌ **Cons:** No context understanding, rigid rules, can't handle edge cases",
            "📝 **Use Case:** Simple, high-volume decisions with clear rules"
        ],
        "Level 2 - Assistant Agents": [
            "✅ **Pros:** LLM reasoning, tool usage, structured analysis",
            "❌ **Cons:** No memory, reactive only, no learning",
            "📝 **Use Case:** Complex analysis requiring reasoning but not continuity"
        ],
        "Level 3 - Autonomous Agents": [
            "✅ **Pros:** Proactive investigation, pattern learning, memory",
            "❌ **Cons:** More complex, requires state management",
            "📝 **Use Case:** Cases requiring historical context and pattern recognition"
        ],
        "Level 4 - Multi-Agent Systems": [
            "✅ **Pros:** Multiple perspectives, specialist knowledge, consensus building",
            "❌ **Cons:** Higher cost, longer processing time, coordination complexity",
            "📝 **Use Case:** High-stakes decisions requiring diverse expertise"
        ]
    }
    
    for agent_level_name, point_list in learning_points_dict.items():
        with st.expander(agent_level_name, expanded=True):
            for point in point_list:
                st.markdown(point)
    
    # MCP Protocol Benefits
    st.markdown("### 🔧 MCP Protocol Benefits")
    st.markdown("""
    - **Structured Tool Calling:** Type-safe, validated inputs and outputs
    - **Separation of Concerns:** Clear distinction between reasoning and tool execution
    - **Scalability:** Same protocol works from simple to complex agents
    - **Observability:** Easy to track and debug tool usage
    - **Extensibility:** Easy to add new tools without changing agent logic
    """)


def display_architecture():
    """Display detailed architecture information for each agent level"""
    st.subheader("🏗️ Agent Architecture Deep Dive")
    
    # Architecture selection
    selected_level = st.selectbox(
        "Select Agent Level to Explore",
        options=["Level 1: Reactive Agent", "Level 2: Assistant Agent", 
                 "Level 3: Autonomous Agent", "Level 4: Multi-Agent System"]
    )
    
    st.markdown("---")
    
    # Create the architecture diagram
    level_num = int(selected_level.split(":")[0].split()[-1])
    fig = create_architecture_diagram(level_num)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display architecture details based on selection
    if "Level 1" in selected_level:
        display_level1_architecture()
    elif "Level 2" in selected_level:
        display_level2_architecture()
    elif "Level 3" in selected_level:
        display_level3_architecture()
    elif "Level 4" in selected_level:
        display_level4_architecture()


def create_architecture_diagram(level):
    """Create architecture diagrams for different agent levels"""
    
    if level == 1:
        # Level 1: Reactive Agent - Simple Decision Tree
        fig = go.Figure()
        
        # Create a sankey diagram for the decision flow
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = ["Claim Input", "Rule Engine", "Amount Check", 
                        "High Amount (>$10k)", "Low Amount (≤$10k)", 
                        "Investigate", "Approve"],
                color = ["#64b5f6", "#fff59d", "#ffcc80", 
                        "#ef5350", "#66bb6a", 
                        "#ff7043", "#81c784"],
                x = [0, 0.3, 0.5, 0.7, 0.7, 1, 1],
                y = [0.5, 0.5, 0.5, 0.3, 0.7, 0.3, 0.7]
            ),
            link = dict(
                source = [0, 1, 2, 2, 3, 4],
                target = [1, 2, 3, 4, 5, 6],
                value = [1, 1, 0.3, 0.7, 0.3, 0.7],
                color = ["rgba(100, 181, 246, 0.4)", "rgba(255, 245, 157, 0.4)", 
                        "rgba(239, 83, 80, 0.4)", "rgba(102, 187, 106, 0.4)",
                        "rgba(255, 112, 67, 0.4)", "rgba(129, 199, 132, 0.4)"]
            )
        )])
        
        fig.update_layout(
            title="Level 1: Reactive Agent - Rule-Based Decision Flow",
            font_size=12,
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    elif level == 2:
        # Level 2: Assistant Agent - Tool Flow
        fig = go.Figure()
        
        # Central LLM node
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            text=['LLM<br>(Claude)'],
            textposition="middle center",
            marker=dict(size=80, color='#ffd54f', line=dict(width=3, color='white')),
            showlegend=False
        ))
        
        # MCP Tools
        tools = ['Check Policy', 'Risk Score', 'Verify Docs', 'Fraud Check']
        angles = [0, 90, 180, 270]
        radius = 1.5
        
        for i, (tool, angle) in enumerate(zip(tools, angles)):
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            
            # Tool node
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[tool],
                textposition="top center" if y > 0 else "bottom center",
                marker=dict(size=50, color='#90caf9', line=dict(width=2, color='white')),
                showlegend=False
            ))
            
            # Connection line
            fig.add_trace(go.Scatter(
                x=[0, x], y=[0, y],
                mode='lines',
                line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash'),
                showlegend=False
            ))
        
        # Input/Output
        fig.add_trace(go.Scatter(
            x=[0], y=[2.5],
            mode='markers+text',
            text=['Claim Input'],
            textposition="top center",
            marker=dict(size=40, color='#a5d6a7', symbol='square'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[0], y=[-2.5],
            mode='markers+text',
            text=['Decision Output'],
            textposition="bottom center",
            marker=dict(size=40, color='#ef9a9a', symbol='square'),
            showlegend=False
        ))
        
        # Arrows
        fig.add_annotation(x=0, y=2.5, ax=0, ay=0.8,
                          xref="x", yref="y", axref="x", ayref="y",
                          arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#a5d6a7")
        
        fig.add_annotation(x=0, y=-0.8, ax=0, ay=-2.5,
                          xref="x", yref="y", axref="x", ayref="y",
                          arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#ef9a9a")
        
        fig.update_layout(
            title="Level 2: Assistant Agent - LLM with MCP Tools",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    elif level == 3:
        # Level 3: Autonomous Agent - Circular Flow with Memory
        fig = go.Figure()
        
        # Create circular layout
        n_nodes = 8
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        radius = 2
        center_x, center_y = 0, 0
        
        nodes = [
            ('Claim Input', '#81c784'),
            ('Memory Recall', '#ba68c8'),
            ('Pattern Analysis', '#9575cd'),
            ('LLM Decision', '#ffd54f'),
            ('Confidence Check', '#ff8a65'),
            ('Tool Execution', '#4dd0e1'),
            ('Memory Update', '#f06292'),
            ('Output/Escalate', '#aed581')
        ]
        
        x_pos = center_x + radius * np.cos(angles)
        y_pos = center_y + radius * np.sin(angles)
        
        # Draw circular flow
        for i in range(n_nodes):
            next_i = (i + 1) % n_nodes
            fig.add_trace(go.Scatter(
                x=[x_pos[i], x_pos[next_i]],
                y=[y_pos[i], y_pos[next_i]],
                mode='lines',
                line=dict(color='rgba(255,255,255,0.3)', width=2),
                showlegend=False
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            text=[node[0] for node in nodes],
            textposition=["top center", "middle left", "middle left", "bottom center",
                         "bottom center", "middle right", "middle right", "top center"],
            marker=dict(
                size=50,
                color=[node[1] for node in nodes],
                line=dict(width=2, color='white')
            ),
            showlegend=False
        ))
        
        # Add center node
        fig.add_trace(go.Scatter(
            x=[center_x], y=[center_y],
            mode='markers+text',
            text=['Autonomous<br>Agent Core'],
            textposition="middle center",
            marker=dict(size=80, color='#37474f', line=dict(width=3, color='white')),
            showlegend=False
        ))
        
        # Add memory database
        fig.add_trace(go.Scatter(
            x=[3], y=[2],
            mode='markers+text',
            text=['Pattern<br>Database'],
            textposition="middle center",
            marker=dict(size=60, color='#4caf50', symbol='square', line=dict(width=2, color='white')),
            showlegend=False
        ))
        
        # Connection to memory
        fig.add_trace(go.Scatter(
            x=[x_pos[1], 3, x_pos[2]],
            y=[y_pos[1], 2, y_pos[2]],
            mode='lines',
            line=dict(color='rgba(76, 175, 80, 0.5)', width=4, dash='dash'),
            showlegend=False
        ))
        
        fig.update_layout(
            title="Level 3: Autonomous Agent - Self-Directed Learning Loop",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 4]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    else:  # Level 4
        # Level 4: Multi-Agent System - Hierarchy
        fig = go.Figure()
        
        # Define positions
        positions = {
            'coordinator': (0, 3, '#ffd54f'),
            'fraud': (-2, 1.5, '#ef5350'),
            'risk': (0, 1.5, '#42a5f5'),
            'customer': (2, 1.5, '#66bb6a'),
            'consensus': (0, 0, '#ff7043'),
            'arbitrator': (0, -1.5, '#ff6f00'),
            'output': (0, -3, '#9ccc65')
        }
        
        labels = {
            'coordinator': 'Agent<br>Coordinator',
            'fraud': 'Fraud<br>Specialist',
            'risk': 'Risk<br>Analyst',
            'customer': 'Customer<br>Advocate',
            'consensus': 'Consensus<br>Engine',
            'arbitrator': 'Senior<br>Arbitrator',
            'output': 'Final<br>Decision'
        }
        
        # Draw connections
        connections = [
            ('coordinator', 'fraud'),
            ('coordinator', 'risk'),
            ('coordinator', 'customer'),
            ('fraud', 'consensus'),
            ('risk', 'consensus'),
            ('customer', 'consensus'),
            ('consensus', 'output'),
            ('consensus', 'arbitrator'),
            ('arbitrator', 'output')
        ]
        
        for start, end in connections:
            x0, y0, _ = positions[start]
            x1, y1, _ = positions[end]
            
            dash = 'dash' if start == 'consensus' and end == 'arbitrator' else 'solid'
            color = 'rgba(255, 99, 71, 0.8)' if dash == 'dash' else 'rgba(255, 255, 255, 0.4)'
            
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(color=color, width=3, dash=dash),
                showlegend=False
            ))
        
        # Add nodes
        for node, (x, y, color) in positions.items():
            size = 80 if node == 'coordinator' else 60
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[labels[node]],
                textposition="middle center",
                marker=dict(size=size, color=color, line=dict(width=3, color='white')),
                showlegend=False
            ))
        
        # Add MCP tools
        fig.add_trace(go.Scatter(
            x=[-3.5], y=[1.5],
            mode='markers+text',
            text=['MCP<br>Tools'],
            textposition="middle center",
            marker=dict(size=60, color='#b0bec5', symbol='diamond', line=dict(width=2, color='white')),
            showlegend=False
        ))
        
        # Tool connections
        for agent in ['fraud', 'risk', 'customer']:
            x1, y1, _ = positions[agent]
            fig.add_trace(go.Scatter(
                x=[-3.5, x1], y=[1.5, y1],
                mode='lines',
                line=dict(color='rgba(176, 190, 197, 0.3)', width=2, dash='dot'),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Level 4: Multi-Agent System - Collaborative Intelligence",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 4]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4, 4]),
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            annotations=[
                dict(x=0, y=-3.7, text="❗ Red dash = No consensus → Arbitration",
                     showarrow=False, font=dict(size=10, color='#ff6b6b'))
            ]
        )
        
        return fig


def display_level1_architecture():
    """Display Level 1 Reactive Agent architecture details"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Reactive Agent Architecture")
        st.markdown("""
        **Core Components:**
        - **Decision Engine**: Simple if-then-else rules
        - **Input Parser**: Extracts claim amount
        - **Output Generator**: Fixed response templates
        
        **Data Flow:**
        1. Receive claim input
        2. Extract key parameter (amount)
        3. Apply rule threshold
        4. Return fixed response
        """)
        
        # Show example rule implementation
        st.code("""
# Simplified Rule Engine
def evaluate(claim):
    if claim.amount > THRESHOLD:
        return {
            "decision": "investigate",
            "risk": "high"
        }
    else:
        return {
            "decision": "approve",
            "risk": "low"
        }
        """, language="python")
    
    with col2:
        st.info("""
        **Key Characteristics:**
        - No external dependencies
        - Deterministic outcomes
        - Millisecond response times
        - No learning capability
        """)
        
        # Performance metrics
        st.markdown("### Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Response Time', 'Accuracy', 'Cost per Decision', 'Scalability'],
            'Value': ['<10ms', '~60%', '$0.001', '1M+ requests/hour']
        })
        st.dataframe(metrics_df, hide_index=True)


def display_level2_architecture():
    """Display Level 2 Assistant Agent architecture details"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🤖 Assistant Agent Architecture")
        st.markdown("""
        **Core Components:**
        - **LLM Interface**: Claude API integration
        - **MCP Tool Registry**: Available tools catalog
        - **Tool Executor**: Handles tool calls
        - **Response Parser**: Extracts decisions from LLM
        
        **MCP Protocol Flow:**
        1. Register available tools with LLM
        2. LLM decides which tools to use
        3. Execute tools with validated inputs
        4. Return results to LLM
        5. LLM synthesizes final decision
        """)
        
        # Show MCP tool definition
        st.code("""
# MCP Tool Definition
Tool(
    name="calculate_risk_score",
    description="Calculate risk score",
    inputSchema={
        "type": "object",
        "properties": {
            "claim_amount": {"type": "number"},
            "claim_history": {"type": "integer"}
        },
        "required": ["claim_amount"]
    }
)
        """, language="python")
    
    with col2:
        st.success("""
        **MCP Benefits:**
        - Type-safe tool calls
        - Structured inputs/outputs
        - Clear separation of concerns
        - Easy to add new tools
        """)
        
        # Tool capabilities
        st.markdown("### Available MCP Tools")
        tools_df = pd.DataFrame({
            'Tool': ['check_policy_coverage', 'calculate_risk_score', 'verify_documents', 'check_fraud_patterns'],
            'Purpose': ['Verify coverage', 'Assess risk level', 'Check completeness', 'Detect fraud']
        })
        st.dataframe(tools_df, hide_index=True)


def display_level3_architecture():
    """Display Level 3 Autonomous Agent architecture details"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🧠 Autonomous Agent Architecture")
        st.markdown("""
        **Core Components:**
        - **Memory Store**: Persistent claim history
        - **Pattern Recognizer**: Identifies trends
        - **Proactive Investigator**: Self-directed tools
        - **Confidence Evaluator**: Self-assessment
        - **Escalation Manager**: Senior review trigger
        
        **Advanced Features:**
        1. **Working Memory**: Recent claims context
        2. **Long-term Memory**: Pattern database
        3. **Meta-cognition**: Confidence tracking
        4. **Autonomous Actions**: Self-directed investigation
        """)
        
        # Show memory structure
        st.code("""
# Memory and Pattern Structure
self.memory = [{
    "timestamp": datetime,
    "claim_id": str,
    "type": str,
    "amount": float,
    "outcome": str
}]

self.patterns = {
    "auto_high_risk": {
        "count": 15,
        "outcomes": {
            "approve": 2,
            "deny": 3,
            "investigate": 10
        }
    }
}
        """, language="python")
    
    with col2:
        st.warning("""
        **Autonomous Capabilities:**
        - Self-directed investigation
        - Pattern learning over time
        - Confidence-based escalation
        - Context-aware decisions
        """)
        
        # Learning metrics
        st.markdown("### Learning Metrics")
        learning_df = pd.DataFrame({
            'Metric': ['Patterns Recognized', 'Accuracy Improvement', 'Escalation Rate', 'Investigation Triggers'],
            'Value': ['15+ patterns', '+15% over time', '8% of cases', '25% proactive']
        })
        st.dataframe(learning_df, hide_index=True)


def display_level4_architecture():
    """Display Level 4 Multi-Agent System architecture details"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🌐 Multi-Agent System Architecture")
        st.markdown("""
        **Core Components:**
        - **Agent Registry**: Specialist catalog
        - **Communication Bus**: Inter-agent messaging
        - **Consensus Engine**: Decision aggregation
        - **Arbitration System**: Conflict resolution
        - **Coordination Layer**: Workflow management
        
        **Specialist Agents:**
        1. **Fraud Specialist**: Deep fraud analysis
        2. **Risk Analyst**: Comprehensive risk assessment
        3. **Customer Advocate**: Fair treatment focus
        4. **Senior Arbitrator**: Final decisions
        
        **Consensus Mechanisms:**
        - Majority voting
        - Confidence weighting
        - Specialist expertise ranking
        - Escalation protocols
        """)
        
        # Show multi-agent communication
        st.code("""
# Multi-Agent Communication
evaluations = {
    "Fraud Specialist": {
        "recommendation": "investigate",
        "confidence": 0.9,
        "risk": "high"
    },
    "Risk Analyst": {
        "recommendation": "investigate", 
        "confidence": 0.8,
        "risk": "high"
    },
    "Customer Advocate": {
        "recommendation": "approve",
        "confidence": 0.6,
        "risk": "medium"
    }
}

# Consensus: 2/3 vote for "investigate"
# Arbitration: Not needed (majority)
        """, language="python")
    
    with col2:
        st.error("""
        **System Complexity:**
        - Multiple parallel LLM calls
        - Complex coordination logic
        - Higher latency (3-4x)
        - Richer decision context
        """)
        
        # Consensus metrics
        st.markdown("### Consensus Statistics")
        consensus_df = pd.DataFrame({
            'Metric': ['Consensus Rate', 'Arbitration Needed', 'Avg Specialists', 'Decision Time'],
            'Value': ['82%', '18%', '3-4 agents', '8-10 seconds']
        })
        st.dataframe(consensus_df, hide_index=True)


if __name__ == "__main__":
    main()