from pydantic import BaseModel, Field, conlist
from typing import List, Dict, Optional, Any

class ComputedStats(BaseModel):
    head_count: int
    tail_count: int
    unique_genres: int
    entropy_proxy: Optional[float] = None

class RecommendationSelection(BaseModel):
    selected_item_ids: List[int] = Field(..., description="List of internal item IDs selected.")
    rationale: str = Field(..., description="Short explanation of the selection strategy.")
    computed_stats: ComputedStats = Field(..., description="Statistics computed by the agent.")
    constraint_checks_claimed: Dict[str, bool] = Field(..., description="Agent's self-reported constraint checks.")
    confidence: float = Field(..., ge=0.0, le=1.0)

class NegotiationRound(BaseModel):
    round_id: int
    user_advocate_summary: str
    platform_policy_summary: str
    mediator_decision: str

class ProofCertificate(BaseModel):
    version: str = "pcnrec-v0.1"
    constraints: Dict[str, Any] = Field(..., description="Constraints configuration used.")
    selected_item_ids: List[int]
    computed_stats_claimed: ComputedStats
    negotiation_trace: List[NegotiationRound]
    signature: str = Field(..., description="Non-cryptographic hash showing traceability.")
