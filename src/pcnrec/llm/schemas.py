from pydantic import BaseModel, Field, conlist, ConfigDict
from typing import List, Dict, Optional, Any

class ComputedStats(BaseModel):
    head_count: int
    tail_count: int
    unique_genres: int
    entropy_proxy: Optional[float] = None
    model_config = ConfigDict(extra='forbid')

class ConstraintChecks(BaseModel):
    popularity: bool
    diversity: bool
    safety: bool
    model_config = ConfigDict(extra='forbid')

class PopularityConfig(BaseModel):
    max_head_in_topn: Optional[int] = None
    min_tail_in_topn: Optional[int] = None
    model_config = ConfigDict(extra='forbid')

class DiversityConfig(BaseModel):
    min_unique_genres_in_topn: Optional[int] = None
    model_config = ConfigDict(extra='forbid')

class SafetyConfig(BaseModel):
    no_duplicates: Optional[bool] = None
    model_config = ConfigDict(extra='forbid')

class ConstraintsConfig(BaseModel):
    popularity: Optional[PopularityConfig] = None
    diversity: Optional[DiversityConfig] = None
    safety: Optional[SafetyConfig] = None
    model_config = ConfigDict(extra='forbid')

class RecommendationSelection(BaseModel):
    selected_item_ids: List[int] = Field(..., description="List of internal item IDs selected.")
    rationale: str = Field(..., description="Short explanation of the selection strategy.")
    computed_stats: ComputedStats = Field(..., description="Statistics computed by the agent.")
    constraint_checks_claimed: ConstraintChecks = Field(..., description="Agent's self-reported constraint checks.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_config = ConfigDict(extra='forbid')

class NegotiationRound(BaseModel):
    round_id: int
    user_advocate_summary: str
    platform_policy_summary: str
    mediator_decision: str
    model_config = ConfigDict(extra='forbid')

class ProofCertificate(BaseModel):
    version: str = "pcnrec-v0.1"
    constraints: ConstraintsConfig = Field(..., description="Constraints configuration used.")
    selected_item_ids: List[int]
    computed_stats_claimed: ComputedStats
    negotiation_trace: List[NegotiationRound]
    signature: str = Field(..., description="Non-cryptographic hash showing traceability.")
    model_config = ConfigDict(extra='forbid')
