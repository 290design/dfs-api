from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class OptimizeRequest(BaseModel):
    players: List[Dict[str, Any]] = Field(..., description="List of player objects")
    options: Dict[str, Any] = Field(..., description="Optimizer options")
    game: Dict[str, Any] = Field(..., description="Game configuration")
    exposure: Dict[str, float] = Field(default_factory=dict, description="Per-player exposure targets")
    num_solutions: int = Field(default=25, ge=1, le=100, description="Number of lineups to generate")


class LineupPlayer(BaseModel):
    player_id: int
    name: str
    salary: int
    position: str
    team: str
    value: float


class Lineup(BaseModel):
    players: List[LineupPlayer]
    total_salary: int
    total_value: float


class OptimizeResponse(BaseModel):
    data: List[List[Dict[str, Any]]]  # Array of lineup arrays
    status: int = 0
    datatype: str = "optimizer"
    maxuid: int = 0
    deleteBeforeUpdate: int = 1
    execution_time: Optional[float] = None
    num_lineups: Optional[int] = None