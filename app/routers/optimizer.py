from fastapi import APIRouter, HTTPException
from app.models.optimizer import OptimizeRequest, OptimizeResponse
from app.optimizer.lp_optimizer import UniversalLPOptimizer
import time

router = APIRouter(prefix="/api/v2", tags=["optimizer"])


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_lineups(request: OptimizeRequest):
    """
    Generate optimized DFS lineups using Linear Programming.

    Performance: 5-10s for 25 lineups (4 vCPUs, no cold starts)
    """
    start_time = time.time()

    try:
        optimizer = UniversalLPOptimizer(
            players=request.players,
            options=request.options,
            game=request.game,
            exposure=request.exposure,
            num_solutions=request.num_solutions
        )

        lineups = optimizer.lineups_with_player_positions()

        return OptimizeResponse(
            lineups=[lineup.to_dict() for lineup in lineups],
            execution_time=time.time() - start_time,
            num_lineups=len(lineups),
            success=True
        )

    except Exception as e:
        return OptimizeResponse(
            lineups=[],
            execution_time=time.time() - start_time,
            num_lineups=0,
            success=False,
            error=str(e)
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "dfs-api"}