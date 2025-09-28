from fastapi import APIRouter, HTTPException
from app.models.optimizer import OptimizeRequest, OptimizeResponse
from app.optimizer.lp_optimizer import UniversalLPOptimizer
import time

router = APIRouter(prefix="/api/v2", tags=["optimizer"])


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_lineups(request: OptimizeRequest):
    """
    Generate optimized DFS lineups using Linear Programming.
    Returns Django-compatible format.
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

        # Convert to Django format: array of lineup arrays
        lineup_data = [lineup.to_dict(django_format=True) for lineup in lineups]

        return OptimizeResponse(
            data=lineup_data,
            status=0,
            datatype="optimizer",
            maxuid=0,
            deleteBeforeUpdate=1,
            execution_time=time.time() - start_time,
            num_lineups=len(lineups)
        )

    except Exception as e:
        # Log the specific error for debugging
        import traceback
        print(f"LP Optimizer Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

        # Return empty data on error with same format
        return OptimizeResponse(
            data=[],
            status=1,  # Error status
            datatype="optimizer",
            maxuid=0,
            deleteBeforeUpdate=1,
            execution_time=time.time() - start_time,
            num_lineups=0
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "dfs-api"}