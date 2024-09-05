from typing import Dict, List, Any

# Type definitions
PlanLeg = Dict[str, Any]  # A single leg of a plan (keys are: from_location, to_location, distance, ...)

PlanSegment = List[PlanLeg]  # A segment of an agent's plan with known start and end locations.
SegmentedPlan = List[PlanSegment]  # Whole plan, split into segments (there can just be one segment). Used by secondary location algos.
SegmentedPlans = Dict[str, SegmentedPlan]  # All agents' plans (key = person ID)

UnSegmentedPlan = List[PlanLeg]  # Whole plan, not split into segments. Used by main location algos.
UnSegmentedPlans = Dict[str, UnSegmentedPlan]  # All agents' plans (key = person ID)
