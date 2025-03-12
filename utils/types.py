from typing import Tuple, NamedTuple
from frozendict import frozendict
import numpy as np

# Old type definitions
# PlanLeg = Dict[str, Any]  # A single leg of a plan (keys are: from_location, to_location, distance, ...)
#
# PlanSegment = List[PlanLeg]  # A segment of an agent's plan with known start and end locations.
# SegmentedPlan = List[PlanSegment]  # Whole plan, split into segments (there can just be one segment). Used by secondary location algos.
# SegmentedPlans = Dict[str, SegmentedPlan]  # All agents' plans (key = person ID)
#
# UnSegmentedPlan = List[PlanLeg]  # Whole plan, not split into segments. Used by main location algos.
# UnSegmentedPlans = Dict[str, UnSegmentedPlan]  # All agents' plans (key = person ID)


class Leg(NamedTuple):
    unique_leg_id: str
    from_location: np.ndarray
    to_location: np.ndarray
    distance: float
    to_act_type: str
    to_act_identifier: str


# Details are later also available in the unified df. This is just for algos that need it.
class DetailedLeg(NamedTuple):
    unique_leg_id: str
    from_location: np.ndarray
    to_location: np.ndarray
    distance: float
    to_act_type: str
    to_act_identifier: str
    mode: str
    is_main_activity: bool
    mirrors_main_activity: bool
    home_to_main_distance: float


Segment = Tuple[Leg, ...]  # A segment of a plan (immutable tuple of legs)
SegmentedPlan = Tuple[Segment, ...]  # A full plan split into segments
SegmentedPlans = frozendict[str, SegmentedPlan]  # All agents' plans (person_id -> SegmentedPlan)

DetailedSegment = Tuple[DetailedLeg, ...]  # A segment of a plan (immutable tuple of legs)
DetailedSegmentedPlan = Tuple[DetailedSegment, ...]  # A full plan split into segments
DetailedSegmentedPlans = frozendict[str, DetailedSegmentedPlan]  # All agents' plans (person_id -> SegmentedPlan)
