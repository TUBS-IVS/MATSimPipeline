from typing import Tuple, NamedTuple
from frozendict import frozendict
import numpy as np

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
SegmentedPlans = frozendict[str, SegmentedPlan]  # Many agents' plans (person_id -> SegmentedPlan)
Households = frozendict[str, SegmentedPlans]

DetailedSegment = Tuple[DetailedLeg, ...]  # A segment of a plan (immutable tuple of legs)
DetailedSegmentedPlan = Tuple[DetailedSegment, ...]  # A full plan split into segments
DetailedSegmentedPlans = frozendict[str, DetailedSegmentedPlan]  # All agents' plans (person_id -> SegmentedPlan)
