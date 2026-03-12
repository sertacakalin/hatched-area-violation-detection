"""Araç durumu state machine — OUTSIDE → ENTERING → INSIDE → VIOLATION."""

import logging
from dataclasses import dataclass, field

from src.core.data_models import VehicleState

logger = logging.getLogger(__name__)


@dataclass
class VehicleTrackState:
    """Tek bir aracın taralı alan durumu."""
    track_id: int
    state: VehicleState = VehicleState.OUTSIDE
    frames_in_zone: int = 0
    frames_outside_zone: int = 0
    violation_triggered: bool = False
    cooldown_remaining: int = 0
    zone_id: str | None = None


class VehicleStateMachine:
    """Her araç için durum geçişlerini yöneten state machine.

    Geçişler:
        OUTSIDE → ENTERING : Araç bölgeye girmeye başladığında (1 kare)
        ENTERING → INSIDE  : Araç bölgede kalmaya devam ederse (2+ kare)
        INSIDE → VIOLATION  : Araç min_frames_in_zone kadar bölgede kalırsa
        * → OUTSIDE          : Araç bölgeden çıktığında
    """

    def __init__(self, min_frames_in_zone: int = 5,
                 cooldown_frames: int = 90,
                 exit_frames: int = 3):
        self.min_frames_in_zone = min_frames_in_zone
        self.cooldown_frames = cooldown_frames
        self.exit_frames = exit_frames  # Bölgeden çıkmak için gereken kare
        self._states: dict[int, VehicleTrackState] = {}

    def get_state(self, track_id: int) -> VehicleTrackState:
        if track_id not in self._states:
            self._states[track_id] = VehicleTrackState(track_id=track_id)
        return self._states[track_id]

    def update(self, track_id: int, is_in_zone: bool,
               zone_id: str | None = None) -> tuple[VehicleState, bool]:
        """Aracın durumunu güncelle.

        Returns:
            (yeni_durum, yeni_ihlal_tetiklendi)
        """
        state = self.get_state(track_id)
        new_violation = False

        # Cooldown azalt
        if state.cooldown_remaining > 0:
            state.cooldown_remaining -= 1

        if is_in_zone:
            state.frames_in_zone += 1
            state.frames_outside_zone = 0
            state.zone_id = zone_id

            if state.state == VehicleState.OUTSIDE:
                state.state = VehicleState.ENTERING
            elif state.state == VehicleState.ENTERING:
                state.state = VehicleState.INSIDE
            elif state.state == VehicleState.INSIDE:
                if (state.frames_in_zone >= self.min_frames_in_zone
                        and state.cooldown_remaining == 0):
                    state.state = VehicleState.VIOLATION
                    state.violation_triggered = True
                    state.cooldown_remaining = self.cooldown_frames
                    new_violation = True
            # VIOLATION durumunda kalabilir (zaten bölgede)

        else:
            state.frames_outside_zone += 1

            if state.frames_outside_zone >= self.exit_frames:
                state.state = VehicleState.OUTSIDE
                state.frames_in_zone = 0
                state.zone_id = None

        return state.state, new_violation

    def cleanup_stale_tracks(self, active_track_ids: set[int]) -> None:
        """Artık takip edilmeyen araçların durumlarını temizle."""
        stale_ids = set(self._states.keys()) - active_track_ids
        for tid in stale_ids:
            del self._states[tid]

    def reset(self) -> None:
        self._states.clear()
