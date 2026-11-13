"""Domain services — business rules isolated from framework/infrastructure."""
from __future__ import annotations

from ml_sentinel.core.errors import QualityGateError
from ml_sentinel.domain.entities import ModelMetrics


class PromotionPolicy:
    def __init__(self, policy: dict) -> None:
        self.policy = policy

    def evaluate(self, challenger: ModelMetrics, champion: ModelMetrics | None) -> tuple[bool, str]:
        if champion is None:
            return True, "no_champion_promoted"

        primary = self.policy.get("primary_metric", "f1")
        min_imp = float(self.policy.get("min_primary_metric_improvement", 0.0))
        drops = self.policy.get("max_secondary_metric_drops", {})

        challenger_primary = challenger.as_dict().get(primary, 0.0)
        champion_primary = champion.as_dict().get(primary, 0.0)
        if challenger_primary - champion_primary < min_imp:
            return False, f"primary metric {primary} improvement {challenger_primary - champion_primary:.4f} < {min_imp}"

        for metric, max_drop in drops.items():
            c_val = challenger.as_dict().get(metric)
            ch_val = champion.as_dict().get(metric)
            if c_val is None or ch_val is None:
                continue
            if ch_val - c_val > float(max_drop):
                return False, f"{metric} dropped {ch_val - c_val:.4f} > {max_drop}"

        return True, "passed"

    def enforce(self, challenger: ModelMetrics, champion: ModelMetrics | None) -> None:
        ok, reason = self.evaluate(challenger, champion)
        if not ok:
            raise QualityGateError(reason)
