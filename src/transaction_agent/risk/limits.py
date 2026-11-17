"""Risk limits that sit between a strategy's signals and the broker."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    """Limits applied to every trade. The defaults apply no limits at all."""

    # Share of available cash put into each new position.
    position_fraction: float = 1.0
    # Sell when the price falls this fraction below the entry price, e.g. 0.05.
    stop_loss: float | None = None
    # Kill switch: when the account falls this fraction below its highest value,
    # sell everything and stop trading for good.
    max_drawdown: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.position_fraction <= 1:
            raise ValueError("position_fraction must be in (0, 1]")
        for name in ("stop_loss", "max_drawdown"):
            value = getattr(self, name)
            if value is not None and not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1")

    def stop_price(self, entry_price: float) -> float | None:
        if self.stop_loss is None:
            return None
        return entry_price * (1 - self.stop_loss)


class DrawdownGuard:
    """Tracks the account's highest value and reports when the kill switch should fire."""

    def __init__(self, max_drawdown: float | None):
        self._max_drawdown = max_drawdown
        self._peak = 0.0

    def tripped(self, equity: float) -> bool:
        self._peak = max(self._peak, equity)
        if self._max_drawdown is None:
            return False
        return equity <= self._peak * (1 - self._max_drawdown)
