"""Progress UI helpers for long-running operations."""

from dataclasses import dataclass


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


@dataclass
class ProgressDisplay:
    title: str
    width: int = 70
    bar_length: int = 30
    item_label: str = "items"
    detail_label: str = "Processing"
    min_percent_delta: int = 1
    _started: bool = False
    _last_percent: int = -1
    _last_current: int = 0
    _last_total: int = 0

    def start(self) -> None:
        print("\n" + "-" * self.width)
        print(self.title.center(self.width))
        print("-" * self.width)
        self._started = True
        self._last_percent = -1

    def _format_line(self, bar: str, percent: int, current: int, total: int, detail_text: str) -> str:
        prefix = f"  [{bar}] {percent:3d}% | {current}/{total} {self.item_label} | {self.detail_label}: "
        max_detail = max(self.width - len(prefix), 0)
        detail_text = _truncate(detail_text, max_detail)
        return prefix + detail_text

    def update(self, current: int, total: int, detail: str = "") -> None:
        if not self._started:
            self.start()

        percent = int((current / total) * 100) if total > 0 else 0
        if percent == self._last_percent and current != total:
            return
        if (percent - self._last_percent) < self.min_percent_delta and current != total:
            return

        self._last_percent = percent
        self._last_current = current
        self._last_total = total
        filled = int((self.bar_length * current) / total) if total > 0 else 0
        bar = "#" * filled + "-" * (self.bar_length - filled)
        detail_text = detail if detail else "..."
        print(self._format_line(bar, percent, current, total, detail_text))

    def finish(self, message: str) -> None:
        if not self._started:
            self.start()

        bar = "#" * self.bar_length
        current = self._last_current if self._last_total else 1
        total = self._last_total if self._last_total else 1
        line = self._format_line(bar, 100, current, total, message)
        print(line)
        print("-" * self.width)
