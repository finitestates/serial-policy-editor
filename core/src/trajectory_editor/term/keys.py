"""Decode terminal input into key, paste, and mouse events."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Key:
    """A key press. ``name`` is e.g. ``"a"``, ``"enter"``, ``"ctrl+left"``."""

    name: str
    char: str | None = None


@dataclass(frozen=True)
class Paste:
    text: str


@dataclass(frozen=True)
class Mouse:
    """A mouse press, release, or wheel step at zero-based cell (x, y)."""

    kind: str  # "press", "release", "wheel_up", "wheel_down"
    x: int
    y: int
    button: int = 0


Event = Key | Paste | Mouse

_PASTE_START = "\x1b[200~"
_PASTE_END = "\x1b[201~"

_TILDE_KEYS = {
    1: "home", 2: "insert", 3: "delete", 4: "end", 5: "pageup", 6: "pagedown",
    7: "home", 8: "end", 11: "f1", 12: "f2", 13: "f3", 14: "f4", 15: "f5",
    17: "f6", 18: "f7", 19: "f8", 20: "f9", 21: "f10", 23: "f11", 24: "f12",
}
_LETTER_KEYS = {
    "A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end",
    "P": "f1", "Q": "f2", "R": "f3", "S": "f4", "E": "begin",
}
_CONTROL_KEYS = {
    "\r": "enter", "\n": "enter", "\t": "tab", "\x7f": "backspace", "\x08": "backspace",
    "\x00": "ctrl+space",
}


def _modified(name: str, modifier: int) -> str:
    bits = max(0, modifier - 1)
    prefix = ""
    if bits & 4:
        prefix += "ctrl+"
    if bits & 2:
        prefix += "alt+"
    if bits & 1:
        prefix += "shift+"
    return prefix + name


def _control_key(character: str) -> Key:
    if character in _CONTROL_KEYS:
        return Key(_CONTROL_KEYS[character])
    code = ord(character)
    if 1 <= code <= 26:
        return Key("ctrl+" + chr(code + 96))
    return Key({0x1C: "ctrl+backslash", 0x1D: "ctrl+]", 0x1E: "ctrl+^", 0x1F: "ctrl+_"}.get(code, "unknown"))


class InputParser:
    """Incremental parser. Feed decoded text; call :meth:`flush` on timeout.

    A lone ESC is ambiguous until either more bytes arrive or the caller's
    short timeout expires, so it stays pending until then.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._paste: list[str] | None = None

    @property
    def pending(self) -> bool:
        return bool(self._buffer) and self._paste is None

    @property
    def in_paste(self) -> bool:
        return self._paste is not None

    def feed(self, data: str) -> list[Event]:
        self._buffer += data
        return self._parse(final=False)

    def flush(self) -> list[Event]:
        """Resolve a pending incomplete sequence after the escape timeout."""
        return self._parse(final=True)

    def _parse(self, *, final: bool) -> list[Event]:
        events: list[Event] = []
        buffer = self._buffer
        index = 0
        while index < len(buffer):
            if self._paste is not None:
                end = buffer.find(_PASTE_END, index)
                if end < 0:
                    # Keep a possible partial terminator for the next feed.
                    keep = 0
                    for size in range(len(_PASTE_END) - 1, 0, -1):
                        if buffer.endswith(_PASTE_END[:size]):
                            keep = size
                            break
                    self._paste.append(buffer[index:len(buffer) - keep])
                    index = len(buffer) - keep
                    break
                self._paste.append(buffer[index:end])
                events.append(Paste("".join(self._paste).replace("\r\n", "\n").replace("\r", "\n")))
                self._paste = None
                index = end + len(_PASTE_END)
                continue
            character = buffer[index]
            if character != "\x1b":
                if character < " " or character == "\x7f":
                    events.append(_control_key(character))
                else:
                    events.append(Key(character, character))
                index += 1
                continue
            consumed, event = self._escape(buffer, index, final=final)
            if consumed == 0:
                break
            if event is not None:
                events.append(event)
            index += consumed
        self._buffer = buffer[index:]
        return events

    def _escape(self, buffer: str, index: int, *, final: bool) -> tuple[int, Event | None]:
        """Parse an escape sequence at ``index``; (0, None) means incomplete."""
        rest = buffer[index + 1:]
        if not rest:
            return (1, Key("escape")) if final else (0, None)
        lead = rest[0]
        if lead == "[":
            if buffer.startswith(_PASTE_START, index):
                self._paste = []
                return len(_PASTE_START), None
            if _PASTE_START.startswith(buffer[index:]) and not final:
                return 0, None
            # CSI: parameter bytes 0x30-0x3F, intermediates 0x20-0x2F, final 0x40-0x7E.
            position = index + 2
            while position < len(buffer) and "\x20" <= buffer[position] <= "\x3f":
                position += 1
            if position >= len(buffer):
                return (1, Key("escape")) if final else (0, None)
            final_byte = buffer[position]
            if not "\x40" <= final_byte <= "\x7e":
                return 1, Key("escape")
            params = buffer[index + 2:position]
            return position + 1 - index, self._csi(params, final_byte)
        if lead == "O":
            if len(rest) < 2:
                return (1, Key("escape")) if final else (0, None)
            name = _LETTER_KEYS.get(rest[1])
            return 3, Key(name) if name else None
        if lead == "\x1b":
            return 1, Key("escape")
        # ESC + key is Alt+key.
        if lead < " " or lead == "\x7f":
            inner = _control_key(lead)
            return 2, Key("alt+" + inner.name)
        return 2, Key("alt+" + lead, None)

    @staticmethod
    def _csi(params: str, final: str) -> Event | None:
        if params.startswith("<") and final in "Mm":
            try:
                button, x, y = (int(part) for part in params[1:].split(";"))
            except ValueError:
                return None
            if button & 32:
                return None  # motion
            if button & 64:
                return Mouse("wheel_down" if button & 1 else "wheel_up", x - 1, y - 1)
            return Mouse("press" if final == "M" else "release", x - 1, y - 1, button & 3)
        parts = params.split(";") if params else []
        try:
            numbers = [int(part) if part else 1 for part in parts]
        except ValueError:
            return None
        modifier = numbers[1] if len(numbers) > 1 else 1
        if final == "~":
            name = _TILDE_KEYS.get(numbers[0] if numbers else 0)
            return Key(_modified(name, modifier)) if name else None
        if final == "Z":
            return Key("shift+tab")
        name = _LETTER_KEYS.get(final)
        if name is not None:
            return Key(_modified(name, modifier))
        return None
