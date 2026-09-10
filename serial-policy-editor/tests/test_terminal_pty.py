"""Check terminal ownership and cleanup using a real Unix pseudoterminal."""
import errno
import fcntl
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import sys
import termios
import time


CHILD = r'''
import os, sys
from dataclasses import replace
from tests.test_persistent_tui import choice_state
from trajectory_editor.edge_tui import EdgeViewState
from trajectory_editor.persistent_tui import PersistentTerminalSession
notify, release = map(int, sys.argv[1:])
class Session(PersistentTerminalSession):
    seen = None
    resized = False
    def _rendered(self, app):
        super()._rendered(app)
        if app.is_done:
            return
        if self.accepting_input and self._current is not self.seen:
            self.seen = self._current
            os.write(notify, b'R')
        elif not self.accepting_input and self.output_device.get_size().columns == 100 and not self.resized:
            self.resized = True
            os.write(notify, b'Z')
with Session() as session:
    assert session.read_choice(choice_state()) == '1'
    os.write(notify, b'B')
    assert os.read(release, 1) == b'!'
    assert session.read_edge(EdgeViewState('pty', 1, 10, 9, 'seed=1')) == 'e'
print('RESTORED')
'''


def test_raw_mode_survives_actions_and_resize_then_restores():
    master, slave = pty.openpty()
    notice_read, notice_write = os.pipe()
    release_read, release_write = os.pipe()
    original = termios.tcgetattr(slave)
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, 80, 0, 0))
    os.set_blocking(master, False)
    transcript = bytearray()
    child = None

    def drain():
        while True:
            try:
                chunk = os.read(master, 65536)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno == errno.EIO:
                    return
                raise
            if not chunk:
                return
            transcript.extend(chunk)

    def notice(expected):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            ready, _, _ = select.select([notice_read, master], [], [], .1)
            if master in ready:
                drain()
            if notice_read in ready:
                value = os.read(notice_read, 1)
                assert value == expected, (value, bytes(transcript)[-2000:])
                return
            assert child.poll() is None, bytes(transcript)[-2000:]
        raise AssertionError(f'Terminal did not send {expected!r}: {bytes(transcript)[-2000:]!r}')

    try:
        env = dict(os.environ, TERM='xterm-256color')
        child = subprocess.Popen(
            [sys.executable, '-c', CHILD, str(notice_write), str(release_read)],
            cwd=Path(__file__).resolve().parents[1], env=env,
            stdin=slave, stdout=slave, stderr=slave,
            pass_fds=(notice_write, release_read), start_new_session=True,
        )
        notice(b'R')
        assert not termios.tcgetattr(slave)[3] & (termios.ECHO | termios.ICANON)
        os.write(master, b'1\r')
        notice(b'B')
        assert not termios.tcgetattr(slave)[3] & (termios.ECHO | termios.ICANON)
        # The owner is blocked between requests; the UI must still handle
        # resize and consume stale commands without releasing terminal modes.
        os.write(master, b'\r2\r')
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 40, 100, 0, 0))
        notice(b'Z')
        os.write(release_write, b'!')
        notice(b'R')
        os.write(master, b'e\r')
        child.wait(timeout=5)
        drain()
        assert child.returncode == 0, bytes(transcript)[-3000:]
        assert termios.tcgetattr(slave) == original
        raw = bytes(transcript)
        assert raw.count(b'\x1b[?1049h') == raw.count(b'\x1b[?1049l') == 1
        assert raw.count(b'\x1b[J') == 3  # Initial paint, actual resize, final exit.
        assert raw.index(b'\x1b[?1049l') < raw.index(b'RESTORED')
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        for descriptor in (master, slave, notice_read, notice_write, release_read, release_write):
            os.close(descriptor)
