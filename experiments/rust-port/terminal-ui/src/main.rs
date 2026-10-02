//! A deliberately small terminal process for proving the Python PTY/pyte seam.
//!
//! This is not a port of the editor. It owns just enough input, resize, and
//! rendering behavior to exercise the cross-language screen oracle.

use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;

const STARTUP: &[u8] = b"\x1b[?1049h\x1b[2J\x1b[?25h";
const FRAME_START: &[u8] = b"\x1b[?2026h";
const FRAME_END: &[u8] = b"\x1b[?2026l";
const INPUT_PREFIX: &str = "Input > ";

static RESIZE_PENDING: AtomicBool = AtomicBool::new(false);

extern "C" fn on_resize(_signal: libc::c_int) {
    RESIZE_PENDING.store(true, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Size {
    columns: usize,
    rows: usize,
}

#[derive(Serialize)]
struct FrameRecord {
    sequence: u64,
    size: [usize; 2],
    lines: Vec<String>,
    cursor: Option<[usize; 2]>,
    end_offset: usize,
}

struct TerminalGuard {
    original: libc::termios,
    mode_changed: bool,
    old_winch_handler: Option<libc::sighandler_t>,
    screen_entered: bool,
}

impl TerminalGuard {
    fn enter() -> io::Result<(Self, usize)> {
        // SAFETY: `original` is written by tcgetattr before it is read below.
        let mut original = unsafe { std::mem::zeroed::<libc::termios>() };
        // SAFETY: the pointer is valid and points to writable termios storage.
        if unsafe { libc::tcgetattr(libc::STDIN_FILENO, &mut original) } != 0 {
            return Err(io::Error::last_os_error());
        }

        let mut guard = Self {
            original,
            mode_changed: false,
            old_winch_handler: None,
            screen_entered: false,
        };

        let mut raw = original;
        raw.c_iflag &= !(libc::BRKINT | libc::ICRNL | libc::INPCK | libc::ISTRIP | libc::IXON);
        raw.c_oflag &= !libc::OPOST;
        raw.c_cflag |= libc::CS8;
        raw.c_lflag &= !(libc::ECHO | libc::ICANON | libc::IEXTEN | libc::ISIG);
        raw.c_cc[libc::VMIN] = 1;
        raw.c_cc[libc::VTIME] = 0;

        // SAFETY: `raw` is a valid termios value for this controlling PTY.
        if unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &raw) } != 0 {
            return Err(io::Error::last_os_error());
        }
        guard.mode_changed = true;

        RESIZE_PENDING.store(false, Ordering::Relaxed);
        // SAFETY: the handler only performs a lock-free atomic store.
        let old_handler =
            unsafe { libc::signal(libc::SIGWINCH, on_resize as *const () as libc::sighandler_t) };
        if old_handler == libc::SIG_ERR {
            return Err(io::Error::last_os_error());
        }
        guard.old_winch_handler = Some(old_handler);

        guard.screen_entered = true;
        let mut stdout = io::stdout().lock();
        stdout.write_all(STARTUP)?;
        stdout.flush()?;
        Ok((guard, STARTUP.len()))
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if let Some(handler) = self.old_winch_handler {
            // SAFETY: restoring the process's previous SIGWINCH handler.
            unsafe {
                libc::signal(libc::SIGWINCH, handler);
            }
        }
        if self.mode_changed {
            // SAFETY: `original` was read from this same terminal with tcgetattr.
            unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.original);
            }
        }
        if self.screen_entered {
            let mut stdout = io::stdout().lock();
            let _ = stdout.write_all(b"\x1b[?25h\x1b[?1049l");
            let _ = stdout.flush();
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Rust terminal PTY smoke failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> io::Result<()> {
    let frame_log_path = env::var_os("SPE_TERMINAL_FRAME_LOG")
        .map(PathBuf::from)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "SPE_TERMINAL_FRAME_LOG is required",
            )
        })?;
    let mut frame_log = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(frame_log_path)?;

    let (_terminal, startup_bytes) = TerminalGuard::enter()?;
    let mut size = terminal_size()?;
    let mut output_offset = startup_bytes;
    let mut sequence = 0_u64;
    let mut input = String::new();
    let mut last_submitted = String::from("(none)");
    present(
        &mut frame_log,
        &mut output_offset,
        &mut sequence,
        size,
        &input,
        &last_submitted,
    )?;

    loop {
        let mut descriptor = libc::pollfd {
            fd: libc::STDIN_FILENO,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: `descriptor` points to one initialized pollfd for the duration
        // of the call. A finite timeout lets the loop observe SIGWINCH promptly.
        let polled = unsafe { libc::poll(&mut descriptor, 1, 100) };
        if polled < 0 && io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
            return Err(io::Error::last_os_error());
        }

        if RESIZE_PENDING.swap(false, Ordering::Relaxed) {
            let current = terminal_size()?;
            if current != size {
                size = current;
                present(
                    &mut frame_log,
                    &mut output_offset,
                    &mut sequence,
                    size,
                    &input,
                    &last_submitted,
                )?;
            }
        }

        if polled <= 0 || descriptor.revents & (libc::POLLIN | libc::POLLHUP) == 0 {
            continue;
        }

        let mut byte = [0_u8; 1];
        // SAFETY: `byte` is writable storage for one byte, and stdin is the
        // controlling PTY descriptor polled above.
        let read = unsafe { libc::read(libc::STDIN_FILENO, byte.as_mut_ptr().cast(), byte.len()) };
        if read < 0 {
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(io::Error::last_os_error());
        }
        if read == 0 {
            break;
        }
        match byte[0] {
            b'\r' | b'\n' => {
                if matches!(input.as_str(), "quit" | "exit") {
                    break;
                }
                last_submitted.clone_from(&input);
                if last_submitted.is_empty() {
                    last_submitted.push_str("(empty)");
                }
                input.clear();
            }
            0x03 | 0x04 => break,
            0x08 | 0x7f => {
                input.pop();
            }
            byte if (0x20..=0x7e).contains(&byte) => input.push(char::from(byte)),
            _ => continue,
        }
        present(
            &mut frame_log,
            &mut output_offset,
            &mut sequence,
            size,
            &input,
            &last_submitted,
        )?;
    }

    Ok(())
}

fn terminal_size() -> io::Result<Size> {
    // SAFETY: zero is a valid initial value for winsize; ioctl fills the struct.
    let mut actual = unsafe { std::mem::zeroed::<libc::winsize>() };
    // SAFETY: `actual` points to writable winsize storage for this ioctl.
    if unsafe { libc::ioctl(libc::STDIN_FILENO, libc::TIOCGWINSZ, &mut actual) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(Size {
        columns: match usize::from(actual.ws_col) {
            0 => 80,
            columns => columns,
        },
        rows: match usize::from(actual.ws_row) {
            0 => 24,
            rows => rows,
        },
    })
}

fn present(
    frame_log: &mut File,
    output_offset: &mut usize,
    sequence: &mut u64,
    size: Size,
    input: &str,
    last_submitted: &str,
) -> io::Result<()> {
    *sequence += 1;
    let mut lines = vec![String::new(); size.rows];
    if size.rows > 0 {
        lines[0] = "Rust terminal PTY smoke".to_owned();
    }
    if size.rows > 1 {
        lines[1] = format!("Geometry: {}x{}", size.columns, size.rows);
    }
    if size.rows > 2 {
        lines[2] = "Type text and press Enter; type quit to exit.".to_owned();
    }
    if size.rows > 3 {
        lines[3] = format!("Submitted: {last_submitted}");
    }

    let input_row = 4.min(size.rows - 1);
    let input_line = format!("{INPUT_PREFIX}{input}");
    lines[input_row] = input_line.clone();
    for line in &mut lines {
        fit_ascii_line(line, size.columns);
    }

    let cursor = [
        (INPUT_PREFIX.len() + input.len()).min(size.columns - 1),
        input_row,
    ];
    let mut transaction = Vec::new();
    transaction.extend_from_slice(FRAME_START);
    for (row, line) in lines.iter().enumerate() {
        transaction.extend_from_slice(format!("\x1b[{};1H", row + 1).as_bytes());
        transaction.extend_from_slice(line.as_bytes());
    }
    transaction.extend_from_slice(
        format!("\x1b[{};{}H\x1b[?25h", cursor[1] + 1, cursor[0] + 1).as_bytes(),
    );
    transaction.extend_from_slice(FRAME_END);

    {
        let mut stdout = io::stdout().lock();
        stdout.write_all(&transaction)?;
        stdout.flush()?;
    }
    *output_offset += transaction.len();

    let record = FrameRecord {
        sequence: *sequence,
        size: [size.columns, size.rows],
        lines,
        cursor: Some(cursor),
        end_offset: *output_offset,
    };
    serde_json::to_writer(&mut *frame_log, &record).map_err(io::Error::other)?;
    frame_log.write_all(b"\n")?;
    frame_log.flush()
}

fn fit_ascii_line(line: &mut String, width: usize) {
    line.truncate(line.len().min(width));
    line.push_str(&" ".repeat(width - line.len()));
}
