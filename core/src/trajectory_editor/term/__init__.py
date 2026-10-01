"""Immediate-mode terminal interface.

Every frame is a pure function of one UI state snapshot. One UI thread owns
that state, drains all pending input and engine events, then renders the whole
screen into a cell canvas and writes only the changed lines to the terminal in
a single synchronized-output transaction. Nothing outside the UI thread draws,
so a frame can never mix two states.
"""
