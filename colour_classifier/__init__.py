"""Sock colour classifier: `from colour_classifier import identify_sock`. See README.md."""

from .sock_classifier import (
    EMPTY,
    ArmCamera,
    UNKNOWN,
    CloseUpClassifier,
    SockClassifier,
    identify_sock,
    identify_socks,
    read_sock_colour,
)

__all__ = ["ArmCamera", "read_sock_colour", "EMPTY", "UNKNOWN", "CloseUpClassifier", "SockClassifier", "identify_sock", "identify_socks"]
