from __future__ import annotations


class TransportError(RuntimeError):
    """Base class for transport-level failures."""


class ProducerError(TransportError):
    """A producer failed to deliver an event after exhausting retries."""


class ConsumerError(TransportError):
    """A consumer failed to receive or decode an event."""
