from edmo.infrastructure.event_bus import topics


def test_topic_names_are_stable() -> None:
    assert topics.OPTIMIZATION_EVENTS == "optimization-events"
    assert topics.OPTIMIZATION_RESULTS == "optimization-results"
