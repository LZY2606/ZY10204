"""
Root-level pytest configuration shared by test runs.

The ``suite`` directory is a separately checked-out external conformance
suite (a git submodule in full checkouts) which ships its own unrelated
test harness with its own dependencies. Exclude it from root collection so
that a plain ``pytest`` invocation tests this package rather than the
external harness; pointing pytest at a path inside ``suite`` explicitly
still works.
"""

collect_ignore_glob = ["suite"]
