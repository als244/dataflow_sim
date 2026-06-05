"""Bench for the static-prepass validator (`validate_chain`).

Parametrized over every `make_invalid_*` fixture in
`invalid_chain_fixtures.py`. Each fixture is run through `run(chain)` and is
expected to raise `ValidationError` (a `ValueError` subclass) BEFORE the
simulator starts stepping. The error message is matched against the
expected keyword(s) declared alongside the fixture.

Some fixtures pin "current behavior" (e.g. empty chain is a no-op today)
with `EXPECTED_ERROR_KEYWORDS[name] is None`. Those are run without
expecting an error.
"""
import pytest

from dataflow_sim import ValidationError
from dataflow_sim.engine.simulator import run
from invalid_chain_fixtures import EXPECTED_ERROR_KEYWORDS
import invalid_chain_fixtures as fixtures_module


FIXTURES = [
    (name, getattr(fixtures_module, name))
    for name in dir(fixtures_module)
    if name.startswith("make_invalid_")
]


def _matches(message: str, expected) -> bool:
    """Return True if the validator message matches the expected keyword(s).

    `expected` may be a single string (must appear as a case-insensitive
    substring), a list of strings (ANY one must match), or None (no
    expectation — see fixture docstring for context).
    """
    if expected is None:
        return True
    msg_lower = message.lower()
    if isinstance(expected, str):
        return expected.lower() in msg_lower
    if isinstance(expected, list):
        return any(kw.lower() in msg_lower for kw in expected)
    raise TypeError(f"unsupported expected-keyword type: {type(expected)!r}")


@pytest.mark.parametrize("name,fixture_fn", FIXTURES)
def test_invalid_chain_rejected(name, fixture_fn):
    chain = fixture_fn()
    expected = EXPECTED_ERROR_KEYWORDS[name]

    if expected is None:
        # Current behavior pin: no error expected (e.g. empty chain → no-op).
        # If the validator starts raising here, flip the fixture's expected
        # keyword instead of editing this branch.
        run(chain)
        return

    with pytest.raises(ValidationError) as excinfo:
        run(chain)  # validate=True by default; raises BEFORE stepping.
    assert _matches(str(excinfo.value), expected), (
        f"validator should mention {expected!r}; got: {excinfo.value}"
    )


def test_validate_can_be_skipped():
    """`run(chain, validate=False)` bypasses the static prepass.

    We use an unknown-input fixture: with validate=True it raises
    ValidationError up front; with validate=False it bypasses the prepass
    and either runs (if the runtime tolerates it) or raises a different
    error class from inside the simulator — but NOT ValidationError.
    """
    chain = fixtures_module.make_invalid_id_resolution_unknown_input()
    # validate=True path: definitely ValidationError.
    with pytest.raises(ValidationError):
        run(chain, validate=True)
    # validate=False path: must NOT raise ValidationError. Any other
    # exception (or a successful run) is acceptable — the contract here is
    # that the prepass was skipped.
    try:
        run(chain, validate=False)
    except ValidationError:  # pragma: no cover - would be a regression
        pytest.fail("validate=False should bypass the static prepass")
    except Exception:
        # Runtime surfaced its own error — fine; prepass was skipped.
        pass
