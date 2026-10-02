class SwarmGridError(Exception):
    code = "internal_error"
    status = 500


class ValidationError(SwarmGridError):
    code = "validation_error"
    status = 400


class NotFoundError(SwarmGridError):
    code = "not_found"
    status = 404


class ConflictError(SwarmGridError):
    code = "conflict"
    status = 409


class UnsolvableError(SwarmGridError):
    """A mission has no collision-free plan for any priority order that was tried."""

    code = "unsolvable"
    status = 409
