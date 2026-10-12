"""Resource failures that must stop search instead of rejecting a placement."""


class BackendCapacityError(RuntimeError):
    """A native workspace cannot represent a requested job."""

    def __init__(self, backend, resource, required, capacity, *, request_id=None, stage=None):
        self.diagnostics = {
            "backend": backend, "resource": resource,
            "required": int(required), "capacity": int(capacity),
            "request_id": request_id, "stage": stage,
            "reason": f"{backend}_capacity_overflow",
        }
        super().__init__(f"{backend} capacity exceeded for {resource}: "
                         f"required {required}, configured capacity {capacity}")
        self.diagnostics["message"] = str(self)
