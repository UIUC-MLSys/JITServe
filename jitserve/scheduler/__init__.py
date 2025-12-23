__all__ = ["SLOScheduler"]


def __getattr__(name: str):
    if name == "SLOScheduler":
        from .slo_scheduler import SLOScheduler
        return SLOScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
