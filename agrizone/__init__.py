# Apply sklearn compatibility patch at Django startup, before any models load.
# This fixes 'monotonic_cst' errors when models trained with sklearn 1.3.x
# are loaded in an environment running sklearn 1.5+.
try:
    from agrizone.sklearn_compat import _apply_sklearn_compat_patch  # noqa: F401
except Exception:
    pass
