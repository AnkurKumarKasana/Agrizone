"""
Scikit-learn Cross-Version Compatibility Patch
================================================
Fixes 'monotonic_cst' AttributeError when loading models trained with
sklearn 1.3.x in an environment running sklearn 1.5+.

This module monkey-patches DecisionTreeClassifier and DecisionTreeRegressor
at the CLASS level so that ALL instances (including those deserialized from
pickle) automatically get the missing attribute. This runs before any model
is loaded.

Import this module early in Django startup (agrizone/__init__.py).
"""

import logging

logger = logging.getLogger(__name__)


def _apply_sklearn_compat_patch():
    """
    Patch sklearn tree estimators at the class level for cross-version
    compatibility. Safe to call multiple times (idempotent).
    """
    try:
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    except ImportError:
        return  # sklearn not installed, nothing to patch

    for cls in (DecisionTreeClassifier, DecisionTreeRegressor):
        if getattr(cls, '_monotonic_cst_patched', False):
            continue  # already patched

        original_getattr = getattr(cls, '__getattr__', None)

        def _make_patched_getattr(orig):
            def __getattr__(self, name):
                if name == 'monotonic_cst':
                    return None
                if orig is not None:
                    return orig(self, name)
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )
            return __getattr__

        cls.__getattr__ = _make_patched_getattr(original_getattr)
        cls._monotonic_cst_patched = True

    # Also suppress the InconsistentVersionWarning spam
    import warnings
    try:
        from sklearn.exceptions import InconsistentVersionWarning
        warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
    except ImportError:
        # Older sklearn versions don't have this warning class
        warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
        warnings.filterwarnings('ignore', message='.*unpickle estimator.*')

    logger.info("sklearn cross-version compatibility patch applied successfully.")


# Apply the patch when this module is imported
_apply_sklearn_compat_patch()
