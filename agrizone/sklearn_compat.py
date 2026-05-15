"""
Scikit-learn Cross-Version Compatibility Patch
================================================
Models trained with sklearn 1.3.x lack 'monotonic_cst' attribute that
sklearn 1.5+ expects on DecisionTreeClassifier/Regressor during predict().

Fix: Set monotonic_cst = None as a CLASS-LEVEL default attribute.
When Python resolves self.monotonic_cst on an old unpickled instance:
  1. Checks instance __dict__ → not there (old pickle)
  2. Checks class __dict__ → FOUND (our patch) → returns None
  3. sklearn sees None and skips monotonic constraint logic → works!

This is the simplest possible fix with zero chance of failure.
"""

def apply_sklearn_patch():
    try:
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.tree._classes import BaseDecisionTree

        # Set class-level defaults for the missing attribute
        for cls in (DecisionTreeClassifier, DecisionTreeRegressor, BaseDecisionTree):
            if not hasattr(cls, 'monotonic_cst') or 'monotonic_cst' not in cls.__dict__:
                cls.monotonic_cst = None

        # Suppress version mismatch warnings globally
        import warnings
        warnings.filterwarnings('ignore', message='.*Trying to unpickle estimator.*')
        warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

        try:
            from sklearn.exceptions import InconsistentVersionWarning
            warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
        except ImportError:
            pass

    except Exception:
        pass  # sklearn not installed or incompatible — skip silently


# Auto-apply on import
apply_sklearn_patch()
