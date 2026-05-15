#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# ── sklearn version compatibility fix (MUST run before any model loading) ──
try:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    DecisionTreeClassifier.monotonic_cst = None
    DecisionTreeRegressor.monotonic_cst = None
    import warnings
    warnings.filterwarnings('ignore', message='.*Trying to unpickle estimator.*')
except Exception:
    pass
# ── end fix ──


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agrizone.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()