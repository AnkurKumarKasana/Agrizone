import os

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

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agrizone.settings')
application = get_wsgi_application()