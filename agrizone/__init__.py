# ── sklearn version compatibility fix ──
# Models trained with sklearn 1.3.x lack 'monotonic_cst' attribute
# that sklearn 1.5+ expects. Setting it as a class-level default
# makes Python's attribute resolution return None for old pickled models.
try:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    DecisionTreeClassifier.monotonic_cst = None
    DecisionTreeRegressor.monotonic_cst = None
    import warnings
    warnings.filterwarnings('ignore', message='.*Trying to unpickle estimator.*')
except Exception:
    pass
