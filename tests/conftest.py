import os

# Force the no-cost stub path.
os.environ.setdefault("PARANOID_QA_PROVIDER", "stub")
os.environ.setdefault("PARANOID_QA_EMBED_PROVIDER", "stub")
