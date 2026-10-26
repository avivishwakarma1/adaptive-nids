from types import SimpleNamespace

from app.db import ensure_indexes


class IndexCollection:
    def __init__(self):
        self.indexes = []

    def create_index(self, keys, **options):
        self.indexes.append((keys, options))


def test_standalone_indexes_are_additive_and_keep_dataset_index_unchanged():
    names = (
        "users",
        "datasets",
        "jobs",
        "predictions",
        "standalone_predictions",
        "alerts",
        "audit_events",
    )
    collections = {
        name: IndexCollection()
        for name in names
    }

    ensure_indexes(SimpleNamespace(**collections))

    assert collections["predictions"].indexes == [
        ([("dataset_id", 1), ("row_index", 1)], {"unique": True}),
    ]
    assert collections["standalone_predictions"].indexes == [
        (
            [("prediction_batch_id", 1), ("row_index", 1)],
            {"unique": True},
        ),
        ("observed_at", {}),
    ]
