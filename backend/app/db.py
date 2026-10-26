from pymongo import MongoClient
from pymongo.errors import PyMongoError

COLLECTIONS = ("users","datasets","jobs","predictions","standalone_predictions","alerts","investigations","feedback","model_registry","model_evaluations","reports","audit_events","simulated_actions")

def create_mongo_client(uri: str):
    if not uri:
        return None
    return MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500, appname="adaptive-ai-nids")

def ensure_indexes(db):
    db.users.create_index("email_normalized", unique=True)
    db.users.create_index([("role", 1), ("is_active", 1)])
    db.datasets.create_index([("created_at", -1), ("_id", -1)])
    db.datasets.create_index([("status", 1), ("created_at", -1)])
    db.datasets.create_index([("created_by", 1), ("created_at", -1)])
    db.jobs.create_index([("job_type", 1), ("idempotency_key", 1)], unique=True)
    db.jobs.create_index([("status", 1), ("created_at", 1)])
    db.jobs.create_index([("resource_id", 1), ("created_at", -1)])
    db.predictions.create_index([("dataset_id", 1), ("row_index", 1)], unique=True)
    db.standalone_predictions.create_index(
        [("prediction_batch_id", 1), ("row_index", 1)],
        unique=True,
    )
    db.standalone_predictions.create_index("observed_at")
    db.alerts.create_index([("created_at", -1), ("_id", -1)])
    db.audit_events.create_index([("created_at", -1), ("_id", -1)])

def ping_database(client) -> bool:
    if client is None:
        return False
    try:
        client.admin.command("ping")
        return True
    except PyMongoError:
        return False
