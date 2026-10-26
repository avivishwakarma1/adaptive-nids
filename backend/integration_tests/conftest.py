import os
import re

import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pymongo import MongoClient


REPLICA_SET_URI = "mongodb://127.0.0.1:27018/?directConnection=true"
STANDALONE_URI = "mongodb://127.0.0.1:27019/?directConnection=true"
DATABASE_PREFIX = "adaptive_nids_it_"
DATABASE_PATTERN = re.compile(r"adaptive_nids_it_[0-9a-f]{32}\Z")
OWNER_COLLECTION = "_nids_integration_test_owner"
TEST_DATA_COLLECTIONS = (
    "users",
    "revoked_tokens",
    "datasets",
    "predictions",
    "standalone_predictions",
    "alerts",
)
APPLICATION_COLLECTIONS = {
    "users",
    "datasets",
    "jobs",
    "predictions",
    "standalone_predictions",
    "alerts",
    "investigations",
    "feedback",
    "model_registry",
    "model_evaluations",
    "reports",
    "audit_events",
    "simulated_actions",
    "revoked_tokens",
}


@dataclass
class MongoIntegrationEnvironment:
    client: MongoClient
    database: object
    mode: str
    uri: str
    database_name: str
    run_id: str


def _require_environment():
    if os.environ.get("NIDS_MONGO_IT") != "1":
        pytest.fail(
            "MongoDB integration tests require NIDS_MONGO_IT=1.",
            pytrace=False,
        )

    mode = os.environ.get("NIDS_MONGO_IT_MODE")
    uri = os.environ.get("NIDS_MONGO_IT_URI")
    expected_uri = {
        "replica_set": REPLICA_SET_URI,
        "standalone": STANDALONE_URI,
    }.get(mode)
    if expected_uri is None or uri != expected_uri:
        pytest.fail(
            "Select replica_set or standalone and provide its exact "
            "documented loopback URI.",
            pytrace=False,
        )

    return mode, uri


def _verify_server(client, mode):
    if client.options.direct_connection is not True:
        raise RuntimeError("MongoDB client is not pinned to the test endpoint.")

    hello = client.admin.command("hello")
    expected_host = (
        ("127.0.0.1", 27018)
        if mode == "replica_set"
        else ("127.0.0.1", 27019)
    )
    if client.nodes != {expected_host}:
        raise RuntimeError(
            "MongoDB driver topology includes an unapproved endpoint."
        )

    expected_address = f"{expected_host[0]}:{expected_host[1]}"

    if not hello.get("isWritablePrimary"):
        raise RuntimeError("The selected MongoDB endpoint is not writable.")

    if mode == "replica_set" and hello.get("me") != expected_address:
        raise RuntimeError("The selected MongoDB endpoint is not the expected replica-set member.")

    if mode == "standalone" and hello.get("setName") is not None:
        raise RuntimeError(
            "The selected MongoDB endpoint is configured as a replica-set member."
        )
    if mode == "replica_set":
        if hello.get("setName") != "rs0":
            raise RuntimeError("The selected endpoint is not in replica set rs0.")

        advertised_members = (
            list(hello.get("hosts") or [])
            + list(hello.get("passives") or [])
            + list(hello.get("arbiters") or [])
        )
        if advertised_members != [expected_address]:
            raise RuntimeError(
                "Replica set advertises members outside the single test endpoint."
            )

        config = client.admin.command({"replSetGetConfig": 1})["config"]
        members = config.get("members", [])
        if (
            len(members) != 1
            or members[0].get("host") != expected_address
            or members[0].get("arbiterOnly", False)
            or members[0].get("hidden", False)
        ):
            raise RuntimeError(
                "Replica set configuration is not the approved single-member test set."
            )
    elif hello.get("setName") is not None:
        raise RuntimeError("The selected endpoint is not a standalone server.")


def _expected_owner(environment):
    return {
        "_id": environment.run_id,
        "database_name": environment.database_name,
        "mode": environment.mode,
        "uri": environment.uri,
    }


def _verify_owned_database(
    environment,
    allowed_collections=APPLICATION_COLLECTIONS | {OWNER_COLLECTION},
):
    if not DATABASE_PATTERN.fullmatch(environment.database_name):
        raise RuntimeError("Integration database name failed validation.")

    _verify_server(environment.client, environment.mode)
    marker = environment.database[OWNER_COLLECTION].find_one({
        "_id": environment.run_id,
    })
    if marker != _expected_owner(environment):
        raise RuntimeError("Integration database ownership marker is missing or invalid.")
    unexpected = (
        set(environment.database.list_collection_names())
        - allowed_collections
    )
    if unexpected:
        raise RuntimeError("Integration database contains unexpected collections.")


@pytest.fixture(scope="session")
def mongo_integration_environment():
    mode, uri = _require_environment()
    run_id = uuid4().hex
    database_name = DATABASE_PREFIX + run_id
    if not DATABASE_PATTERN.fullmatch(database_name):
        pytest.fail("Generated integration database name is invalid.", pytrace=False)

    client = MongoClient(
        uri,
        directConnection=True,
        serverSelectionTimeoutMS=3000,
        connectTimeoutMS=3000,
        appname="adaptive-nids-integration-tests",
        tz_aware=True,
    )
    database = client[database_name]
    environment = MongoIntegrationEnvironment(
        client=client,
        database=database,
        mode=mode,
        uri=uri,
        database_name=database_name,
        run_id=run_id,
    )
    ownership_established = False

    try:
        _verify_server(client, mode)
        if database_name in client.list_database_names():
            pytest.fail(
                "Generated integration database already exists; refusing to use it.",
                pytrace=False,
            )
        if database.list_collection_names():
            pytest.fail(
                "Generated integration database contains unexpected collections.",
                pytrace=False,
            )

        _verify_server(client, mode)
        database[OWNER_COLLECTION].insert_one(_expected_owner(environment))
        _verify_owned_database(
            environment,
            allowed_collections={OWNER_COLLECTION},
        )
        ownership_established = True

        yield environment
    except Exception:
        if not ownership_established:
            warnings.warn(
                "Integration database ownership was not confirmed; "
                "the harness will not remove any database.",
                RuntimeWarning,
                stacklevel=2,
            )
        raise
    finally:
        try:
            if ownership_established:
                try:
                    _verify_owned_database(environment)
                    client.drop_database(database_name)
                except Exception as exc:
                    warnings.warn(
                        "Refusing to drop the MongoDB integration database "
                        "because endpoint, topology, or ownership validation "
                        f"failed ({type(exc).__name__}).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        finally:
            client.close()


@pytest.fixture
def integration_client(
    monkeypatch,
    mongo_integration_environment,
):
    environment = mongo_integration_environment
    _verify_owned_database(environment)

    collection_names = set(environment.database.list_collection_names())
    allowed_collections = APPLICATION_COLLECTIONS | {OWNER_COLLECTION}
    unexpected_collections = collection_names - allowed_collections
    if unexpected_collections:
        raise RuntimeError(
            "Owned integration database contains unexpected collections."
        )

    _verify_owned_database(environment)
    for collection_name in TEST_DATA_COLLECTIONS:
        environment.database[collection_name].delete_many({})

    # The app module creates a default app when imported. Import from an empty
    # working directory so a project .env file cannot supply those defaults.
    with tempfile.TemporaryDirectory(prefix="nids-it-import-") as empty_cwd:
        original_cwd = Path.cwd()
        os.chdir(empty_cwd)
        try:
            from app.config import Settings

            settings_env_names = {
                name.upper()
                for name in Settings.model_fields
            }
            saved_settings_env = {
                name: os.environ.pop(name)
                for name in settings_env_names
                if name in os.environ
            }
            try:
                from app import main
            finally:
                os.environ.update(saved_settings_env)
        finally:
            os.chdir(original_cwd)

    # class StubPredictor:
    #     results = []

    #     def predict(self, observations):
    #         return self.results
    class StubPredictor:
        def __init__(self):
            self.results = []

        def predict(self, observations):
            return self.results

    predictor = StubPredictor()
    monkeypatch.setattr(
        main,
        "load_active_predictor",
        lambda *args, **kwargs: predictor,
    )
    settings = Settings(
        _env_file=None,
        jwt_secret_key="integration-test-secret-not-for-production",
        mongodb_uri=environment.uri,
        mongodb_database=environment.database_name,
    )
    app = main.create_app(
        settings,
        mongo_client=environment.client,
    )
    from app.config import get_settings

    app.dependency_overrides[get_settings] = lambda: settings

    _verify_owned_database(environment)
    with TestClient(app) as test_client:
        from bson import ObjectId
        from app.security import create_access_token

        _verify_owned_database(environment)
        user_id = ObjectId()
        environment.database.users.insert_one({
            "_id": user_id,
            "is_active": True,
            "role": "analyst",
        })
        token, _ = create_access_token(
            str(user_id),
            "analyst",
            settings,
        )
        yield {
            "client": test_client,
            "database": environment.database,
            "main": main,
            "predictor": predictor,
            "token": token,
            "mode": environment.mode,
            "environment": environment,
            "verify": lambda: _verify_owned_database(environment),
        }
