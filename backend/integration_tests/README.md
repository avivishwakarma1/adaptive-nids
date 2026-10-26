# Opt-in MongoDB integration tests

These tests are separate from the default backend unit-test suite. They use
real PyMongo operations and the backend's actual persistence and dashboard
routes, while replacing model loading and inference with a deterministic stub.
They do not use application `.env` settings.

The harness refuses to run unless all of the following are true:

- `NIDS_MONGO_IT=1` explicitly opts in.
- `NIDS_MONGO_IT_MODE` is `replica_set` or `standalone`.
- `NIDS_MONGO_IT_URI` exactly matches the selected loopback endpoint below.
- The harness generates a fresh UUID run identifier and database name of the
  form `adaptive_nids_it_<32 lowercase hexadecimal characters>`.
- The selected server reports the expected topology, is writable, and the
  generated database does not already exist.

The supported endpoints are fixed:

- Replica set: `mongodb://127.0.0.1:27018/?directConnection=true`
- Standalone: `mongodb://127.0.0.1:27019/?directConnection=true`

The tests never use port 27017 or the `adaptive_nids` database. They do not
start or stop MongoDB processes. Run them only when these two endpoints are
dedicated test instances with no unrelated test data. The session fixture drops
only its exact generated database after rechecking the server topology and
fixture-specific ownership marker. The marker is an accidental-deletion guard,
not a security boundary against a process with unrestricted database access.

PyMongo is created with `directConnection=True`, so it remains pinned to the
approved loopback seed rather than following replica-set discovery. The
replica-set server must additionally advertise and configure exactly one
member, `127.0.0.1:27018`; any additional, hidden, passive, or arbiter member
causes the harness to fail closed. Direct connection still sends transactions
to the selected server, which must be a writable primary in replica set `rs0`.
Topology is checked before fixture writes, before test setup writes, and before
teardown.

The database name is generated inside the harness; it is not accepted from an
environment variable. Existing database names are rejected. The ownership
marker is inserted before application test data. If ownership or topology
cannot be revalidated, teardown refuses to drop the database and emits a
diagnostic without connection details.

## Prepare isolated MongoDB instances

Install MongoDB Community Server if `mongod.exe` is not available. In two
separate PowerShell windows, create fresh data directories and start dedicated
instances. Replace the example temporary root with a new, empty directory for
each test run. Do not reuse the running application MongoDB service or its data
directory.

Replica-set process:

```powershell
$testRoot = Join-Path $env:TEMP 'adaptive-nids-mongo-it-<unique>'
$replicaData = Join-Path $testRoot 'replica'
New-Item -ItemType Directory -Path $replicaData -Force | Out-Null
mongod.exe --dbpath $replicaData --port 27018 --bind_ip 127.0.0.1 --replSet rs0
```

In another PowerShell window, initialize `rs0` once using the installed PyMongo
from the backend environment:

```powershell
@'
from pymongo import MongoClient
client = MongoClient(
    "mongodb://127.0.0.1:27018/?directConnection=true",
    serverSelectionTimeoutMS=3000,
)
client.admin.command("replSetInitiate", {
    "_id": "rs0",
    "members": [{"_id": 0, "host": "127.0.0.1:27018"}],
})
client.close()
'@ | py -3.11 -
```

Wait until the replica set elects its primary before running tests. In a
separate PowerShell window, start the standalone server using a different new
data directory:

```powershell
$standaloneData = Join-Path $testRoot 'standalone'
New-Item -ItemType Directory -Path $standaloneData -Force | Out-Null
mongod.exe --dbpath $standaloneData --port 27019 --bind_ip 127.0.0.1
```

Because each `mongod.exe` command occupies its PowerShell window, repeat the
same `$testRoot` assignment in the window used to start the standalone
instance. Replace `<unique>` with the same new, unused directory name in both
windows.

These commands are setup instructions only; the integration harness does not
manage MongoDB processes. Stop only the two test processes you started, using
their individual process IDs, after testing.

## Run

Run commands from the repository root in PowerShell. The backend environment
must have the dependencies from `backend\requirements.txt` installed.

Replica-set mode (tests successful persistence, alert rule, dashboard
aggregation, and transaction rollback):

```powershell
$env:NIDS_MONGO_IT = '1'
$env:NIDS_MONGO_IT_MODE = 'replica_set'
$env:NIDS_MONGO_IT_URI = 'mongodb://127.0.0.1:27018/?directConnection=true'
py -3.11 -m pytest -q backend\integration_tests
Remove-Item Env:NIDS_MONGO_IT, Env:NIDS_MONGO_IT_MODE, Env:NIDS_MONGO_IT_URI
```

Standalone mode (tests successful persistence, alert rule, dashboard
aggregation, and compensating cleanup after an injected partial-write failure):

```powershell
$env:NIDS_MONGO_IT = '1'
$env:NIDS_MONGO_IT_MODE = 'standalone'
$env:NIDS_MONGO_IT_URI = 'mongodb://127.0.0.1:27019/?directConnection=true'
py -3.11 -m pytest -q backend\integration_tests
Remove-Item Env:NIDS_MONGO_IT, Env:NIDS_MONGO_IT_MODE, Env:NIDS_MONGO_IT_URI
```

The generated database is checked for nonexistence before app startup, then
dropped by the session fixture after tests. A function-scoped fixture clears
only known application test collections inside that owned database before each
test; the ownership collection is retained. This makes dashboard totals and
risk aggregation independent of test order. If the server identity or owner
marker cannot be verified, the harness refuses to proceed or drop the
database. Do not manually point the URI variable at another endpoint.

## Coverage and limits

- Real writes through `POST /api/v1/predictions`, preserving its response
  contract and checking stored observation timestamps and row mappings.
- Alert eligibility: `known_attack`, risk score at least 60, and severity
  `high` or `critical`; benign predictions do not alert.
- Real MongoDB dashboard aggregation across legacy dataset predictions and
  standalone predictions, including bucket count and weighted mean risk.
- Replica-set rollback after a successful transactional prediction insert
  followed by an injected alert write failure.
- Standalone transaction rejection, successful fallback prediction insertion,
  compensating deletion, and controlled API error after an injected alert
  write failure.

Authentication and API behavior use the real backend routes with a temporary
test user in the generated database. Model inference is stubbed, so these are
not model-quality tests. The failure injection is at the PyMongo collection
method boundary; the documents written before each injected failure are real
MongoDB writes.
