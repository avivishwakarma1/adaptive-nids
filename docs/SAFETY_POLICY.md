# Safety Policy
**Version:** 1.0.0  
**Applies to:** frontend, backend, ingestion tools, ML, scripts, tests, and future integrations

## 1. System classification

This project is a **passive detection, offline analysis, and analyst decision-support tool**. It does not enforce network policy and is not an inline intrusion prevention system (IPS). Predictions and alerts are advisory and may be wrong.

## 2. Prohibited operations

The application must not:
- Modify host network configuration.
- Change IP addresses, DNS settings, routes, proxy configuration, network interfaces, adapters, firewall rules, or packet-filter rules.
- Inject, drop, alter, replay, redirect, or transmit packets as a response action.
- Perform actual attack prevention, host isolation, account lockout, or device remediation.
- Execute shell commands derived from uploaded data or API requests.
- Invoke external enforcement APIs for firewall, router, switch, endpoint, or cloud security-group changes.
- Require administrator/root privileges for ordinary startup, CSV processing, offline PCAP analysis, or dashboard use.
- Automatically run arbitrary uploaded scripts, plugins, model files, macros, or binaries.

Any future feature that could violate these constraints requires a separate explicit architecture and safety review; it is not an acceptable implementation of v1.

## 3. Permitted operations

- Read user-selected offline CSV, PCAP/PCAPNG, and supported Zeek log files under configured size/resource limits.
- Parse and summarize traffic without transmitting or modifying traffic.
- Optionally perform authorized passive live capture in a future opt-in module, with no packet injection or response enforcement.
- Generate predictions, risk scores, alerts, reports, and audit events.
- Record simulated actions such as `mark_host_for_review`, `create_ticket_simulation`, or `notify_team_simulation`.
- Use synthetic/sample datasets and local test fixtures.

## 4. Offline analysis and parser isolation

Uploaded files are untrusted input. Parser implementation must:
- Validate extension, MIME/signature where practical, size, row/packet limits, and format before processing.
- Use maintained parsing libraries with pinned versions.
- Run with least privilege, resource limits, and no unnecessary network access.
- Avoid unsafe shell interpolation and avoid invoking external binaries unless explicitly approved and arguments are passed as a structured list.
- Enforce timeouts, memory limits where available, maximum decompressed size, and bounded error reporting.
- Reject malformed or unsupported inputs safely.
- Store files outside the web root using generated opaque names; prevent path traversal.
- Delete temporary files on success/failure according to retention policy.
- Never log full packet payloads, credentials, tokens, or unnecessarily sensitive identifiers.

For PCAP/PCAPNG, extract supported flow-level metadata. Payload inspection is not required for the initial release and should be disabled unless separately justified and reviewed. No packet replay or injection library should be included.

## 5. Optional live capture policy

Live capture is not required for basic operation and is deferred unless explicitly implemented later. If added:
- It must be opt-in, passive, and read-only.
- The operator must confirm authorization and select an approved interface.
- It must not change interface state, routing, DNS, firewall, or adapter settings.
- Capture must stop cleanly and expose visible status.
- It must not require elevated privileges by default. If a platform requires elevated permissions for a capture driver, document this as an optional platform prerequisite, not a general application requirement; do not attempt to elevate automatically.
- No live capture data may trigger an enforcement action.
- Tests must verify the capture path does not transmit packets or modify system configuration.

## 6. Simulated response actions

All response actions in v1 are records only:
- `mark_host_for_review`
- `create_ticket_simulation`
- `notify_team_simulation`

Each record must include alert ID, action, reason, acting user, timestamp, and status `simulated`. No real notification or ticket API is invoked in v1. The UI must label the action clearly as **Simulation only — no network or host action performed**. Do not label the feature “blocked”, “contained”, or “prevented”.

## 7. Authentication and authorization safety

- No public registration in v1.
- Use Argon2id or an approved secure password hashing scheme.
- Use short-lived signed access tokens, validate issuer/audience/expiry, and keep secrets outside source control.
- Enforce roles on every protected backend endpoint.
- Apply rate limits to login and upload operations.
- Do not reveal whether an email account exists through login errors.
- Never log passwords, tokens, session secrets, MongoDB URI, or secret environment values.
- Apply CORS allowlist from configuration; never use wildcard CORS with credentials.
- Use TLS in deployed environments.
- Use least-privilege database credentials.

## 8. Data privacy and retention

- Treat PCAPs, IP addresses, hostnames, usernames, and traffic metadata as potentially sensitive.
- Collect only features needed for analysis.
- Do not commit customer/user captures or secrets to Git.
- Use safe, licensed, non-sensitive fixtures.
- Limit report access to authorized users.
- Delete raw artifacts and temporary files according to configured retention.
- Redact sensitive fields from logs and API error responses.
- Do not send data to external AI services, telemetry, or hosted model endpoints unless an operator explicitly configures and approves that integration. Default is no outbound data transfer beyond configured MongoDB Atlas/API access.

## 9. ML safety and integrity

- Do not claim predictions are ground truth.
- Keep `risk_score` distinct from probability.
- Preserve uncertainty and model-unavailable states.
- Never train automatically from unreviewed feedback.
- Verify artifact checksum and schema before loading.
- Never load user-uploaded serialized model artifacts.
- Require evaluation and explicit admin approval before promotion.
- Keep a prior validated version for rollback.
- Audit all model lifecycle operations.
- Present evaluation scope, dataset provenance, class imbalance, and false-positive limitations.

## 10. Dependency and supply-chain controls

- Pin dependencies and commit lockfiles.
- Review optional dependencies and remove unused packet injection, firewall management, or privilege-escalation libraries.
- Run static checks, dependency vulnerability scans, and secret scanning in CI where available.
- Do not install packages at runtime.
- Treat uploaded data and external parser output as data, never instructions.
- Review license and provenance for datasets and models.

## 11. Required safety tests

1. Grep/static review confirms no implementation path calls firewall, route, DNS, proxy, adapter, or interface modification tools.
2. Tests confirm response simulation writes a record only and never invokes an enforcement adapter.
3. Upload path traversal and oversized files are rejected.
4. Malformed PCAP/CSV does not crash the process or leak stack traces.
5. Basic startup and CSV tests run as a normal non-admin user.
6. No packet-send/injection API is used by the project.
7. Unauthorized role actions return 403 and are audited.
8. Logs and errors do not expose credentials or raw packet payloads.
9. Model promotion is impossible without a passing validation report.
10. No automatic retraining occurs after feedback submission.

## 12. Incident handling

If a safety test reveals prohibited behavior:
- Disable the affected feature and block release.
- Preserve sanitized logs and test evidence.
- Remove the prohibited path and add a regression test.
- Review dependencies and agent-generated changes.
- Document the corrective change in the changelog and audit/release notes.
