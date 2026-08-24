# Capabilities

Read `CAPABILITY_AVAILABILITY.json` for the machine-readable capability registry snapshot available to this experiment.

Use registered capabilities before attempting ad hoc package installation. If a required tool is not listed here or does not match the needed version, document that gap and only then install it yourself.

Install modes:
- `runtime_image`: already belongs in a launch/runtime profile; do not install inside the active Researcher container.
- `container_wrapper`: available as a command wrapper on `PATH`; the wrapper runs the registered container image through the inner Docker daemon.
- `manual_fallback`: instructions only; use when no generated wrapper exists.

Available capabilities: 0. Available versions: 0.
Generated wrappers: 0.
API search scopes in this launch: none.

| Key | Name | Version | Install mode | Tags |
| --- | --- | --- | --- | --- |
| None registered |  |  |  |  |

When using the API, search with `GET /capabilities/search?scopeType=<entry.scope.type>&scopeId=<entry.scope.id>&q=<tool-or-task>` for an advertised scope from `CAPABILITY_AVAILABILITY.json`, then inspect `GET /capability-versions/<id>/install-plan` if you need version details.
