# AgentGateway external processor

This service processes AgentGateway's Envoy `ext_proc` streams. It validates admitted requests, sends selected text to PII Engine, applies validated policy decisions, and restores request-local placeholders in eligible response fields. It does not scan all provider output for new personal information.

AgentGateway owns caller authentication, route admission, provider credentials, and upstream dispatch. PII Engine owns text detection, policy evaluation, and anonymization. This repository owns the adapter, shared request-segment package, and container image; Base owns deployment charts and gateway integration.

- [Architecture](architecture/README.md): traffic flow, supported formats, and protection boundaries.
- [Development](development/README.md): local checks, source layout, and builds.
- [Operations](operations/README.md): configuration, health checks, failures, and release ownership.

See the shared [platform architecture](https://github.com/neurwerk/docs/blob/main/dev/architecture/README.md) and [authentication guide](https://github.com/neurwerk/docs/blob/main/dev/authentication/README.md) for the surrounding services.
