# neurwerk.base - AgentGateway External Processor

AgentGateway External Processor adapts gateway requests and responses for the PII Engine in neurwerk.base. See the [neurwerk.base website](https://base.neurwerk.com/) for more information.

## Stack

| Repository | Purpose |
| --- | --- |
| [Base](https://github.com/neurwerk/k8s_stack_base) | Shared platform charts and release packages. |
| [Studio](https://github.com/neurwerk/k8s_stack_studio) | Platform dashboard and API. |
| [Tooling](https://github.com/neurwerk/k8s_stack_tooling) | Setup automation, operational tools, and operations image. |
| [PII Engine](https://github.com/neurwerk/k8s_stack_pii_engine) | Personal-information detection and safety policies. |
| [Gateway Processor](https://github.com/neurwerk/k8s_stack_agentgateway_extproc) | Connects gateway traffic to the PII Engine. **(this repository)** |
| [API Key Bridge](https://github.com/neurwerk/k8s_stack_keycloak_api_key_bridge) | Issues and validates API keys using Keycloak permissions. |
| [Keycloak Theme](https://github.com/neurwerk/k8s_stack_keycloak_theme) | Keycloak login and email themes. |

## Add-ons

| Repository | Purpose |
| --- | --- |
| [Dify](https://github.com/neurwerk/k8s_stack_addon_dify) | Dify deployment and API/web customizations. |
| [LibreChat](https://github.com/neurwerk/k8s_stack_addon_librechat) | LibreChat deployment and platform integration. |

## Examples

| Repository | Purpose |
| --- | --- |
| [Example Client](https://github.com/neurwerk/k8s_stack_client_example_com) | Example client settings and Flux deployment setup. |

## Documentation

See [Repository documentation](docs/README.md).

## Contributing and support

- **Contributions:** Read [CONTRIBUTING.md](.github/CONTRIBUTING.md) before proposing a change.
- **Bug reports and feature requests:** Use [GitHub Issues](https://github.com/neurwerk/k8s_stack_agentgateway_extproc/issues) for reproducible bugs and clearly scoped feature requests.

## Security

Report vulnerabilities privately by following the instructions in [SECURITY.md](SECURITY.md).

## Licensing

Project-owned content is licensed under the [MIT License](LICENSE). Third-party content retains its upstream license.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for bundled third-party licenses.
