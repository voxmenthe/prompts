# [Dependency bots](#dependency-bots)

It is considered best practice to regularly update dependencies, to avoid being exposed to vulnerabilities, limit incompatibilities between dependencies, and avoid complex upgrades when upgrading from a too old version. A variety of tools can help staying up-to-date by creating automated pull requests. Several of them support uv, or have work underway to support it. 

## [Renovate](#renovate)

uv is supported by [Renovate](https://github.com/renovatebot/renovate). 

### [`uv.lock `output](#uvlock-output)

Renovate uses the presence of a `uv.lock `file to determine that uv is used for managing dependencies, and will suggest upgrades to [project dependencies](../../../concepts/projects/dependencies/#project-dependencies), [optional dependencies](../../../concepts/projects/dependencies/#optional-dependencies)and [development dependencies](../../../concepts/projects/dependencies/#development-dependencies). Renovate will update both the `pyproject.toml `and `uv.lock `files. 

The lockfile can also be refreshed on a regular basis (for instance to update transitive dependencies) by enabling the [`lockFileMaintenance `](https://docs.renovatebot.com/configuration-options/#lockfilemaintenance)option: 

renovate.json5 

```
[#__codelineno-0-1](#__codelineno-0-1){
[#__codelineno-0-2](#__codelineno-0-2)  $schema: "https://docs.renovatebot.com/renovate-schema.json",
[#__codelineno-0-3](#__codelineno-0-3)  lockFileMaintenance: {
[#__codelineno-0-4](#__codelineno-0-4)    enabled: true,
[#__codelineno-0-5](#__codelineno-0-5)  },
[#__codelineno-0-6](#__codelineno-0-6)}

```

### [Inline script metadata](#inline-script-metadata)

Renovate supports updating dependencies defined using [script inline metadata](../../scripts/#declaring-script-dependencies). 

Since it cannot automatically detect which Python files use script inline metadata, their locations need to be explicitly defined using [`fileMatch `](https://docs.renovatebot.com/configuration-options/#filematch), like so: 

renovate.json5 

```
[#__codelineno-1-1](#__codelineno-1-1){
[#__codelineno-1-2](#__codelineno-1-2)  $schema: "https://docs.renovatebot.com/renovate-schema.json",
[#__codelineno-1-3](#__codelineno-1-3)  pep723: {
[#__codelineno-1-4](#__codelineno-1-4)    fileMatch: [
[#__codelineno-1-5](#__codelineno-1-5)      "scripts/generate_docs\\.py",
[#__codelineno-1-6](#__codelineno-1-6)      "scripts/run_server\\.py",
[#__codelineno-1-7](#__codelineno-1-7)    ],
[#__codelineno-1-8](#__codelineno-1-8)  },
[#__codelineno-1-9](#__codelineno-1-9)}

```

## [Dependabot](#dependabot)

Dependabot has announced support for uv, but there are some use cases that are not yet working. See [astral-sh/uv#2512](https://github.com/astral-sh/uv/issues/2512)for updates. 

Dependabot supports updating `uv.lock `files. To enable it, add the uv `package-ecosystem `to your `updates `list in the `dependabot.yml `: 

dependabot.yml 

```
[#__codelineno-2-1](#__codelineno-2-1)version: 2
[#__codelineno-2-2](#__codelineno-2-2)
[#__codelineno-2-3](#__codelineno-2-3)updates:
[#__codelineno-2-4](#__codelineno-2-4)  - package-ecosystem: "uv"
[#__codelineno-2-5](#__codelineno-2-5)    directory: "/"
[#__codelineno-2-6](#__codelineno-2-6)    schedule:
[#__codelineno-2-7](#__codelineno-2-7)      interval: "weekly"

```

May 28, 2025
