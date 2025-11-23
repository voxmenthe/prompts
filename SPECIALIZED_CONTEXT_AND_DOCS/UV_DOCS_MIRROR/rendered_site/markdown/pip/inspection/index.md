# [Inspecting environments](#inspecting-environments)

## [Listing installed packages](#listing-installed-packages)

To list all the packages in the environment: 

```
[#__codelineno-0-1](#__codelineno-0-1)$ uv pip list

```

To list the packages in a JSON format: 

```
[#__codelineno-1-1](#__codelineno-1-1)$ uv pip list --format json

```

To list all the packages in the environment in a `requirements.txt `format: 

```
[#__codelineno-2-1](#__codelineno-2-1)$ uv pip freeze

```

## [Inspecting a package](#inspecting-a-package)

To show information about an installed package, e.g., `numpy `: 

```
[#__codelineno-3-1](#__codelineno-3-1)$ uv pip show numpy

```

Multiple packages can be inspected at once. 

## [Verifying an environment](#verifying-an-environment)

It is possible to install packages with conflicting requirements into an environment if installed in multiple steps. 

To check for conflicts or missing dependencies in the environment: 

```
[#__codelineno-4-1](#__codelineno-4-1)$ uv pip check

```

June 10, 2025
