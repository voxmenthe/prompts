# [Building and publishing a package](#building-and-publishing-a-package)

uv supports building Python packages into source and binary distributions via `uv build `and uploading them to a registry with `uv publish `. 

## [Preparing your project for packaging](#preparing-your-project-for-packaging)

Before attempting to publish your project, you'll want to make sure it's ready to be packaged for distribution. 

If your project does not include a `[build-system] `definition in the `pyproject.toml `, uv will not build it by default. This means that your project may not be ready for distribution. Read more about the effect of declaring a build system in the [project concept](../../concepts/projects/config/#build-systems)documentation. 

!!! note "Note"

    If you have internal packages that you do not want to be published, you can mark them as private: 

    ```
[#__codelineno-0-1](#__codelineno-0-1)[project]
[#__codelineno-0-2](#__codelineno-0-2)classifiers = ["Private :: Do Not Upload"]

```

    This setting makes PyPI reject your uploaded package from publishing. It does not affect security or privacy settings on alternative registries. 

    We also recommend only generating [per-project PyPI API tokens](https://pypi.org/help/#apitoken): Without a PyPI token matching the project, it can't be accidentally published. 

## [Building your package](#building-your-package)

Build your package with `uv build `: 

```
[#__codelineno-1-1](#__codelineno-1-1)$ uv build

```

By default, `uv build `will build the project in the current directory, and place the built artifacts in a `dist/ `subdirectory. 

Alternatively, `uv build `will build the package in the specified directory, while `uv build --package `will build the specified package within the current workspace. 

!!! info "Info"

    By default, `uv build `respects `tool.uv.sources `when resolving build dependencies from the `build-system.requires `section of the `pyproject.toml `. When publishing a package, we recommend running `uv build --no-sources `to ensure that the package builds correctly when `tool.uv.sources `is disabled, as is the case when using other build tools, like [`pypa/build `](https://github.com/pypa/build). 

## [Updating your version](#updating-your-version)

The `uv version `command provides conveniences for updating the version of your package before you publish it. [See the project docs for reading your package's version](../projects/#managing-version). 

To update to an exact version, provide it as a positional argument: 

```
[#__codelineno-2-1](#__codelineno-2-1)$ uv version 1.0.0
[#__codelineno-2-2](#__codelineno-2-2)hello-world 0.7.0 => 1.0.0

```

To preview the change without updating the `pyproject.toml `, use the `--dry-run `flag: 

```
[#__codelineno-3-1](#__codelineno-3-1)$ uv version 2.0.0 --dry-run
[#__codelineno-3-2](#__codelineno-3-2)hello-world 1.0.0 => 2.0.0
[#__codelineno-3-3](#__codelineno-3-3)$ uv version
[#__codelineno-3-4](#__codelineno-3-4)hello-world 1.0.0

```

To increase the version of your package semantics, use the `--bump `option: 

```
[#__codelineno-4-1](#__codelineno-4-1)$ uv version --bump minor
[#__codelineno-4-2](#__codelineno-4-2)hello-world 1.2.3 => 1.3.0

```

The `--bump `option supports the following common version components: `major `, `minor `, `patch `, `stable `, `alpha `, `beta `, `rc `, `post `, and `dev `. When provided more than once, the components will be applied in order, from largest ( `major `) to smallest ( `dev `). 

You can optionally provide a numeric value with `--bump = `to set the resulting component explicitly: 

```
[#__codelineno-5-1](#__codelineno-5-1)$ uv version --bump patch --bump dev=66463664
[#__codelineno-5-2](#__codelineno-5-2)hello-world 0.0.1 => 0.0.2.dev66463664

```

To move from a stable to pre-release version, bump one of the major, minor, or patch components in addition to the pre-release component: 

```
[#__codelineno-6-1](#__codelineno-6-1)$ uv version --bump patch --bump beta
[#__codelineno-6-2](#__codelineno-6-2)hello-world 1.3.0 => 1.3.1b1
[#__codelineno-6-3](#__codelineno-6-3)$ uv version --bump major --bump alpha
[#__codelineno-6-4](#__codelineno-6-4)hello-world 1.3.0 => 2.0.0a1

```

When moving from a pre-release to a new pre-release version, just bump the relevant pre-release component: 

```
[#__codelineno-7-1](#__codelineno-7-1)$ uv version --bump beta
[#__codelineno-7-2](#__codelineno-7-2)hello-world 1.3.0b1 => 1.3.0b2

```

When moving from a pre-release to a stable version, the `stable `option can be used to clear the pre-release component: 

```
[#__codelineno-8-1](#__codelineno-8-1)$ uv version --bump stable
[#__codelineno-8-2](#__codelineno-8-2)hello-world 1.3.1b2 => 1.3.1

```

!!! info "Info"

    By default, when `uv version `modifies the project it will perform a lock and sync. To prevent locking and syncing, use `--frozen `, or, to just prevent syncing, use `--no-sync `. 

## [Publishing your package](#publishing-your-package)

!!! note "Note"

    A complete guide to publishing from GitHub Actions to PyPI can be found in the [GitHub Guide](../integration/github/#publishing-to-pypi)

Publish your package with `uv publish `: 

```
[#__codelineno-9-1](#__codelineno-9-1)$ uv publish

```

Set a PyPI token with `--token `or `UV_PUBLISH_TOKEN `, or set a username with `--username `or `UV_PUBLISH_USERNAME `and password with `--password `or `UV_PUBLISH_PASSWORD `. For publishing to PyPI from GitHub Actions or another Trusted Publisher, you don't need to set any credentials. Instead, [add a trusted publisher to the PyPI project](https://docs.pypi.org/trusted-publishers/adding-a-publisher/). 

!!! note "Note"

    PyPI does not support publishing with username and password anymore, instead you need to generate a token. Using a token is equivalent to setting `--username __token__ `and using the token as password. 

If you're using a custom index through `[[tool.uv.index]] `, add `publish-url `and use `uv publish --index `. For example: 

```
[#__codelineno-10-1](#__codelineno-10-1)[[tool.uv.index]]
[#__codelineno-10-2](#__codelineno-10-2)name = "testpypi"
[#__codelineno-10-3](#__codelineno-10-3)url = "https://test.pypi.org/simple/"
[#__codelineno-10-4](#__codelineno-10-4)publish-url = "https://test.pypi.org/legacy/"
[#__codelineno-10-5](#__codelineno-10-5)explicit = true

```

!!! note "Note"

    When using `uv publish --index `, the `pyproject.toml `must be present, i.e., you need to have a checkout step in a publish CI job. 

Even though `uv publish `retries failed uploads, it can happen that publishing fails in the middle, with some files uploaded and some files still missing. With PyPI, you can retry the exact same command, existing identical files will be ignored. With other registries, use `--check-url `with the index URL (not the publishing URL) the packages belong to. When using `--index `, the index URL is used as check URL. uv will skip uploading files that are identical to files in the registry, and it will also handle raced parallel uploads. Note that existing files need to match exactly with those previously uploaded to the registry, this avoids accidentally publishing source distribution and wheels with different contents for the same version. 

## [Installing your package](#installing-your-package)

Test that the package can be installed and imported with `uv run `: 

```
[#__codelineno-11-1](#__codelineno-11-1)$ uv run --with  --no-project -- python -c "import "

```

The `--no-project `flag is used to avoid installing the package from your local project directory. 

!!! tip "Tip"

    If you have recently installed the package, you may need to include the `--refresh-package `option to avoid using a cached version of the package. 

## [Next steps](#next-steps)

To learn more about publishing packages, check out the [PyPA guides](https://packaging.python.org/en/latest/guides/section-build-and-publish/)on building and publishing. 

Or, read on for [guides](../integration/)on integrating uv with other software. 

November 11, 2025
