# [Troubleshooting build failures](#troubleshooting-build-failures)

uv needs to build packages when there is not a compatible wheel (a pre-built distribution of the package) available. Building packages can fail for many reasons, some of which may be unrelated to uv itself. 

## [Recognizing a build failure](#recognizing-a-build-failure)

An example build failure can be produced by trying to install and old version of numpy on a new, unsupported version of Python: 

```
[#__codelineno-0-1](#__codelineno-0-1)$ uv pip install -p 3.13 'numpy<1.20'
[#__codelineno-0-2](#__codelineno-0-2)Resolved 1 package in 62ms
[#__codelineno-0-3](#__codelineno-0-3)  × Failed to build `numpy==1.19.5`
[#__codelineno-0-4](#__codelineno-0-4)  ├─▶ The build backend returned an error
[#__codelineno-0-5](#__codelineno-0-5)  ╰─▶ Call to `setuptools.build_meta:__legacy__.build_wheel()` failed (exit status: 1)
[#__codelineno-0-6](#__codelineno-0-6)
[#__codelineno-0-7](#__codelineno-0-7)      [stderr]
[#__codelineno-0-8](#__codelineno-0-8)      Traceback (most recent call last):
[#__codelineno-0-9](#__codelineno-0-9)        File "", line 8, in 
[#__codelineno-0-10](#__codelineno-0-10)          from setuptools.build_meta import __legacy__ as backend
[#__codelineno-0-11](#__codelineno-0-11)        File "/home/konsti/.cache/uv/builds-v0/.tmpi4bgKb/lib/python3.13/site-packages/setuptools/__init__.py", line 9, in 
[#__codelineno-0-12](#__codelineno-0-12)          import distutils.core
[#__codelineno-0-13](#__codelineno-0-13)      ModuleNotFoundError: No module named 'distutils'
[#__codelineno-0-14](#__codelineno-0-14)
[#__codelineno-0-15](#__codelineno-0-15)      hint: `distutils` was removed from the standard library in Python 3.12. Consider adding a constraint (like `numpy >1.19.5`) to avoid building a version of `numpy` that depends
[#__codelineno-0-16](#__codelineno-0-16)      on `distutils`.

```

Notice that the error message is prefaced by "The build backend returned an error". 

The build failure includes the `[stderr] `(and `[stdout] `, if present) from the build backend that was used for the build. The error logs are not from uv itself. 

The message following the `╰─▶ `is a hint provided by uv, to help resolve common build failures. A hint will not be available for all build failures. 

## [Confirming that a build failure is specific to uv](#confirming-that-a-build-failure-is-specific-to-uv)

Build failures are usually related to your system and the build backend. It is rare that a build failure is specific to uv. You can confirm that the build failure is not related to uv by attempting to reproduce it with pip: 

```
[#__codelineno-1-1](#__codelineno-1-1)$ uv venv -p 3.13 --seed
[#__codelineno-1-2](#__codelineno-1-2)$ source .venv/bin/activate
[#__codelineno-1-3](#__codelineno-1-3)$ pip install --use-pep517 --no-cache --force-reinstall 'numpy==1.19.5'
[#__codelineno-1-4](#__codelineno-1-4)Collecting numpy==1.19.5
[#__codelineno-1-5](#__codelineno-1-5)  Using cached numpy-1.19.5.zip (7.3 MB)
[#__codelineno-1-6](#__codelineno-1-6)  Installing build dependencies ... done
[#__codelineno-1-7](#__codelineno-1-7)  Getting requirements to build wheel ... done
[#__codelineno-1-8](#__codelineno-1-8)ERROR: Exception:
[#__codelineno-1-9](#__codelineno-1-9)Traceback (most recent call last):
[#__codelineno-1-10](#__codelineno-1-10)  ...
[#__codelineno-1-11](#__codelineno-1-11)  File "/Users/example/.cache/uv/archive-v0/3783IbOdglemN3ieOULx2/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 321, in _call_hook
[#__codelineno-1-12](#__codelineno-1-12)    raise BackendUnavailable(data.get('traceback', ''))
[#__codelineno-1-13](#__codelineno-1-13)pip._vendor.pyproject_hooks._impl.BackendUnavailable: Traceback (most recent call last):
[#__codelineno-1-14](#__codelineno-1-14)  File "/Users/example/.cache/uv/archive-v0/3783IbOdglemN3ieOULx2/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 77, in _build_backend
[#__codelineno-1-15](#__codelineno-1-15)    obj = import_module(mod_path)
[#__codelineno-1-16](#__codelineno-1-16)  File "/Users/example/.local/share/uv/python/cpython-3.13.0-macos-aarch64-none/lib/python3.13/importlib/__init__.py", line 88, in import_module
[#__codelineno-1-17](#__codelineno-1-17)    return _bootstrap._gcd_import(name[level:], package, level)
[#__codelineno-1-18](#__codelineno-1-18)           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[#__codelineno-1-19](#__codelineno-1-19)  File "", line 1387, in _gcd_import
[#__codelineno-1-20](#__codelineno-1-20)  File "", line 1360, in _find_and_load
[#__codelineno-1-21](#__codelineno-1-21)  File "", line 1310, in _find_and_load_unlocked
[#__codelineno-1-22](#__codelineno-1-22)  File "", line 488, in _call_with_frames_removed
[#__codelineno-1-23](#__codelineno-1-23)  File "", line 1387, in _gcd_import
[#__codelineno-1-24](#__codelineno-1-24)  File "", line 1360, in _find_and_load
[#__codelineno-1-25](#__codelineno-1-25)  File "", line 1331, in _find_and_load_unlocked
[#__codelineno-1-26](#__codelineno-1-26)  File "", line 935, in _load_unlocked
[#__codelineno-1-27](#__codelineno-1-27)  File "", line 1022, in exec_module
[#__codelineno-1-28](#__codelineno-1-28)  File "", line 488, in _call_with_frames_removed
[#__codelineno-1-29](#__codelineno-1-29)  File "/private/var/folders/6p/k5sd5z7j31b31pq4lhn0l8d80000gn/T/pip-build-env-vdpjme7d/overlay/lib/python3.13/site-packages/setuptools/__init__.py", line 9, in 
[#__codelineno-1-30](#__codelineno-1-30)    import distutils.core
[#__codelineno-1-31](#__codelineno-1-31)ModuleNotFoundError: No module named 'distutils'

```

!!! important "Important"

    The `--use-pep517 `flag should be included with the `pip install `invocation to ensure the same build isolation behavior. uv always uses [build isolation by default](../../../pip/compatibility/#pep-517-build-isolation). 

    We also recommend including the `--force-reinstall `and `--no-cache `options when reproducing failures. 

Since this build failure occurs in pip too, it is not likely to be a bug with uv. 

If a build failure is reproducible with another installer, you should investigate upstream (in this example, `numpy `or `setuptools `), find a way to avoid building the package in the first place, or make the necessary adjustments to your system for the build to succeed. 

## [Why does uv build a package?](#why-does-uv-build-a-package)

When generating the cross-platform lockfile, uv needs to determine the dependencies of all packages, even those only installed on other platforms. uv tries to avoid package builds during resolution. It uses any wheel if exist for that version, then tries to find static metadata in the source distribution (mainly pyproject.toml with static `project.version `, `project.dependencies `and `project.optional-dependencies `or METADATA v2.2+). Only if all of that fails, it builds the package. 

When installing, uv needs to have a wheel for the current platform for each package. If no matching wheel exists in the index, uv tries to build the source distribution. 

You can check which wheels exist for a PyPI project under “Download Files”, e.g. [https://pypi.org/project/numpy/2.1.1.md#files](https://pypi.org/project/numpy/2.1.1.md#files). Wheels with `...-py3-none-any.whl `filenames work everywhere, others have the operating system and platform in the filename. In the linked `numpy `example, you can see that there are pre-built distributions for Python 3.10 to 3.13 on macOS, Linux and Windows. 

## [Common build failures](#common-build-failures)

The following examples demonstrate common build failures and how to resolve them. 

### [Command is not found](#command-is-not-found)

If the build error mentions a missing command, for example, `gcc `: 

```
[#__codelineno-2-1](#__codelineno-2-1)× Failed to build `pysha3==1.0.2`
[#__codelineno-2-2](#__codelineno-2-2)├─▶ The build backend returned an error
[#__codelineno-2-3](#__codelineno-2-3)╰─▶ Call to `setuptools.build_meta:__legacy__.build_wheel` failed (exit status: 1)
[#__codelineno-2-4](#__codelineno-2-4)
[#__codelineno-2-5](#__codelineno-2-5)    [stdout]
[#__codelineno-2-6](#__codelineno-2-6)    running bdist_wheel
[#__codelineno-2-7](#__codelineno-2-7)    running build
[#__codelineno-2-8](#__codelineno-2-8)    running build_py
[#__codelineno-2-9](#__codelineno-2-9)    creating build/lib.linux-x86_64-cpython-310
[#__codelineno-2-10](#__codelineno-2-10)    copying sha3.py -> build/lib.linux-x86_64-cpython-310
[#__codelineno-2-11](#__codelineno-2-11)    running build_ext
[#__codelineno-2-12](#__codelineno-2-12)    building '_pysha3' extension
[#__codelineno-2-13](#__codelineno-2-13)    creating build/temp.linux-x86_64-cpython-310/Modules/_sha3
[#__codelineno-2-14](#__codelineno-2-14)    gcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -DPY_WITH_KECCAK=1 -I/root/.cache/uv/builds-v0/.tmp8V4iEk/include -I/usr/local/include/python3.10 -c
[#__codelineno-2-15](#__codelineno-2-15)    Modules/_sha3/sha3module.c -o build/temp.linux-x86_64-cpython-310/Modules/_sha3/sha3module.o
[#__codelineno-2-16](#__codelineno-2-16)
[#__codelineno-2-17](#__codelineno-2-17)    [stderr]
[#__codelineno-2-18](#__codelineno-2-18)    error: command 'gcc' failed: No such file or directory

```

Then, you'll need to install it with your system package manager, e.g., to resolve the error above: 

```
[#__codelineno-3-1](#__codelineno-3-1)$ apt install gcc

```

!!! tip "Tip"

    When using the uv-managed Python versions, it's common to need `clang `installed instead of `gcc `. 

    Many Linux distributions provide a package that includes all the common build dependencies. You can address most build requirements by installing it, e.g., for Debian or Ubuntu: 

    ```
[#__codelineno-4-1](#__codelineno-4-1)$ apt install build-essential

```

### [Header or library is missing](#header-or-library-is-missing)

If the build error mentions a missing header or library, e.g., a `.h `file, then you'll need to install it with your system package manager. 

For example, installing `pygraphviz `requires Graphviz to be installed: 

```
[#__codelineno-5-1](#__codelineno-5-1)× Failed to build `pygraphviz==1.14`
[#__codelineno-5-2](#__codelineno-5-2)├─▶ The build backend returned an error
[#__codelineno-5-3](#__codelineno-5-3)╰─▶ Call to `setuptools.build_meta.build_wheel` failed (exit status: 1)
[#__codelineno-5-4](#__codelineno-5-4)
[#__codelineno-5-5](#__codelineno-5-5)  [stdout]
[#__codelineno-5-6](#__codelineno-5-6)  running bdist_wheel
[#__codelineno-5-7](#__codelineno-5-7)  running build
[#__codelineno-5-8](#__codelineno-5-8)  running build_py
[#__codelineno-5-9](#__codelineno-5-9)  ...
[#__codelineno-5-10](#__codelineno-5-10)  gcc -fno-strict-overflow -Wsign-compare -DNDEBUG -g -O3 -Wall -fPIC -DSWIG_PYTHON_STRICT_BYTE_CHAR -I/root/.cache/uv/builds-v0/.tmpgLYPe0/include -I/usr/local/include/python3.12 -c pygraphviz/graphviz_wrap.c -o
[#__codelineno-5-11](#__codelineno-5-11)  build/temp.linux-x86_64-cpython-312/pygraphviz/graphviz_wrap.o
[#__codelineno-5-12](#__codelineno-5-12)
[#__codelineno-5-13](#__codelineno-5-13)  [stderr]
[#__codelineno-5-14](#__codelineno-5-14)  ...
[#__codelineno-5-15](#__codelineno-5-15)  pygraphviz/graphviz_wrap.c:9: warning: "SWIG_PYTHON_STRICT_BYTE_CHAR" redefined
[#__codelineno-5-16](#__codelineno-5-16)      9 | #define SWIG_PYTHON_STRICT_BYTE_CHAR
[#__codelineno-5-17](#__codelineno-5-17)        |
[#__codelineno-5-18](#__codelineno-5-18)  : note: this is the location of the previous definition
[#__codelineno-5-19](#__codelineno-5-19)  pygraphviz/graphviz_wrap.c:3023:10: fatal error: graphviz/cgraph.h: No such file or directory
[#__codelineno-5-20](#__codelineno-5-20)    3023 | #include "graphviz/cgraph.h"
[#__codelineno-5-21](#__codelineno-5-21)        |          ^~~~~~~~~~~~~~~~~~~
[#__codelineno-5-22](#__codelineno-5-22)  compilation terminated.
[#__codelineno-5-23](#__codelineno-5-23)  error: command '/usr/bin/gcc' failed with exit code 1
[#__codelineno-5-24](#__codelineno-5-24)
[#__codelineno-5-25](#__codelineno-5-25)  hint: This error likely indicates that you need to install a library that provides "graphviz/cgraph.h" for `[[email protected]](/cdn-cgi/l/email-protection)`

```

To resolve this error on Debian, you'd install the `libgraphviz-dev `package: 

```
[#__codelineno-6-1](#__codelineno-6-1)$ apt install libgraphviz-dev

```

Note that installing the `graphviz `package is not sufficient, the development headers need to be installed. 

!!! tip "Tip"

    To resolve an error where `Python.h `is missing, install the [`python3-dev `package](https://packages.debian.org/bookworm/python3-dev). 

### [Module is missing or cannot be imported](#module-is-missing-or-cannot-be-imported)

If the build error mentions a failing import, consider [disabling build isolation](../../../concepts/projects/config/#build-isolation). 

For example, some packages assume that `pip `is available without declaring it as a build dependency: 

```
[#__codelineno-7-1](#__codelineno-7-1)  × Failed to build `chumpy==0.70`
[#__codelineno-7-2](#__codelineno-7-2)  ├─▶ The build backend returned an error
[#__codelineno-7-3](#__codelineno-7-3)  ╰─▶ Call to `setuptools.build_meta:__legacy__.build_wheel` failed (exit status: 1)
[#__codelineno-7-4](#__codelineno-7-4)
[#__codelineno-7-5](#__codelineno-7-5)    [stderr]
[#__codelineno-7-6](#__codelineno-7-6)    Traceback (most recent call last):
[#__codelineno-7-7](#__codelineno-7-7)      File "", line 9, in 
[#__codelineno-7-8](#__codelineno-7-8)    ModuleNotFoundError: No module named 'pip'
[#__codelineno-7-9](#__codelineno-7-9)
[#__codelineno-7-10](#__codelineno-7-10)    During handling of the above exception, another exception occurred:
[#__codelineno-7-11](#__codelineno-7-11)
[#__codelineno-7-12](#__codelineno-7-12)    Traceback (most recent call last):
[#__codelineno-7-13](#__codelineno-7-13)      File "", line 14, in 
[#__codelineno-7-14](#__codelineno-7-14)      File "/root/.cache/uv/builds-v0/.tmpvvHaxI/lib/python3.12/site-packages/setuptools/build_meta.py", line 334, in get_requires_for_build_wheel
[#__codelineno-7-15](#__codelineno-7-15)        return self._get_build_requires(config_settings, requirements=[])
[#__codelineno-7-16](#__codelineno-7-16)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[#__codelineno-7-17](#__codelineno-7-17)      File "/root/.cache/uv/builds-v0/.tmpvvHaxI/lib/python3.12/site-packages/setuptools/build_meta.py", line 304, in _get_build_requires
[#__codelineno-7-18](#__codelineno-7-18)        self.run_setup()
[#__codelineno-7-19](#__codelineno-7-19)      File "/root/.cache/uv/builds-v0/.tmpvvHaxI/lib/python3.12/site-packages/setuptools/build_meta.py", line 522, in run_setup
[#__codelineno-7-20](#__codelineno-7-20)        super().run_setup(setup_script=setup_script)
[#__codelineno-7-21](#__codelineno-7-21)      File "/root/.cache/uv/builds-v0/.tmpvvHaxI/lib/python3.12/site-packages/setuptools/build_meta.py", line 320, in run_setup
[#__codelineno-7-22](#__codelineno-7-22)        exec(code, locals())
[#__codelineno-7-23](#__codelineno-7-23)      File "", line 11, in 
[#__codelineno-7-24](#__codelineno-7-24)    ModuleNotFoundError: No module named 'pip'

```

To resolve this error, pre-install the build dependencies then disable build isolation for the package: 

```
[#__codelineno-8-1](#__codelineno-8-1)$ uv pip install pip setuptools
[#__codelineno-8-2](#__codelineno-8-2)$ uv pip install chumpy --no-build-isolation-package chumpy

```

Note you will need to install the missing package, e.g., `pip `, _and _all the other build dependencies of the package, e.g, `setuptools `. 

### [Old version of the package is built](#old-version-of-the-package-is-built)

If a package fails to build during resolution and the version that failed to build is older than the version you want to use, try adding a [constraint](../../settings/#constraint-dependencies)with a lower bound (e.g., `numpy>=1.17 `). Sometimes, due to algorithmic limitations, the uv resolver tries to find a fitting version using unreasonably old packages, which can be prevented by using lower bounds. 

For example, when resolving the following dependencies on Python 3.10, uv attempts to build an old version of `apache-beam `. 

requirements.txt 

```
[#__codelineno-9-1](#__codelineno-9-1)dill<0.3.9,>=0.2.2
[#__codelineno-9-2](#__codelineno-9-2)apache-beam<=2.49.0

```

```
[#__codelineno-10-1](#__codelineno-10-1)× Failed to build `apache-beam==2.0.0`
[#__codelineno-10-2](#__codelineno-10-2)├─▶ The build backend returned an error
[#__codelineno-10-3](#__codelineno-10-3)╰─▶ Call to `setuptools.build_meta:__legacy__.build_wheel` failed (exit status: 1)
[#__codelineno-10-4](#__codelineno-10-4)
[#__codelineno-10-5](#__codelineno-10-5)    [stderr]
[#__codelineno-10-6](#__codelineno-10-6)    ...

```

Adding a lower bound constraint, e.g., `apache-beam < =2.49.0,>2.30.0 `, resolves this build failure as uv will avoid using an old version of `apache-beam `. 

Constraints can also be defined for indirect dependencies using `constraints.txt `files or the [`constraint-dependencies `](../../settings/#constraint-dependencies)setting. 

### [Old Version of a build dependency is used](#old-version-of-a-build-dependency-is-used)

If a package fails to build because `uv `selects an incompatible or outdated version of a build-time dependency, you can enforce constraints specifically for build dependencies. The [`build-constraint-dependencies `](../../settings/#build-constraint-dependencies)setting (or an analogous `build-constraints.txt `file) can be used to ensure that `uv `selects an appropriate version of a given build requirements. 

For example, the issue described in [#5551](https://github.com/astral-sh/uv/issues/5551#issuecomment-2256055975)could be addressed by specifying a build constraint that excludes `setuptools `version `72.0.0 `: 

pyproject.toml 

```
[#__codelineno-11-1](#__codelineno-11-1)[tool.uv]
[#__codelineno-11-2](#__codelineno-11-2)# Prevent setuptools version 72.0.0 from being used as a build dependency.
[#__codelineno-11-3](#__codelineno-11-3)build-constraint-dependencies = ["setuptools!=72.0.0"]

```

The build constraint will thus ensure that any package requiring `setuptools `during the build process will avoid using the problematic version, preventing build failures caused by incompatible build dependencies. 

### [Package is only needed for an unused platform](#package-is-only-needed-for-an-unused-platform)

If locking fails due to building a package from a platform you do not need to support, consider [limiting resolution](../../../concepts/projects/config/#limited-resolution-environments)to your supported platforms. 

### [Package does not support all Python versions](#package-does-not-support-all-python-versions)

If you support a large range of Python versions, consider using markers to use older versions for older Python versions and newer versions for newer Python version. For example, `numpy `only supports four Python minor version at a time, so to support a wider range of Python versions, e.g., Python 3.8 to 3.13, the `numpy `requirement needs to be split: 

```
[#__codelineno-12-1](#__codelineno-12-1)numpy>=1.23; python_version >= "3.10"
[#__codelineno-12-2](#__codelineno-12-2)numpy<1.23; python_version < "3.10"

```

### [Package is only usable on a specific platform](#package-is-only-usable-on-a-specific-platform)

If locking fails due to building a package that is only usable on another platform, you can [provide dependency metadata manually](../../settings/#dependency-metadata)to skip the build. uv can not verify this information, so it is important to specify correct metadata when using this override. 

June 10, 2025
