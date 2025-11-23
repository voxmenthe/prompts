# [Getting help](#getting-help)

## [Help menus](#help-menus)

The `--help `flag can be used to view the help menu for a command, e.g., for `uv `: 

```
[#__codelineno-0-1](#__codelineno-0-1)$ uv --help

```

To view the help menu for a specific command, e.g., for `uv init `: 

```
[#__codelineno-1-1](#__codelineno-1-1)$ uv init --help

```

When using the `--help `flag, uv displays a condensed help menu. To view a longer help menu for a command, use `uv help `: 

```
[#__codelineno-2-1](#__codelineno-2-1)$ uv help

```

To view the long help menu for a specific command, e.g., for `uv init `: 

```
[#__codelineno-3-1](#__codelineno-3-1)$ uv help init

```

When using the long help menu, uv will attempt to use `less `or `more `to "page" the output so it is not all displayed at once. To exit the pager, press `q `. 

## [Displaying verbose output](#displaying-verbose-output)

The `-v `flag can be used to display verbose output for a command, e.g., for `uv sync `: 

```
[#__codelineno-4-1](#__codelineno-4-1)$ uv sync -v

```

The `-v `flag can be repeated to increase verbosity, e.g.: 

```
[#__codelineno-5-1](#__codelineno-5-1)$ uv sync -vv

```

Often, the verbose output will include additional information about why uv is behaving in a certain way. 

## [Viewing the version](#viewing-the-version)

When seeking help, it's important to determine the version of uv that you're using — sometimes the problem is already solved in a newer version. 

To check the installed version: 

```
[#__codelineno-6-1](#__codelineno-6-1)$ uv self version

```

The following are also valid: 

```
[#__codelineno-7-1](#__codelineno-7-1)$ uv --version      # Same output as `uv self version`
[#__codelineno-7-2](#__codelineno-7-2)$ uv -V             # Will not include the build commit and date

```

!!! note "Note"

    Before uv 0.7.0, `uv version `was used instead of `uv self version `. 

## [Troubleshooting issues](#troubleshooting-issues)

The reference documentation contains a [troubleshooting guide](../../reference/troubleshooting/)for common issues. 

## [Open an issue on GitHub](#open-an-issue-on-github)

The [issue tracker](https://github.com/astral-sh/uv/issues)on GitHub is a good place to report bugs and request features. Make sure to search for similar issues first, as it is common for someone else to encounter the same problem. 

## [Chat on Discord](#chat-on-discord)

Astral has a [Discord server](https://discord.com/invite/astral-sh), which is a great place to ask questions, learn more about uv, and engage with other community members. 

September 17, 2025
