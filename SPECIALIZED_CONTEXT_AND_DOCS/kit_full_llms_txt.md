<SYSTEM>This is the full developer documentation for kit </SYSTEM>

# 🛠️ kit documentation

![Kit Toolkit Logo](/kit.png)

Welcome to **kit** – the Python toolkit from [Cased](https://cased.com) for building LLM-powered developer tools and workflows.

kit shines for getting **precise, accurate, and relevant context** to LLMs. Use kit to build code reviewers, code generators and graphs, even full-fledged coding assistants: all enriched with the right code context.

MIT-licensed on [GitHub](https://github.com/cased/kit).

***

### Production-Ready AI PR Reviewer

kit includes an MIT-licensed, complete AI-powered pull request reviewer: professional-grade code analysis with full repository context, at cost. Choose any LLM, pay just for tokens with transparent pricing, priority filtering, CI/CD integration, more.

[📖 Full Documentation](/pr-reviewer)

[🛠️ Build Custom](/tutorials/ai_pr_reviewer)

#### Map any codebase

Get a structured view with file trees, language-aware symbol extraction (powered by tree-sitter), and dependency insights. Built-in support for 12+ languages, with intelligent caching.

#### Support for multiple search methods

Mix and match to optimize for speed, accuracy, and use case. Combine fast text search with semantic vector search to find relevant code snippets instantly.

#### Fine-grained docstring context

Use generated docstrings to find code snippets, answer questions, and improve code generation based on summarized content.

#### Build AI Workflows

Leverage ready-made utilities for code chunking, context retrieval, and interacting with LLMs. Enhanced with PR summarization and intelligent commit message generation.

## Explore the docs

[AI PR Reviewer](/pr-reviewer)

[Production-ready AI pull requests reviewer with 10 models, transparent pricing, CI/CD integration, and roadmap.](/pr-reviewer)

[Overview](/introduction/overview)

[What Kit is, why it exists, and how to install it.](/introduction/overview)

[Core Concepts](/core-concepts/repository-api)

[Deep-dive into the Repository API, symbol extraction, vector search, and architecture.](/core-concepts/repository-api)

[Tutorials](/tutorials/ai_pr_reviewer)

[Step-by-step tutorials to build real-world tools with Kit.](/tutorials/ai_pr_reviewer)

[API docs](/api/repository)

[Detailed API documentation for the primary classes.](/api/repository)

[Recipes](/recipes)

[Common patterns and code examples for working with Kit.](/recipes)

# CodeSearcher API

This page details the API for the `CodeSearcher` class, used for performing text and regular expression searches across your repository.

## Initialization

To use the `CodeSearcher`, you first need to initialize it with the path to your repository:

```python
from kit.code_searcher import CodeSearcher


searcher = CodeSearcher(repo_path="/path/to/your/repo")
# Or, if you have a kit.Repository object:
searcher = repo.get_code_searcher()
```

Note

If you are using the `kit.Repository` object, you can obtain a `CodeSearcher` instance via `repo.get_code_searcher()` which comes pre-configured with the repository path.

## `SearchOptions` Dataclass

The `search_text` method uses a `SearchOptions` dataclass to control search behavior. You can import it from `kit.code_searcher`.

```python
from kit.code_searcher import SearchOptions
```

**Fields:**

* `case_sensitive` (bool):

  * If `True` (default), the search query is case-sensitive.
  * If `False`, the search is case-insensitive.

* `context_lines_before` (int):
  * The number of lines to include before each matching line. Defaults to `0`.

* `context_lines_after` (int):
  * The number of lines to include after each matching line. Defaults to `0`.

* `use_gitignore` (bool):

  * If `True` (default), files and directories listed in the repository’s `.gitignore` file will be excluded from the search.
  * If `False`, `.gitignore` rules are ignored.

## Methods

### `search_text(query: str, file_pattern: str = "*.py", options: Optional[SearchOptions] = None) -> List[Dict[str, Any]]`

Searches for a text pattern (which can be a regular expression) in files matching the `file_pattern`.

* **Parameters:**

  * `query` (str): The text pattern or regular expression to search for.
  * `file_pattern` (str): A glob pattern specifying which files to search in. Defaults to `"*.py"` (all Python files).
  * `options` (Optional\[SearchOptions]): An instance of `SearchOptions` to customize search behavior. If `None`, default options are used.

* **Returns:**

  * `List[Dict[str, Any]]`: A list of dictionaries, where each dictionary represents a match and contains:

    * `"file"` (str): The relative path to the file from the repository root.
    * `"line_number"` (int): The 1-indexed line number where the match occurred.
    * `"line"` (str): The content of the matching line (with trailing newline stripped).
    * `"context_before"` (List\[str]): A list of strings, each being a line of context before the match.
    * `"context_after"` (List\[str]): A list of strings, each being a line of context after the match.

* **Raises:**
  * The method includes basic error handling for file operations and will print an error message to the console if a specific file cannot be processed, then continue with other files.

**Example Usage:**

```python
from kit.code_searcher import CodeSearcher, SearchOptions


# Assuming 'searcher' is an initialized CodeSearcher instance


# Basic search for 'my_function' in Python files
results_basic = searcher.search_text("my_function")


# Case-insensitive search with 2 lines of context before and after
custom_options = SearchOptions(
    case_sensitive=False,
    context_lines_before=2,
    context_lines_after=2
)
results_with_options = searcher.search_text(
    query=r"my_variable\s*=\s*\d+", # Example regex query
    file_pattern="*.txt",
    options=custom_options
)


for match in results_with_options:
    print(f"Found in {match['file']} at line {match['line_number']}:")
    for before_line in match['context_before']:
        print(f"  {before_line}")
    print(f"> {match['line']}")
    for after_line in match['context_after']:
        print(f"  {after_line}")
    print("---")
```

# DependencyAnalyzer API

> API documentation for the DependencyAnalyzer class and its language-specific implementations.

The `DependencyAnalyzer` class and its derivatives provide tools for analyzing dependencies between components in a codebase. These analyzers help you understand module relationships, detect circular dependencies, export dependency graphs, and generate visualization and LLM-friendly context about codebase architecture.

## Base Class

**Class: `DependencyAnalyzer`**\
*(defined in `kit/dependency_analyzer/dependency_analyzer.py`)*

`DependencyAnalyzer` is an abstract base class that defines the common interface for all language-specific dependency analyzers. You typically don’t instantiate this class directly; instead, use the factory method `Repository.get_dependency_analyzer(language)` to get the appropriate analyzer for your target language.

```python
from kit import Repository


repo = Repository("/path/to/your/codebase")
analyzer = repo.get_dependency_analyzer('python')  # or 'terraform'
```

### Constructor

```python
DependencyAnalyzer(repository: Repository)
```

**Parameters:**

* **`repository`** (`Repository`, required):\
  A Kit `Repository` instance that provides access to the codebase.

### Methods

#### `build_dependency_graph`

**Method: `DependencyAnalyzer.build_dependency_graph`**\
*(defined in `kit/dependency_analyzer/dependency_analyzer.py`)*

Analyzes the entire repository and builds a dependency graph.

```python
graph = analyzer.build_dependency_graph()
```

**Returns:**

* A dictionary representing the dependency graph where:

  * Keys are component identifiers (e.g., module names for Python, resource IDs for Terraform)
  * Values are dictionaries containing component metadata and dependencies

#### `export_dependency_graph`

**Method: `DependencyAnalyzer.export_dependency_graph`**\
*(defined in `kit/dependency_analyzer/dependency_analyzer.py`)*

Exports the dependency graph to various formats.

```python
# Export to JSON file
result = analyzer.export_dependency_graph(
    output_format="json",
    output_path="dependencies.json"
)


# Export to DOT file (for Graphviz)
result = analyzer.export_dependency_graph(
    output_format="dot",
    output_path="dependencies.dot"
)


# Export to GraphML file (for tools like Gephi or yEd)
result = analyzer.export_dependency_graph(
    output_format="graphml",
    output_path="dependencies.graphml"
)
```

**Parameters:**

* **`output_format`** (`str`, optional):\
  Format to export. One of: `"json"`, `"dot"`, `"graphml"`. Defaults to `"json"`.
* **`output_path`** (`str`, optional):\
  Path to save the output file. If `None`, returns the formatted data as a string.

**Returns:**

* If `output_path` is provided: Path to the output file
* If `output_path` is `None`: Formatted dependency data as a string

#### `find_cycles`

**Method: `DependencyAnalyzer.find_cycles`**\
*(defined in `kit/dependency_analyzer/dependency_analyzer.py`)*

Finds cycles (circular dependencies) in the dependency graph.

```python
cycles = analyzer.find_cycles()
if cycles:
    print(f"Found {len(cycles)} circular dependencies:")
    for cycle in cycles:
        print(f"  {' → '.join(cycle)} → {cycle[0]}")
```

**Returns:**

* A list of cycles, where each cycle is a list of component identifiers

#### `visualize_dependencies`

**Method: `DependencyAnalyzer.visualize_dependencies`**\
*(defined in `kit/dependency_analyzer/dependency_analyzer.py`)*

Generates a visualization of the dependency graph.

```python
# Generate a PNG visualization
viz_file = analyzer.visualize_dependencies(
    output_path="dependency_graph",
    format="png"
)
```

**Parameters:**

* **`output_path`** (`str`, required):\
  Path to save the visualization (without extension).
* **`format`** (`str`, optional):\
  Output format. One of: `"png"`, `"svg"`, `"pdf"`. Defaults to `"png"`.

**Returns:**

* Path to the generated visualization file

#### `generate_llm_context`

**Method: `DependencyAnalyzer.generate_llm_context`**\
*(defined in `kit/dependency_analyzer/dependency_analyzer.py`)*

Generates a concise, natural language description of the dependency graph optimized for LLM consumption.

```python
# Generate markdown context
context = analyzer.generate_llm_context(
    max_tokens=4000,
    output_format="markdown",
    output_path="dependency_context.md"
)


# Or generate plain text context
context = analyzer.generate_llm_context(
    max_tokens=4000,
    output_format="text",
    output_path="dependency_context.txt"
)
```

**Parameters:**

* **`max_tokens`** (`int`, optional):\
  Approximate maximum number of tokens in the output (rough guideline). Defaults to 4000.
* **`output_format`** (`str`, optional):\
  Format of the output. One of: `"markdown"`, `"text"`. Defaults to `"markdown"`.
* **`output_path`** (`str`, optional):\
  Path to save the output to a file. If `None`, returns the formatted string.

**Returns:**

* A string containing the natural language description of the dependency structure

#### Factory Method: `get_for_language`

**Method: `DependencyAnalyzer.get_for_language`**\
*(defined in `kit/dependency_analyzer/dependency_analyzer.py`)*

Factory method to get an appropriate `DependencyAnalyzer` for the specified language. This is typically used internally by the `Repository.get_dependency_analyzer` method.

```python
analyzer = DependencyAnalyzer.get_for_language(repository, "python")
```

**Parameters:**

* **`repository`** (`Repository`, required):\
  A Kit `Repository` instance.
* **`language`** (`str`, required):\
  Language identifier (e.g., `"python"`, `"terraform"`).

**Returns:**

* An appropriate `DependencyAnalyzer` instance for the language

## Language-Specific Implementations

### PythonDependencyAnalyzer

**Class: `PythonDependencyAnalyzer`**\
*(defined in `kit/dependency_analyzer/python_dependency_analyzer.py`)*

The `PythonDependencyAnalyzer` extends the base `DependencyAnalyzer` to analyze Python codebases, focusing on import relationships between modules.

#### Additional Methods

##### `get_module_dependencies`

**Method: `PythonDependencyAnalyzer.get_module_dependencies`**\
*(defined in `kit/dependency_analyzer/python_dependency_analyzer.py`)*

Gets dependencies for a specific Python module.

```python
# Get direct dependencies
deps = python_analyzer.get_module_dependencies("my_package.my_module")


# Get all dependencies (including indirect)
all_deps = python_analyzer.get_module_dependencies(
    "my_package.my_module",
    include_indirect=True
)
```

**Parameters:**

* **`module_name`** (`str`, required):\
  Name of the module to check.
* **`include_indirect`** (`bool`, optional):\
  Whether to include indirect dependencies. Defaults to `False`.

**Returns:**

* List of module names this module depends on

##### `get_file_dependencies`

**Method: `PythonDependencyAnalyzer.get_file_dependencies`**\
*(defined in `kit/dependency_analyzer/python_dependency_analyzer.py`)*

Gets detailed dependency information for a specific file.

```python
file_deps = python_analyzer.get_file_dependencies("path/to/file.py")
```

**Parameters:**

* **`file_path`** (`str`, required):\
  Path to the file to analyze.

**Returns:**

* Dictionary with dependency information for the file

##### `generate_dependency_report`

**Method: `PythonDependencyAnalyzer.generate_dependency_report`**\
*(defined in `kit/dependency_analyzer/python_dependency_analyzer.py`)*

Generates a comprehensive dependency report for the repository.

```python
report = python_analyzer.generate_dependency_report(
    output_path="dependency_report.json"
)
```

**Parameters:**

* **`output_path`** (`str`, optional):\
  Path to save the report JSON. If `None`, returns the report data without saving.

**Returns:**

* Dictionary with the complete dependency report

### TerraformDependencyAnalyzer

**Class: `TerraformDependencyAnalyzer`**\
*(defined in `kit/dependency_analyzer/terraform_dependency_analyzer.py`)*

The `TerraformDependencyAnalyzer` extends the base `DependencyAnalyzer` to analyze Terraform (HCL) codebases, focusing on relationships between infrastructure resources, modules, variables, and other Terraform components.

#### Additional Methods

##### `get_resource_dependencies`

**Method: `TerraformDependencyAnalyzer.get_resource_dependencies`**\
*(defined in `kit/dependency_analyzer/terraform_dependency_analyzer.py`)*

Gets dependencies for a specific Terraform resource.

```python
# Get direct dependencies
deps = terraform_analyzer.get_resource_dependencies("aws_s3_bucket.example")


# Get all dependencies (including indirect)
all_deps = terraform_analyzer.get_resource_dependencies(
    "aws_s3_bucket.example",
    include_indirect=True
)
```

**Parameters:**

* **`resource_id`** (`str`, required):\
  ID of the resource to check (e.g., `"aws_s3_bucket.example"`).
* **`include_indirect`** (`bool`, optional):\
  Whether to include indirect dependencies. Defaults to `False`.

**Returns:**

* List of resource IDs this resource depends on

##### `get_resource_by_type`

**Method: `TerraformDependencyAnalyzer.get_resource_by_type`**\
*(defined in `kit/dependency_analyzer/terraform_dependency_analyzer.py`)*

Finds all resources of a specific type.

```python
# Find all S3 buckets
s3_buckets = terraform_analyzer.get_resource_by_type("aws_s3_bucket")
```

**Parameters:**

* **`resource_type`** (`str`, required):\
  Type of resource to find (e.g., `"aws_s3_bucket"`).

**Returns:**

* List of resource IDs matching the specified type

##### `generate_resource_documentation`

**Method: `TerraformDependencyAnalyzer.generate_resource_documentation`**\
*(defined in `kit/dependency_analyzer/terraform_dependency_analyzer.py`)*

Generates documentation for Terraform resources in the codebase.

```python
docs = terraform_analyzer.generate_resource_documentation(
    output_path="terraform_resources.md"
)
```

**Parameters:**

* **`output_path`** (`str`, optional):\
  Path to save the documentation. If `None`, returns the documentation string.

**Returns:**

* String containing the markdown documentation of resources

## Key Features and Notes

* All dependency analyzers store absolute file paths for resources, making it easy to locate components in complex codebases with files that might have the same name in different directories.

* The `generate_llm_context` method produces summaries specially formatted for use as context with LLMs, focusing on the most significant patterns and keeping the token count manageable.

* Visualizations require the Graphviz software to be installed on your system.

* The dependency graph is built on first use and cached. If the codebase changes, you may need to call `build_dependency_graph()` again to refresh the analysis.

# DocstringIndexer API

> API documentation for the DocstringIndexer class.

The `DocstringIndexer` class is responsible for building a vector index of AI-generated code summaries (docstrings). It processes files in a repository, generates summaries for code symbols (or entire files), embeds these summaries, and stores them in a configurable vector database backend. Once an index is built, it can be queried using the [`SummarySearcher`](/api/summary-searcher) class.

## Constructor

**Class: `DocstringIndexer`** *(defined in `kit/docstring_indexer.py`)*

```python
from kit import Repository, Summarizer
from kit.docstring_indexer import DocstringIndexer, EmbedFn # EmbedFn is Optional[Callable[[str], List[float]]]
from kit.vector_searcher import VectorDBBackend # Optional


# Example basic initialization
repo = Repository("/path/to/your/repo")
summarizer = Summarizer() # Assumes OPENAI_API_KEY is set or local model configured
indexer = DocstringIndexer(repo=repo, summarizer=summarizer)


# Example with custom embedding function and backend
# def my_custom_embed_fn(text: str) -> List[float]:
#     # ... your embedding logic ...
#     return [0.1, 0.2, ...]
#
# from kit.vector_searcher import ChromaDBBackend
# custom_backend = ChromaDBBackend(collection_name="my_custom_index", persist_dir="./my_chroma_db")
#
# indexer_custom = DocstringIndexer(
#     repo=repo,
#     summarizer=summarizer,
#     embed_fn=my_custom_embed_fn,
#     backend=custom_backend,
#     persist_dir="./my_custom_index_explicit_persist" # Can also be set directly on backend
# )
```

**Parameters:**

* **`repo`** (`Repository`, required): An instance of `kit.Repository` pointing to the codebase to be indexed.
* **`summarizer`** (`Summarizer`, required): An instance of `kit.Summarizer` used to generate summaries for code symbols or files.
* **`embed_fn`** (`Optional[Callable[[str], List[float]]]`, default: `SentenceTransformer('all-MiniLM-L6-v2')`): A function that takes a string and returns its embedding (a list of floats). If `None`, a default embedding function using `sentence-transformers` (`all-MiniLM-L6-v2` model) will be used. The `sentence-transformers` package must be installed for the default to work (`pip install sentence-transformers`).
* **`backend`** (`Optional[VectorDBBackend]`, default: `ChromaDBBackend`): The vector database backend to use for storing and querying embeddings. If `None`, a `ChromaDBBackend` instance will be created. The default collection name is `kit_docstring_index`.
* **`persist_dir`** (`Optional[str]`, default: `'./.kit_index/' + repo_name_slug + '/docstrings'`): The directory where the vector database (e.g., ChromaDB) should persist its data. If `None`, a default path is constructed based on the repository name within a `.kit_index` directory in the current working directory. If a custom `backend` is provided, this parameter might be ignored if the backend itself has persistence configured. It’s primarily used for the default `ChromaDBBackend` if no explicit `backend` is given or if the default backend needs a specific persistence path.

## Methods

### `build`

**Method: `DocstringIndexer.build`** *(defined in `kit/docstring_indexer.py`)*

Builds or rebuilds the docstring index. It iterates through files in the repository (respecting `.gitignore` and `file_extensions`), extracts symbols or uses whole file content based on the `level`, generates summaries, embeds them, and adds them to the vector database. It also handles caching to avoid re-processing unchanged symbols/files.

```python
# Build the index (symbol-level by default for .py files)
indexer.build()


# Force a rebuild, ignoring any existing cache
indexer.build(force=True)


# Index at file level instead of symbol level
indexer.build(level="file")


# Index only specific file extensions
indexer.build(file_extensions=[".py", ".mdx"])
```

**Parameters:**

* **`force`** (`bool`, default: `False`): If `True`, the entire index is rebuilt, ignoring any existing cache and potentially overwriting existing data in the backend. If `False`, uses cached summaries/embeddings for unchanged code and only processes new/modified code. It also avoids re-initializing the backend if it already contains data, unless changes are detected.

* **`level`** (`str`, default: `'symbol'`): The granularity of indexing.

  * `'symbol'`: Extracts and summarizes individual symbols (functions, classes, methods) from files.
  * `'file'`: Summarizes the entire content of each file.

* **`file_extensions`** (`Optional[List[str]]`, default: `None` (uses Repository’s default, typically .py)): A list of file extensions (e.g., `['.py', '.md']`) to include in the indexing process. If `None`, uses the default behavior of the `Repository` instance, which typically focuses on Python files but can be configured.

**Returns:** `None`

### `get_searcher`

**Method: `DocstringIndexer.get_searcher`** *(defined in `kit/docstring_indexer.py`)*

Returns a `SummarySearcher` instance that is configured to query the index managed by this `DocstringIndexer`.

This provides a convenient way to obtain a search interface after the indexer has been built or loaded, without needing to manually instantiate `SummarySearcher`.

```python
# Assuming 'indexer' is an initialized DocstringIndexer instance
# indexer.build() # or it has been loaded with a pre-built index


search_interface = indexer.get_searcher()
results = search_interface.search("my search query", top_k=3)


for result in results:
    print(result)
```

**Parameters:** None

**Returns:** `SummarySearcher`

An instance of `SummarySearcher` linked to this indexer.

# Repository API

> Complete reference for Repository class methods and properties

# Repository API

The `Repository` class is the main entry point for analyzing code repositories with Kit.

## Core Methods

### `get_file_tree(subpath=None)`

Returns the file tree structure of the repository.

**Parameters:**

* `subpath` (str, optional): Subdirectory path relative to repo root. If None, returns entire repo tree. If specified, returns tree starting from that subdirectory.

**Returns:**

* `List[Dict[str, Any]]`: List of file/directory objects with `path`, `is_dir`, `name`, and `size` fields.

**Examples:**

```python
from kit import Repository


repo = Repository("/path/to/repo")


# Get full repository tree
full_tree = repo.get_file_tree()


# Get tree for specific subdirectory
src_tree = repo.get_file_tree(subpath="src")
components_tree = repo.get_file_tree(subpath="src/components")
```

### `grep(pattern, *, case_sensitive=True, include_pattern=None, exclude_pattern=None, max_results=1000)`

Performs literal grep search on repository files using system grep.

**Parameters:**

* `pattern` (str): The literal string to search for (not a regex)
* `case_sensitive` (bool): Whether the search should be case sensitive. Defaults to True
* `include_pattern` (str, optional): Glob pattern for files to include (e.g. ‘\*.py’)
* `exclude_pattern` (str, optional): Glob pattern for files to exclude
* `max_results` (int): Maximum number of results to return. Defaults to 1000

**Returns:**

* `List[Dict[str, Any]]`: List of matches with `file`, `line_number`, and `line_content` fields

**Examples:**

```python
from kit import Repository


repo = Repository("/path/to/repo")


# Basic search
matches = repo.grep("TODO")


# Case insensitive search
matches = repo.grep("function", case_sensitive=False)


# Search only Python files
matches = repo.grep("class", include_pattern="*.py")


# Exclude test files
matches = repo.grep("import", exclude_pattern="*test*")


# Limit results
matches = repo.grep("error", max_results=50)
```

**Note:** Requires system `grep` command to be available in PATH.

### `extract_symbols(file_path=None)`

Extracts symbols (functions, classes, etc.) from repository files.

**Parameters:**

* `file_path` (str, optional): Specific file to extract from. If None, extracts from all files.

**Returns:**

* `List[Dict[str, Any]]`: List of symbol objects with `name`, `type`, `start_line`, `end_line`, and `code` fields.

### `search_text(query, file_pattern="*")`

Searches for text patterns in repository files using Kit’s built-in search.

**Parameters:**

* `query` (str): Text or regex pattern to search for
* `file_pattern` (str): Glob pattern for files to search. Defaults to ”\*”

**Returns:**

* `List[Dict[str, Any]]`: List of search results

### `get_file_content(file_path)`

Reads and returns file content(s).

**Parameters:**

* `file_path` (str | List\[str]): Single file path or list of file paths

**Returns:**

* `str | Dict[str, str]`: File content (single) or mapping of path → content (multiple)

## Properties

### `current_sha`

Current git commit SHA (full).

### `current_sha_short`

Current git commit SHA (short).

### `current_branch`

Current git branch name.

### `remote_url`

Git remote URL.

### `is_dirty`

Whether the working directory has uncommitted changes.

## Utility Methods

### `get_abs_path(relative_path)`

Converts relative path to absolute path within repository.

### `index()`

Returns comprehensive repository index with file tree and symbols.

### `write_symbols(file_path, symbols=None)`

Writes symbols to JSON file.

### `write_file_tree(file_path)`

Writes file tree to JSON file.

### `find_symbol_usages(symbol_name, symbol_type=None)`

Finds all usages of a symbol across the repository.

## Examples

### Basic Repository Analysis

```python
from kit import Repository


# Open repository
repo = Repository("/path/to/project")


# Get repository structure
tree = repo.get_file_tree()
print(f"Found {len(tree)} files and directories")


# Extract all symbols
symbols = repo.extract_symbols()
print(f"Found {len(symbols)} symbols")


# Search for TODOs
todos = repo.grep("TODO", case_sensitive=False)
print(f"Found {len(todos)} TODO items")
```

### Working with Subdirectories

```python
# Analyze only source code directory
src_tree = repo.get_file_tree(subpath="src")
src_symbols = repo.extract_symbols("src/main.py")


# Search within specific directory
api_todos = repo.grep("TODO", include_pattern="src/api/*.py")
```

### File Content Operations

```python
# Single file
content = repo.get_file_content("README.md")


# Multiple files
contents = repo.get_file_content(["src/main.py", "src/utils.py"])
for file_path, content in contents.items():
    print(f"{file_path}: {len(content)} characters")
```

## Initialization

```python
from kit import Repository


# Initialize with a local path
local_repo = Repository("/path/to/your/local/project")


# Initialize with a remote URL (requires git)
# remote_repo = Repository("https://github.com/user/repo.git")


# Initialize at a specific commit, tag, or branch
# versioned_repo = Repository("https://github.com/user/repo.git", ref="v1.2.3")
```

## Incremental Analysis Methods

* `extract_symbols_incremental()`: High-performance symbol extraction with caching.
* `get_incremental_stats()`: Get cache performance statistics.
* `cleanup_incremental_cache()`: Remove stale cache entries.
* `clear_incremental_cache()`: Clear all cached data.

## Creating a `Repository` Instance

To start using `kit`, first create an instance of the `Repository` class. This points `kit` to the codebase you want to analyze.

```python
from kit import Repository


# For a local directory
repository_instance = Repository(path_or_url="/path/to/local/project")


# For a remote Git repository (public or private)
# repository_instance = Repository(
#     path_or_url="https://github.com/owner/repo-name",
#     github_token="YOUR_GITHUB_TOKEN",  # Optional: For private repos
#     cache_dir="/path/to/cache",        # Optional: For caching clones
#     ref="v1.2.3",                       # Optional: Specific commit, tag, or branch
#     cache_ttl_hours=2,                  # Optional: New in v1.2.0
# )
```

**Parameters:**

* `path_or_url` (str): The path to a local directory or the URL of a remote Git repository.
* `github_token` (Optional\[str]): A GitHub personal access token required for cloning private repositories. If not provided, automatically checks the `KIT_GITHUB_TOKEN` and `GITHUB_TOKEN` environment variables. Defaults to `None`.
* `cache_dir` (Optional\[str]): Path to a directory for caching cloned repositories. Defaults to a system temporary directory.
* `ref` (Optional\[str]): Git reference (commit SHA, tag, or branch) to checkout. For remote repositories, this determines which version to clone. For local repositories, this will checkout the specified ref. Defaults to `None`.
* `cache_ttl_hours` (Optional\[float]): **New in v1.2.0.** When cloning a remote repository, Kit stores the clone under `tmp/kit-repo-cache` (or your custom `cache_dir`). On the *first* clone in each Python process Kit will delete any cached repo directory whose **modification time** is older than this many hours before continuing. Pass `None` (default) or `0` to disable the cleanup. You can also set the global environment variable `KIT_TMP_REPO_TTL_HOURS` to apply the policy process-wide without changing code.

Tip

**Automatic GitHub Token Pickup**

For convenience, the Repository class automatically checks for GitHub tokens in environment variables when cloning remote repositories:

1. First checks `KIT_GITHUB_TOKEN`
2. Falls back to `GITHUB_TOKEN` if `KIT_GITHUB_TOKEN` is not set
3. Uses `None` if neither environment variable is set

This means you can set `export KIT_GITHUB_TOKEN="ghp_your_token"` and omit the `github_token` parameter entirely.

Once you have a `repository` object, you can call the following methods on it:

## `repository.get_file_tree()`

Returns the file tree structure of the repository.

```python
repository.get_file_tree() -> List[Dict[str, Any]]
```

**Returns:**

* `List[Dict[str, Any]]`: A list of dictionaries, where each dictionary represents a file or directory with keys like `path`, `name`, `is_dir`, `size`.

## `repository.get_file_content()`

Reads and returns the content of a specified file within the repository as a string.

```python
repository.get_file_content(file_path: str) -> str
```

**Parameters:**

* `file_path` (str): The path to the file, relative to the repository root.

**Returns:**

* `str`: The content of the file.

**Raises:**

* `FileNotFoundError`: If the file does not exist at the specified path.
* `IOError`: If any other I/O error occurs during file reading.

## `repository.extract_symbols()`

Extracts code symbols (functions, classes, variables, etc.) from the repository.

```python
repository.extract_symbols(file_path: Optional[str] = None) -> List[Dict[str, Any]]
```

**Parameters:**

* `file_path` (Optional\[str]): If provided, extracts symbols only from this specific file path relative to the repo root. If `None`, extracts symbols from all supported files in the repository. Defaults to `None`.

**Returns:**

* `List[Dict[str, Any]]`: A list of dictionaries, each representing a symbol with keys like `name`, `type`, `file`, `line_start`, `line_end`, `code`.

## `repository.extract_symbols_incremental()`

Extracts code symbols with intelligent caching for dramatically improved performance. Uses multiple invalidation strategies (mtime, size, content hash, git state) to ensure accuracy while providing 25-36x speedups for warm cache scenarios.

```python
repository.extract_symbols_incremental(file_path: Optional[str] = None) -> List[Dict[str, Any]]
```

**Parameters:**

* `file_path` (Optional\[str]): If provided, extracts symbols only from this specific file path relative to the repo root. If `None`, extracts symbols from all supported files in the repository. Defaults to `None`.

**Returns:**

* `List[Dict[str, Any]]`: A list of dictionaries, each representing a symbol with keys like `name`, `type`, `file`, `line_start`, `line_end`, `code`.

**Performance:**

* **Cold cache**: Full analysis with cache building
* **Warm cache**: 25x faster using cached results
* **Automatic invalidation**: Cache invalidated on git state changes, file modifications

**Example:**

```python
# First call builds cache
symbols = repository.extract_symbols_incremental()
print(f"Found {len(symbols)} symbols")


# Subsequent calls use cache (much faster)
symbols = repository.extract_symbols_incremental()
print(f"Found {len(symbols)} symbols (cached)")


# Check performance
stats = repository.get_incremental_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']}")
```

## `repository.get_incremental_stats()`

Returns performance statistics for the incremental analysis system.

```python
repository.get_incremental_stats() -> Dict[str, Any]
```

**Returns:**

* `Dict[str, Any]`: Statistics dictionary containing:

  * `cache_hit_rate`: Percentage of cache hits (0.0-1.0)
  * `files_analyzed`: Number of files analyzed in last operation
  * `cache_hits`: Total cache hits
  * `cache_misses`: Total cache misses
  * `avg_analysis_time`: Average time per file analysis
  * `cache_size_mb`: Cache size in megabytes

**Example:**

```python
stats = repository.get_incremental_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Files analyzed: {stats['files_analyzed']}")
print(f"Cache size: {stats['cache_size_mb']:.1f}MB")
```

## `repository.cleanup_incremental_cache()`

Removes stale cache entries for files that no longer exist in the repository.

```python
repository.cleanup_incremental_cache() -> int
```

**Returns:**

* `int`: Number of stale entries removed.

**Example:**

```python
removed_count = repository.cleanup_incremental_cache()
print(f"Removed {removed_count} stale cache entries")
```

## `repository.clear_incremental_cache()`

Clears all cached data for the incremental analysis system.

```python
repository.clear_incremental_cache() -> None
```

**Example:**

```python
repository.clear_incremental_cache()
print("Cache cleared")
```

## `repository.search_text()`

Searches for literal text or regex patterns within files.

```python
repository.search_text(query: str, file_pattern: str = "*.py") -> List[Dict[str, Any]]
```

**Parameters:**

* `query` (str): The text or regex pattern to search for.
* `file_pattern` (str): A glob pattern to filter files to search within. Defaults to `"*.py"`.

**Returns:**

* `List[Dict[str, Any]]`: A list of dictionaries representing search matches, with keys like `file`, `line_number`, `line_content`.

## `repository.chunk_file_by_lines()`

Chunks a file’s content based on line count.

```python
repository.chunk_file_by_lines(file_path: str, max_lines: int = 50) -> List[str]
```

**Parameters:**

* `file_path` (str): The path to the file (relative to repo root) to chunk.
* `max_lines` (int): The maximum number of lines per chunk. Defaults to `50`.

**Returns:**

* `List[str]`: A list of strings, where each string is a chunk of the file content.

## `repository.chunk_file_by_symbols()`

Chunks a file’s content based on its top-level symbols (functions, classes).

```python
repository.chunk_file_by_symbols(file_path: str) -> List[Dict[str, Any]]
```

**Parameters:**

* `file_path` (str): The path to the file (relative to repo root) to chunk.

**Returns:**

* `List[Dict[str, Any]]`: A list of dictionaries, each representing a symbol chunk with keys like `name`, `type`, `code`.

## `repository.extract_context_around_line()`

Extracts the surrounding code context (typically the containing function or class) for a specific line number.

```python
repository.extract_context_around_line(file_path: str, line: int) -> Optional[Dict[str, Any]]
```

**Parameters:**

* `file_path` (str): The path to the file (relative to repo root).
* `line` (int): The (0-based) line number to find context for.

**Returns:**

* `Optional[Dict[str, Any]]`: A dictionary representing the symbol context (with keys like `name`, `type`, `code`), or `None` if no context is found.

## `repository.index()`

Builds and returns a comprehensive index of the repository, including both the file tree and all extracted symbols.

```python
repository.index() -> Dict[str, Any]
```

**Returns:**

* `Dict[str, Any]`: A dictionary containing the full index, typically with keys like `file_tree` and `symbols`.

## `repository.get_vector_searcher()`

Initializes and returns the `VectorSearcher` instance for performing semantic search.

```python
repository.get_vector_searcher(embed_fn=None, backend=None, persist_dir=None) -> VectorSearcher
```

**Parameters:**

* `embed_fn` (Callable): **Required on first call.** A function that takes a list of strings and returns a list of embedding vectors.
* `backend` (Optional\[Any]): Specifies the vector database backend. If `None`, `kit` defaults to using `ChromaDBBackend`.
* `persist_dir` (Optional\[str]): Path to a directory to persist the vector index. If `None`, the `VectorSearcher` will default to `YOUR_REPO_PATH/.kit/vector_db/` for ChromaDB. Setting to `None` implies using this default persistence path for ChromaDB.

**Returns:**

* `VectorSearcher`: An instance of the vector searcher configured for this repository.

(See [Configuring Semantic Search](/core-concepts/configuring-semantic-search) for more details.)

## `repository.search_semantic()`

Performs a semantic search query over the indexed codebase.

```python
repository.search_semantic(query: str, top_k: int = 5, embed_fn=None) -> List[Dict[str, Any]]
```

**Parameters:**

* `query` (str): The natural language query to search for.
* `top_k` (int): The maximum number of results to return. Defaults to `5`.
* `embed_fn` (Callable): Required if the vector searcher hasn’t been initialized yet.

**Returns:**

* `List[Dict[str, Any]]`: A list of dictionaries representing the search results, typically including matched code snippets and relevance scores.

## `repository.find_symbol_usages()`

Finds definitions and references of a specific symbol across the repository.

```python
repository.find_symbol_usages(symbol_name: str, symbol_type: Optional[str] = None) -> List[Dict[str, Any]]
```

**Parameters:**

* `symbol_name` (str): The name of the symbol to find usages for.
* `symbol_type` (Optional\[str]): Optionally restrict the search to a specific symbol type (e.g., ‘function’, ‘class’). Defaults to `None` (search all types).

**Returns:**

* `List[Dict[str, Any]]`: A list of dictionaries representing symbol usages, including file, line number, and context/snippet.

## `repository.write_index()`

Writes the full repository index (file tree and symbols) to a JSON file.

```python
repository.write_index(file_path: str) -> None
```

**Parameters:**

* `file_path` (str): The path to the output JSON file.

## `repository.write_symbol_usages()`

Writes the found usages of a specific symbol to a JSON file.

```python
repository.write_symbol_usages(symbol_name: str, file_path: str, symbol_type: Optional[str] = None) -> None
```

**Parameters:**

* `symbol_name` (str): The name of the symbol whose usages were found.
* `file_path` (str): The path to the output JSON file.
* `symbol_type` (Optional\[str]): The symbol type filter used when finding usages. Defaults to `None`.

## `repository.get_context_assembler()`

Convenience helper that returns a fresh `ContextAssembler` bound to this repository. Use it instead of importing the class directly:

```python
assembler = repository.get_context_assembler()
assembler.add_diff(my_diff)
context_blob = assembler.format_context()
```

**Returns:**

* `ContextAssembler`: Ready-to-use assembler instance.

## `repository.get_summarizer()`

### Automatic cleanup of the tmp repo cache

When you work with remote URLs Kit tries to be fast by keeping a shallow clone on disk (default location: `/tmp/kit-repo-cache`). Over time those can add up, especially in containerised environments. With `cache_ttl_hours` (or the `KIT_TMP_REPO_TTL_HOURS` env-var) you can make Kit self-purge old clones:

```python
# Purge anything older than 2 hours, then clone
repo = Repository(
    "https://github.com/owner/repo",
    cache_ttl_hours=2,
)
```

**When does the purge run?**

* The very first time `_clone_github_repo` is executed in a Python process (i.e., when the first remote repo is opened).
* A light-weight helper walks the top-level folders inside the cache directory and removes those whose directory `mtime` is older than the TTL.
* The helper is wrapped in `functools.lru_cache(maxsize=1)`, so subsequent clones in the same process don’t repeat the walk.

If you never set a TTL Kit keeps the previous behavior (clones live until the OS or you remove them).

# Kit REST API

`kit` ships a lightweight FastAPI server that exposes most of the same capabilities as the Python API and the MCP server, but over good old HTTP. This page lists every route, its query-parameters and example `curl` invocations.

The server lives in `kit.api.app`. Run it directly with:

```bash
uvicorn kit.api.app:app --reload
```

***

## 1 Opening a repository

```http
POST /repository
```

Body (JSON):

| field         | type   | required | description                    |
| ------------- | ------ | -------- | ------------------------------ |
| path\_or\_url | string | yes      | Local path **or** Git URL      |
| ref           | string | no       | Commit SHA / branch / tag      |
| github\_token | string | no       | OAuth token for private clones |

Return → `{ "id": "8b1d4f29c7b1" }`

The ID is deterministic: `sha1(<canonical-path>@<ref>)[:12]`. Re-POSTing the same path+ref combination always returns the same ID – so clients can cache it.

**Examples:**

```bash
# Open repository at current state
curl -X POST localhost:8000/repository \
  -d '{"path_or_url": "/my/project"}' \
  -H 'Content-Type: application/json'


# Open repository at specific tag
curl -X POST localhost:8000/repository \
  -d '{"path_or_url": "https://github.com/owner/repo", "ref": "v1.2.3"}' \
  -H 'Content-Type: application/json'


# Open repository at specific commit
curl -X POST localhost:8000/repository \
  -d '{"path_or_url": "https://github.com/owner/repo", "ref": "abc123def456"}' \
  -H 'Content-Type: application/json'


# Open private repository with authentication
curl -X POST localhost:8000/repository \
  -d '{"path_or_url": "https://github.com/owner/private-repo", "github_token": "ghp_xxxx"}' \
  -H 'Content-Type: application/json'
```

> **Note:** If `path_or_url` is a **GitHub URL**, the server shells out to `git clone`. Pass `github_token` to authenticate when cloning **private** repositories. The `ref` parameter allows you to analyze specific versions - useful for historical analysis, release comparisons, or ensuring reproducible results.

> **Authentication Note:** Unlike the Python API and MCP server, the REST API requires explicit `github_token` parameters and does not automatically check environment variables like `KIT_GITHUB_TOKEN`. This is by design to keep the HTTP API stateless and explicit.

## 2 Navigation

| Method & path                       | Purpose                 |
| ----------------------------------- | ----------------------- |
| `GET /repository/{id}/file-tree`    | JSON list of files/dirs |
| `GET /repository/{id}/files/{path}` | Raw text response       |
| `DELETE /repository/{id}`           | Evict from registry/LRU |

Example:

```bash
curl "$KIT_URL/repository/$ID/files/models/user.py"
```

## 3 Search & Grep

### Text Search (Regex)

```http
GET /repository/{id}/search?q=<regex>&pattern=*.py
```

Returns regex-based search hits with file & line numbers.

### Literal Grep Search

```http
GET /repository/{id}/grep?pattern=<literal>&case_sensitive=true&include_pattern=*.py&exclude_pattern=*test*&max_results=1000&directory=src&include_hidden=false
```

Fast literal string search using system grep with smart directory exclusions.

**Query Parameters:**

* `pattern` (required): Literal string to search for
* `case_sensitive` (default: true): Case sensitive search
* `include_pattern`: Include files matching glob pattern (e.g., ‘\*.py’)
* `exclude_pattern`: Exclude files matching glob pattern
* `max_results` (default: 1000): Maximum number of results
* `directory`: Limit search to specific directory within repository
* `include_hidden` (default: false): Include hidden directories in search

**Default Exclusions:** Automatically excludes `node_modules`, `__pycache__`, `.git`, `dist`, `build`, `.venv`, and other common directories for better performance.

**Examples:**

```bash
# Basic literal search
curl "localhost:8000/repository/$ID/grep?pattern=TODO"


# Case insensitive search in Python files only
curl "localhost:8000/repository/$ID/grep?pattern=function&case_sensitive=false&include_pattern=*.py"


# Search specific directory with custom limits
curl "localhost:8000/repository/$ID/grep?pattern=class&directory=src/api&max_results=50"


# Include hidden directories (search .github, .vscode, etc.)
curl "localhost:8000/repository/$ID/grep?pattern=workflow&include_hidden=true"
```

## 4 Symbols & usages

```http
GET /repository/{id}/symbols?file_path=...&symbol_type=function
GET /repository/{id}/usages?symbol_name=foo&symbol_type=function
```

`/symbols` without `file_path` scans the whole repo (cached).

## 5 Composite index

```http
GET /repository/{id}/index
```

Response:

```json
{
  "files": [ ... file-tree items ... ],
  "symbols": { "path/to/file.py": [ {"name": "foo", ...} ] }
}
```

## 6 Advanced Capabilities

These endpoints are included in the standard `kit` installation but may have specific runtime requirements:

| Route           | Key Runtime Requirement(s)                          | Notes                                                                                                |
| --------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `/summary`      | LLM API key (e.g., `OPENAI_API_KEY` in environment) | Generates code summaries. Returns `400` if key is missing/invalid, `503` if LLM service fails.       |
| `/dependencies` | None for fetching graph data (Python/Terraform)     | Returns dependency graph. `graphviz` needed only for local visualization helpers, not this endpoint. |

## 7 Git Metadata

```http
GET /repository/{id}/git-info
```

Returns git repository metadata including current SHA, branch, and remote URL.

**Response:**

```json
{
  "current_sha": "8cf426abe80f6cd3ab02ffc6fb11b00dd60995c8",
  "current_sha_short": "8cf426a",
  "current_branch": "main",
  "remote_url": "git@github.com:cased/kit.git"
}
```

Fields will be `null` if the repository is not a git repository or the information is not available.

### Upcoming Features

The following features are currently in development and will be added in future releases:

| Planned Feature          | Description                                                                            | Status      |
| ------------------------ | -------------------------------------------------------------------------------------- | ----------- |
| `/semantic-search`       | Embedding-based search using vector databases to find semantically similar code chunks | Coming soon |
| Enhanced symbol analysis | Improved cross-language symbol detection and relationship mapping                      | Planned     |

## 8 Common HTTP Status Codes

* `200 OK`: Request succeeded.
* `201 Created`: Repository opened successfully.
* `204 No Content`: Repository deleted successfully.
* `400 Bad Request`: Invalid parameters in the request (e.g., unsupported language for dependencies, missing API key for summaries).
* `404 Not Found`: Requested resource (repository, file, symbol) could not be found.
* `500 Internal Server Error`: An unexpected error occurred on the server.
* `503 Service Unavailable`: An external service required by the handler (e.g., an LLM API) failed or is unavailable.

***

### Example session

```bash
# 1 Open local repo (deterministic id)
ID=$(curl -sX POST localhost:8000/repository \
     -d '{"path_or_url": "/my/project"}' \
     -H 'Content-Type: application/json' | jq -r .id)


# 1b Open remote repo at specific version
VERSION_ID=$(curl -sX POST localhost:8000/repository \
     -d '{"path_or_url": "https://github.com/owner/repo", "ref": "v1.2.3"}' \
     -H 'Content-Type: application/json' | jq -r .id)


# 2 Check git metadata
curl "localhost:8000/repository/$ID/git-info"


# 3 Find every file that mentions "KeyError" (regex search)
curl "localhost:8000/repository/$ID/search?q=KeyError"


# 3b Fast literal search for TODO comments in Python files
curl "localhost:8000/repository/$ID/grep?pattern=TODO&include_pattern=*.py&case_sensitive=false"


# 4 Show snippet
curl "localhost:8000/repository/$ID/files/auth/session.py" | sed -n '80,95p'
```

# Summarizer API

This page details the API for the `Summarizer` class, used for interacting with LLMs for code summarization tasks.

## Initialization

Details on how to initialize the `Summarizer` (likely via `repo.get_summarizer()`).

Note

Typically, you obtain a `Summarizer` instance via `repo.get_summarizer()` rather than initializing it directly.

## Methods

### `summarize_file(file_path: str) -> str`

Summarizes the content of the specified file.

* **Parameters:**
  * `file_path` (str): The path to the file within the repository.

* **Returns:**
  * `str`: The summary generated by the LLM.

* **Raises:**

  * `FileNotFoundError`: If the `file_path` does not exist in the repo.
  * `LLMError`: If there’s an issue communicating with the LLM.

### `summarize_function(file_path: str, function_name: str) -> str`

Summarizes a specific function within the specified file.

* **Parameters:**

  * `file_path` (str): The path to the file containing the function.
  * `function_name` (str): The name of the function to summarize.

* **Returns:**
  * `str`: The summary generated by the LLM.

* **Raises:**

  * `FileNotFoundError`: If the `file_path` does not exist in the repo.
  * `SymbolNotFoundError`: If the function cannot be found in the file.
  * `LLMError`: If there’s an issue communicating with the LLM.

### `summarize_class(file_path: str, class_name: str) -> str`

Summarizes a specific class within the specified file.

* **Parameters:**

  * `file_path` (str): The path to the file containing the class.
  * `class_name` (str): The name of the class to summarize.

* **Returns:**
  * `str`: The summary generated by the LLM.

* **Raises:**

  * `FileNotFoundError`: If the `file_path` does not exist in the repo.
  * `SymbolNotFoundError`: If the class cannot be found in the file.
  * `LLMError`: If there’s an issue communicating with the LLM.

## Configuration

Details on the configuration options (`OpenAIConfig`, etc.). This is typically handled when calling `repo.get_summarizer(config=...)` or via environment variables read by the default `OpenAIConfig`.

The `Summarizer` currently uses `OpenAIConfig` for its LLM settings. When a `Summarizer` is initialized without a specific config object, it creates a default `OpenAIConfig` with the following parameters:

* `api_key` (str, optional): Your OpenAI API key. Defaults to the `OPENAI_API_KEY` environment variable. If not found, an error will be raised.
* `model` (str): The OpenAI model to use. Defaults to `"gpt-4o"`.
* `temperature` (float): Sampling temperature for the LLM. Defaults to `0.7`.
* `max_tokens` (int): The maximum number of tokens to generate in the summary. Defaults to `1000`.

You can customize this by creating an `OpenAIConfig` instance and passing it to `repo.get_summarizer()`:

```python
from kit.summaries import OpenAIConfig


# Example: Customize model and temperature
my_config = OpenAIConfig(model="o3-mini", temperature=0.2)
summarizer = repo.get_summarizer(config=my_config)


# Now summarizer will use o3-mini with temperature 0.2
summary = summarizer.summarize_file("path/to/your/file.py")
```

# SummarySearcher API

> API documentation for the SummarySearcher class.

The `SummarySearcher` class provides a simple way to query an index built by [`DocstringIndexer`](/api/docstring-indexer). It takes a search query, embeds it using the same embedding function used for indexing, and retrieves the most semantically similar summaries from the vector database.

## Constructor

**Class: `SummarySearcher`** *(defined in `kit/docstring_indexer.py`)*

The `SummarySearcher` is typically initialized with an instance of `DocstringIndexer`. It uses the `DocstringIndexer`’s configured backend and embedding function to perform searches.

```python
from kit.docstring_indexer import DocstringIndexer, SummarySearcher


# Assuming 'indexer' is an already initialized DocstringIndexer instance
# indexer = DocstringIndexer(repo=my_repo, summarizer=my_summarizer)
# indexer.build() # Ensure the index is built


searcher = SummarySearcher(indexer=indexer)
```

**Parameters:**

* **`indexer`** (`DocstringIndexer`, required): An instance of `DocstringIndexer` that has been configured and preferably has had its `build()` method called. The `SummarySearcher` will use this indexer’s `backend` and `embed_fn`. See the [`DocstringIndexer API docs`](./docstring-indexer) for more details on the indexer.

## Methods

### `search`

**Method: `SummarySearcher.search`** *(defined in `kit/docstring_indexer.py`)*

Embeds the given `query` string and searches the vector database (via the indexer’s backend) for the `top_k` most similar document summaries.

```python
query_text = "How is user authentication handled?"
results = searcher.search(query=query_text, top_k=3)


for result in results:
    print(f"Found in: {result.get('file_path')} ({result.get('symbol_name')})")
    print(f"Score: {result.get('score')}")
    print(f"Summary: {result.get('summary')}")
    print("----")}
```

**Parameters:**

* **`query`** (`str`, required): The natural language query string to search for.
* **`top_k`** (`int`, default: `5`): The maximum number of search results to return.

**Returns:** `List[Dict[str, Any]]`

A list of dictionaries, where each dictionary represents a search hit. Each hit typically includes metadata, a score, an ID, and the summary text.

# Changelog

> Track changes and improvements in Kit releases

# Changelog

All notable changes to Kit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## \[1.4.0] - 2025-06-27

### New Features

* **Dependency Analysis CLI Command**: Complete command-line interface for code dependency analysis

  * New `kit dependencies` command supporting Python and Terraform projects
  * Dependency graph generation with multiple output formats (JSON, DOT, GraphML, adjacency)
  * Circular dependency detection with detailed cycle visualization
  * Module-specific analysis with direct and indirect dependency tracking
  * LLM-friendly context generation for AI-powered dependency analysis
  * Graphviz integration for dependency visualization (PNG, SVG, PDF output)

## \[1.3.3] - 2025-06-26

### Security / Bug Fixes

* **Private-repo cloning with installation tokens**: `Repository` now uses an improved `GIT_ASKPASS` helper that supplies `x-access-token` as the username and the token as the password.

  * Fixes 404 “repository not found” errors when using GitHub App installation tokens.
  * Backwards-compatible: personal-access tokens (PATs) continue to work unchanged.

## \[1.3.2] - 2025-06-25

### Improvements

* **More actionable API errors**: REST endpoints now return precise HTTP status codes and JSON bodies.

  * `/repository` returns `404 {"detail": "Repository not found: <url>"}` instead of a generic 500.
  * Dependency and search endpoints propagate upstream errors for easier debugging.

* **Verbose logging**: Repository creation and search endpoints log request URLs, headers, and truncated bodies for better traceability.

## \[1.3.1] - 2025-06-24

### Improvements

* **Better grep**

  * Sensible ignore directories
  * More grep options

## \[1.3.0]

### New Features

* **Grep Command**: Fast literal text search across repository files

  * New `kit grep` command with system grep integration
  * Case-sensitive and case-insensitive search options
  * File inclusion/exclusion patterns with glob support
  * Configurable result limits and automatic .git directory exclusion
  * 30-second timeout protection for large repositories

* **File Tree Subpath Support**: Analyze specific directories within repositories

  * New `--path` option for `kit file-tree` command enables directory-specific analysis
  * Supports relative paths from repository root (e.g., `--path src/components`)
  * Compatible with existing output formats and export functionality
  * Enables focused analysis of large repository subdirectories

### Improvements

* **Path Validation**: Centralized path checking across all file operations

  * Unified validation utility prevents path traversal in user-provided file paths
  * Maintains compatibility with existing symlink handling on macOS

## \[1.2.3]

### Bug Fixes

* **Duplicate TypeScript / TSX Symbols**: Resolved an issue where some symbols could be recorded twice in `symbols.json` depending on tree-sitter queries

  * The `TreeSitterSymbolExtractor` now removes duplicate symbols based on `(name, type, start_line, end_line)`
  * Added dedicated regression test suite covering TypeScript and TSX symbol deduplication

### Improvements

* **Optional Dependencies**: Made `tiktoken` dependency optional in `summaries.py` to reduce installation requirements
* **Dependency Management**: Relaxed FastAPI version constraints for better compatibility
* **Development Dependencies**: Moved linters and type checking dependencies to development group

## \[1.2.2]

### Enhanced Features

* **C++ Language Support**: Added comprehensive C++ symbol extraction with tree-sitter

  * Support for classes, functions, structs, enums, namespaces, and templates
  * Dedicated tree-sitter query patterns for C++ language constructs
  * Full integration with existing symbol extraction pipeline

* **Kotlin Language Support**: Initial Kotlin language integration with tree-sitter

  * Classes, functions, objects, interfaces, and enum support
  * Kotlin-specific language patterns and symbol detection
  * Comprehensive test coverage for Kotlin symbol extraction

## \[1.2.0]

### Enhanced Repository Cache Management

* **Automatic Cache Cleanup**: Repository now accepts optional `cache_ttl_hours` argument for automatic cleanup

  * Environment variable support via `KIT_TMP_REPO_TTL_HOURS`
  * Automatically deletes cached repository clones older than TTL
  * One-shot purge per Python process using `functools.lru_cache`
  * Fallback to previous behavior when TTL is unset

### Improvements

* **Enhanced Logging**: Debug logging for invalid TTL values and cleanup warnings
* **Cache Management**: Improved error handling for cache cleanup operations
* **Documentation**: Updated API documentation and README with cache management details

### Tests

* Added comprehensive test coverage in `tests/test_cache_cleanup.py`
* TTL parsing validation and error handling tests
* Cache cleanup logic verification

## \[1.1.0]

### Major Features

* **Incremental Analysis System**: High-performance caching for symbol extraction

  * 25x performance improvements for warm cache scenarios
  * Multi-strategy cache invalidation (mtime, size, content hash, git state detection)
  * Automatic cache invalidation on git operations (checkout, commit, merge, rebase)
  * LRU eviction with configurable cache size limits
  * New CLI commands: `kit cache status`, `kit cache clear`, `kit cache cleanup`, `kit cache stats`

### Enhanced Features

* **Dart Language Support**: Initial anguage integration with tree-sitter

  * Classes, functions, constructors, getters, setters, enums, extensions, mixins
  * Flutter widget detection patterns for StatelessWidget and StatefulWidget
  * Comprehensive symbol extraction with inheritance relationships
  * Plugin system support for custom Dart patterns

### Bug Fixes

* **Cache Invalidation Fix**: Resolved stale data issues causing incorrect line numbers

  * Repository objects now properly detect git state changes (branch switches, commits, merges)
  * Automatic cache invalidation prevents serving outdated symbol data from previous git states
  * Comprehensive git SHA tracking ensures data consistency across git operations

* **TSX Language Support**: Fixed missing TypeScript JSX support

  * Proper fallback to TypeScript queries when tsx directory doesn’t exist
  * Ensures consistent symbol extraction for React TypeScript projects

## \[1.0.3]

### New Features

* **Batch File Retrieval**: `Repository.get_file_content` now accepts a list of paths and returns a mapping of `path → content`, eliminating the need for multiple calls when bulk loading files.
  * Backwards-compatible: single-path calls still return a plain string.

## \[1.0.2]

### Improvements

* **Automatic GitHub Token Pickup**: Repository class and MCP server now automatically detect GitHub tokens from environment variables

  * Checks `KIT_GITHUB_TOKEN` first, then falls back to `GITHUB_TOKEN`
  * Simplifies private repository access - no need to pass tokens explicitly
  * Consistent behavior across Python API and MCP server interfaces
  * Explicit token parameters still override environment variables

* **macOS Symlink Compatibility**: Fixed path validation issues on macOS for MCP server

  * Resolves `/tmp` → `/private/tmp` symlink conflicts that caused path validation errors
  * Maintains security while ensuring compatibility with macOS filesystem structure
  * Fixes “not in subpath” errors when using Claude Desktop and other MCP clients

## \[1.0.0]

### Major Release

Kit 1.0.0 represents the first stable release of the code intelligence toolkit, marking a significant milestone in providing production-ready code analysis capabilities.

### Major Features

* **Production-Ready Core**: Stable API for code intelligence operations

  * Comprehensive symbol extraction across multiple programming languages
  * Advanced code search with regex and semantic capabilities
  * Repository analysis with git metadata integration
  * Cross-platform compatibility (Windows, macOS, Linux)

* **Multi-Access Architecture**: Four distinct ways to interact with Kit

  * **Python API**: Direct integration for applications and scripts
  * **Command Line Interface**: 15+ commands for shell scripting and automation
  * **REST API**: HTTP endpoints for web applications and microservices
  * **MCP Server**: Model Context Protocol integration for AI agents and development tools

* **AI-Powered PR Intelligence**: Complete PR review and analysis system

  * Smart PR reviews with configurable depth (quick, standard, thorough)
  * Cost-effective PR summaries for rapid triage
  * Intelligent commit message generation from staged changes
  * Support for multiple LLM providers (OpenAI, Anthropic, Google, Ollama)
  * Repository-aware analysis with symbol and dependency context

* **Advanced Code Understanding**: Deep codebase intelligence capabilities

  * Multi-language symbol extraction (Python, JavaScript, TypeScript, Go, etc.)
  * Dependency analysis for Python and Terraform projects
  * Semantic search with vector embeddings
  * Context-aware code chunking for LLM consumption

### Enhanced Features

* **Repository Versioning**: Analyze code at specific commits, tags, or branches
* **Caching System**: Intelligent repository caching for improved performance
* **Security**: Path traversal protection and input validation
* **Cost Tracking**: Real-time LLM usage monitoring and pricing transparency
* **Custom Profiles**: Organization-specific coding standards and review guidelines

### Architecture Highlights

* **Extensible Design**: Plugin architecture for adding new languages and analyzers
* **Memory Efficient**: Streaming operations for large codebases
* **Git Integration**: Native git operations with branch and commit support
* **Type Safety**: Comprehensive type annotations and validation

### Supported Languages

* **Primary**: Python, JavaScript, TypeScript, Go
* **Additional**: Java, C/C++, Rust, Ruby, PHP, and more via tree-sitter
* **Configuration**: JSON, YAML, TOML, Dockerfile
* **Infrastructure**: Terraform, Kubernetes YAML

### LLM Provider Support

* **Cloud Providers**: OpenAI GPT models, Anthropic Claude, Google Gemini
* **Alternative Providers**: OpenRouter, Together AI, Groq, Fireworks
* **Local Providers**: Ollama with free local models (DeepSeek R1, Qwen2.5-coder, CodeLlama)
* **Cost Optimization**: Hybrid workflows combining free local and premium cloud analysis

### Developer Experience

* **Comprehensive Documentation**: Full API reference and usage guides
* **Example Applications**: Real-world usage patterns and integrations
* **Community Support**: Discord server and GitHub discussions
* **Testing**: Extensive test suite with CI/CD integration

## \[0.7.0]

### Major Features

* **Custom Context Profiles**: Store and apply organization-specific coding standards and guidelines

  * Create reusable profiles: `kit review-profile create --name company-standards`
  * Apply to any PR: `kit review --profile company-standards <pr-url>`
  * Export/import for team collaboration

* **Priority Filtering**: Focus reviews on what matters most

  * Filter by priority levels: `kit review --priority=high,medium <pr-url>`
  * Reduce noise and costs by focusing on critical issues
  * Combine with other modes for targeted workflows

## \[0.6.4]

### Major Features

* **OpenRouter & LiteLLM Provider Support**: Complete integration with OpenAI-compatible providers

  * Access to 100+ models through OpenRouter at competitive prices
  * Support for Together AI, Groq, Perplexity, Replicate, and other popular providers
  * Additional cost tracking with accurate model name handling
  * Thanks to @AlanVerbner for this contribution

* **Google Gemini Support for PR Reviews**: Complete integration of Google’s Gemini models

  * Support for latest models: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-flash-8b`
  * Ultra-budget option with Gemini 1.5 Flash 8B at \~$0.003 per large PR
  * Automatic provider detection and accurate token-based cost tracking for Gemini

### Bug Fixes

* **Better fork PR Support**: Fixed issue preventing reviews of PRs from certain outside forks

  * Now uses base repository coordinates instead of head repository for diff fetching
  * Resolves 404 errors when reviewing external contributor PRs
  * Thanks to @redvelvets for this contribution

### Enhanced Features

* **Expanded Provider Ecosystem**: Three major categories now supported

  * **Cloud Providers**: Anthropic Claude, OpenAI GPT, Google Gemini
  * **Alternative Providers**: OpenRouter, Together AI, Groq, Fireworks
  * **Local Providers**: Ollama with free local models

* **Smart Model Detection**: Intelligent routing based on model names

  * Handles complex model naming like `openrouter/anthropic/claude-3.5-sonnet`
  * Automatically strips provider prefixes for accurate cost calculation
  * Maintains compatibility with all existing configurations

## \[0.6.3]

### Bug Fixes

* **Symbol Type Extraction Fix**: Fixed bug where some symbol types were incorrectly processed

  * Classes and other symbol types no longer have characters incorrectly stripped
  * Added comprehensive test coverage for symbol type processing edge cases

## \[0.6.2]

### Major Features

* **Ollama Support**: Complete local LLM inference support with Ollama

  * Zero-cost PR reviews with local models
  * Support for popular models like DeepSeek R1, Qwen2.5-coder, CodeLlama
  * Automatic provider detection from model names (e.g., `deepseek-r1:latest`)
  * First-class integration with kit’s repository intelligence

* **DeepSeek R1 Reasoning Model Support**

  * **Thinking Token Stripping**: Automatically removes `<think>...</think>` tags from reasoning models
  * Clean, professional output without internal reasoning clutter
  * Preserves the analytical capabilities while improving output quality
  * Works in both summarization and PR review workflows

* **Plain Output Mode**: New `--plain` / `-p` flag for pipe-friendly output

  * Removes all formatting and status messages
  * Perfect for piping to Claude Code or other AI tools
  * Enables powerful multi-stage AI workflows (e.g., `kit review -p | claude`)
  * Quiet mode suppresses all progress/status output

### Enhanced Features

* **CLI Improvements**

  * Added `--version` flag to display current kit version
  * Model override support: `--model` / `-m` flag for per-review model selection
  * Better error messages and help text

* **Documentation**

  * Comprehensive Ollama integration guides
  * Claude Code workflow examples
  * Multi-stage AI analysis patterns
  * Updated CLI reference with new flags

### Developer Experience

* **Community**

  * Added Discord community server for support and discussions
  * Improved README with better getting started instructions

* **Testing**

  * Comprehensive test suite for thinking token stripping
  * Ollama integration tests with mock scenarios
  * PR reviewer test coverage for new features

### Cost Optimization

* **Free Local Analysis**: Use Ollama for zero-cost code analysis
* **Hybrid Workflows**: Combine free local analysis with premium cloud implementation
* **Provider Switching**: Automatic provider detection and switching

## \[0.6.1]

### Improvements

* Enhanced line number accuracy in PR reviews
* Improved debug output for troubleshooting
* Better test coverage for core functionality
* Performance optimizations for large repositories

### Bug Fixes

* Fixed edge cases in symbol extraction
* Improved error handling for malformed diffs
* Better validation for GitHub URLs

## \[0.6.0]

### Major Features

* Advanced PR reviews
* Enhanced line number context and accuracy fore reviews
* Comprehensive cost tracking and pricing updates for reviews
* Improved repository intelligence with better symbol analysis

### Enhanced Features

* Better diff parsing and analysis
* Enhanced file prioritization algorithms for reviews
* Improved cost breakdown reporting

## Links

* [GitHub Releases](https://github.com/cased/kit/releases)
* [Issues](https://github.com/cased/kit/issues)

# Code Summarization

In addition to the non-LLM based functions of the `Repository` class, `kit` also integrates directly with LLMs via the `Summarizer` class to provide intelligent code summarization capabilities. This helps you quickly understand the purpose and functionality of entire code files, specific functions, or classes.

## Getting Started

To use code summarization, you’ll need an LLM provider configured. Currently, OpenAI, Anthropic, and Google Cloud’s Generative AI models are supported.

1. **Install Dependencies:**

   ```bash
   # Ensure you are in your project's virtual environment
   uv pip install cased-kit
   ```

   The installation includes all dependencies for OpenAI, Anthropic, and Google Cloud’s Generative AI models.

2. **Set API Key(s):** Configure the API key(s) for your chosen provider(s) as environment variables:

   ```bash
   # For OpenAI
   export OPENAI_API_KEY="sk-..."


   # For Anthropic
   export ANTHROPIC_API_KEY="sk-ant-..."


   # For Google
   export GOOGLE_API_KEY="AIzaSy..."
   ```

   You only need to set the key for the provider(s) you intend to use. If no specific configuration is provided to the `Summarizer` (see ‘Configuration (Advanced)’ below), `kit` defaults to using OpenAI via `OpenAIConfig()`, which expects `OPENAI_API_KEY`.

   For `OpenAIConfig`, you can also customize parameters such as the `model`, `temperature`, or `base_url` (e.g., for connecting to services like OpenRouter). See the ‘Configuration (Advanced)’ section for detailed examples.

## Basic Usage: Summarizing Files

The primary way to access summarization is through the `Repository` object’s `get_summarizer()` factory method.

```python
import kit
import os


# Ensure API key is set (replace with your actual key handling)
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"


try:
    # Load the repository
    repo = kit.Repository("/path/to/your/project")


    # Get the summarizer instance (defaults to OpenAIConfig using env var OPENAI_API_KEY)
    # See 'Configuration (Advanced)' for using Anthropic or Google.
    summarizer = repo.get_summarizer()


    # Summarize a specific file
    file_path = "src/main_logic.py"
    summary = summarizer.summarize_file(file_path)


    print(f"Summary for {file_path}:\n{summary}")


except Exception as e:
    print(f"An error occurred with the LLM provider: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
```

Note

When you call `repo.get_summarizer()` (or instantiate `Summarizer` directly), it will use the appropriate LLM library based on the specified configuration (e.g., `openai`, `anthropic`, `google.genai`). All necessary libraries are included in the standard installation.

### How it Works

When you call `summarize_file`, `kit` performs the following steps:

1. Retrieves the content of the specified file using `repo.get_file_content()`.
2. Constructs a prompt containing the file content, tailored for code summarization.
3. Sends the prompt to the configured LLM provider and model (e.g., OpenAI’s GPT-4o).
4. Parses the response and returns the summary text.

### Configuration (Advanced)

While environment variables are the default, you can provide specific configuration:

```python
from kit.summaries import OpenAIConfig, AnthropicConfig, GoogleConfig


# Load repo
repo = kit.Repository("/path/to/your/project")


# Define custom OpenAI configuration
openai_custom_config = OpenAIConfig(
    api_key="sk-...", # Can be omitted if OPENAI_API_KEY is set
    model="gpt-4o-mini"
)
# Get summarizer with specific OpenAI config
openai_summarizer = repo.get_summarizer(config=openai_custom_config)
# Summarize using the custom OpenAI configuration
openai_summary = openai_summarizer.summarize_file("src/utils_openai.py")
print(f"OpenAI Summary:\n{openai_summary}")
```

#### Using OpenAI-Compatible Endpoints (e.g., OpenRouter)

The `OpenAIConfig` also supports a `base_url` parameter, allowing you to use any OpenAI-compatible API endpoint, such as [OpenRouter](https://openrouter.ai/). This is useful for accessing a wide variety of models through a single API key and endpoint.

To use OpenRouter:

1. Your `api_key` in `OpenAIConfig` should be your OpenRouter API key.
2. Set the `base_url` to OpenRouter’s API endpoint (e.g., `https://openrouter.ai/api/v1`).
3. The `model` parameter should be the specific model string as recognized by OpenRouter (e.g., `meta-llama/llama-3.3-70b-instruct`).

```plaintext
# Example for OpenRouter
openrouter_config = OpenAIConfig(
    api_key="YOUR_OPENROUTER_API_KEY", # Replace with your OpenRouter key
    model="meta-llama/llama-3.3-70b-instruct", # Example model on OpenRouter
    base_url="https://openrouter.ai/api/v1"
)


openrouter_summarizer = repo.get_summarizer(config=openrouter_config)
```

#### Additional Configs

# Define custom Anthropic configuration

anthropic\_config = AnthropicConfig( api\_key=“sk-ant-…”, # Can be omitted if ANTHROPIC\_API\_KEY is set model=“claude-3-haiku-20240307” )

# Define custom Google configuration

google\_config = GoogleConfig( api\_key=“AIzaSy…”, # Can be omitted if GOOGLE\_API\_KEY is set model=“gemini-2.5-flash” # Or “gemini-2.5-pro” for complex reasoning, “gemini-2.0-flash-lite” for speed )

````plaintext
## Advanced Usage


### Summarizing Functions and Classes


Beyond entire files, you can target specific functions or classes:


```python
import kit


repo = kit.Repository("/path/to/your/project")
summarizer = repo.get_summarizer() # Assumes OPENAI_API_KEY is set


# Summarize a specific function
function_summary = summarizer.summarize_function(
    file_path="src/core/processing.py",
    function_name="process_main_data"
)
print(f"Function Summary:\n{function_summary}")


# Summarize a specific class
class_summary = summarizer.summarize_class(
    file_path="src/models/user.py",
    class_name="UserProfile"
)
print(f"Class Summary:\n{class_summary}")
````

Note

Under the hood, `summarize_function` and `summarize_class` will use `kit`’s symbol extraction capabilities (`repo.extract_symbols`) to locate the precise code snippet for the target function or class before sending it to the LLM, providing more focused summaries.

### Combining with Other Repository Features

You can combine the `Summarizer` with other `Repository` methods for powerful workflows. For example, find all classes in a file and then summarize each one:

```python
import kit


repo = kit.Repository("/path/to/your/project")
summarizer = repo.get_summarizer()


file_to_analyze = "src/services/notification_service.py"


# 1. Find all symbols in the file
symbols = repo.extract_symbols(file_path=file_to_analyze)


# 2. Filter for classes
class_symbols = [s for s in symbols if s.get('type') == 'class']


# 3. Summarize each class
for sym in class_symbols:
    class_name = sym.get('name')
    if class_name:
        print(f"--- Summarizing Class: {class_name} ---")
        try:
            summary = summarizer.summarize_class(
                file_path=file_to_analyze,
                class_name=class_name
            )
            print(summary)
        except Exception as e:
            print(f"Could not summarize {class_name}: {e}")
    print("\n")
```

Note

While `repo.get_summarizer()` is the most convenient way to get a configured `Summarizer`, you can also instantiate it directly if needed:

```python
from kit import Repository
from kit.summaries import Summarizer, OpenAIConfig, AnthropicConfig, GoogleConfig


my_repo = Repository("/path/to/code")


# Example with AnthropicConfig
# Similar approach for OpenAIConfig or GoogleConfig
my_anthropic_config = AnthropicConfig(
    api_key="sk-ant-your-key",
    model="claude-3-sonnet-20240229"
)
direct_summarizer = Summarizer(repo=my_repo, config=my_anthropic_config)


# Or for OpenAI:
# my_openai_config = OpenAIConfig(api_key="sk-your-key", model="gpt-4o")
# direct_summarizer = Summarizer(repo=my_repo, config=my_openai_config)


# Or for Google:
# my_google_config = GoogleConfig(api_key="AIzaSy-your-key", model="gemini-pro")
# direct_summarizer = Summarizer(repo=my_repo, config=my_google_config)


summary = direct_summarizer.summarize_file("some/file.py")
print(summary)
```

# Assembling Context

When you send code to an LLM you usually **don’t** want the entire repository – just the *most relevant* bits. `ContextAssembler` helps you stitch those bits together into a single prompt-sized string.

## Why you need it

* **Token limits** – GPT-4o tops out at \~128k tokens; some models less.
* **Signal-to-noise** – Cut boilerplate, focus the model on what matters.
* **Automatic truncation** – Keeps prompts within your chosen character budget.

## Quick start

```python
from kit import Repository, ContextAssembler


repo = Repository("/path/to/project")


# Assume you already have chunks, e.g. from repo.search_semantic()
chunks = repo.search_text("jwt decode")


assembler = ContextAssembler(max_chars=12_000)
context = assembler.from_chunks(chunks)


print(context)  # → Ready to drop into your chat prompt
```

`chunks` can be any list of dicts that include a `code` key – the helper trims and orders them by length until the budget is filled.

### Fine-tuning

| Parameter           | Default         | Description                               |
| ------------------- | --------------- | ----------------------------------------- |
| `max_chars`         | `12000`         | Rough character cap for the final string. |
| `separator`         | `"\n\n---\n\n"` | Separator inserted between chunks.        |
| `header` / `footer` | `""`            | Optional strings prepended/appended.      |

```python
assembler = ContextAssembler(
    max_chars=8000,
    header="### Code context\n",
    footer="\n### End context",
)
```

## Combining with other tools

1. **Vector search → assemble → chat**

   ```python
   chunks = repo.search_semantic("retry backoff", embed_fn, top_k=10)
   prompt = assembler.from_chunks(chunks)
   response = my_llm.chat(prompt + "\n\nQ: …")
   ```

2. **Docstring search first** – Use `SummarySearcher` for high-level matches, then pull full code for those files via `repo.context`.

3. **Diff review bots** – Feed only the changed lines + surrounding context.

## API reference

```python
from kit.llm_context import ContextAssembler
```

### `__init__(repo, *, title=None)`

Constructs a new `ContextAssembler`.

* `repo`: A `kit.repository.Repository` instance.
* `title` (optional): A string to prepend to the assembled context.

### `from_chunks(chunks, max_chars=12000, separator="...", header="", footer="")`

This is the primary method for assembling context from a list of code chunks.

* `chunks`: A list of dictionaries, each with a `"code"` key.
* `max_chars`: Target maximum character length for the output string.
* `separator`: String to insert between chunks.
* `header` / `footer`: Optional strings to wrap the entire context.

Returns a single string with concatenated, truncated chunks.

### Other methods

While `from_chunks` is the most common entry point, `ContextAssembler` also offers methods to add specific types of context if you’re building a prompt manually:

* `add_diff(diff_text)`: Adds a Git diff.
* `add_file(file_path, highlight_changes=False)`: Adds the full content of a file.
* `add_symbol_dependencies(file_path, max_depth=1)`: Adds content of files that `file_path` depends on.
* `add_search_results(results, query)`: Formats and adds semantic search results.
* `format_context()`: Returns the accumulated context as a string.

# Dependency Analysis

The dependency analysis feature in `kit` allows you to map, visualize, and analyze the relationships between different components in your codebase. This helps you understand complex codebases, identify potential refactoring opportunities, detect circular dependencies, and prepare dependency context for large language models.

## Why Dependency Analysis?

Understanding dependencies in a codebase is crucial for:

* **Codebase Understanding:** Quickly grasp how different modules interact with each other.
* **Refactoring Planning:** Identify modules with excessive dependencies or cyclic relationships that might benefit from refactoring.
* **Technical Debt Assessment:** Map dependencies to visualize potential areas of technical debt or architectural concerns.
* **Impact Analysis:** Determine the potential impact of changes to specific components.
* **LLM Context Preparation:** Generate concise, structured descriptions of codebase architecture for LLM context.

## Getting Started

You can access the dependency analyzer through the `Repository` object:

```python
from kit import Repository


# Load your codebase
repo = Repository("/path/to/your/codebase")


# Get a language-specific dependency analyzer
# Currently supports 'python' and 'terraform'
analyzer = repo.get_dependency_analyzer('python')  # or 'terraform'


# Build the dependency graph
graph = analyzer.build_dependency_graph()


print(f"Found {len(graph)} components in the dependency graph")
```

## Exploring Dependencies

Once you’ve built the dependency graph, you can explore it in various ways:

```python
# Find cycles (circular dependencies)
cycles = analyzer.find_cycles()
if cycles:
    print(f"Found {len(cycles)} circular dependencies:")
    for cycle in cycles[:5]:
        print(f"  {' → '.join(cycle)} → {cycle[0]}")


# Get dependencies for a specific module
module_deps = analyzer.get_resource_dependencies('module_name')
print(f"Module depends on: {module_deps}")


# Get dependents (modules that depend on a specific module)
dependents = analyzer.get_dependents('module_name')
print(f"Modules that depend on this: {dependents}")
```

## Visualizing Dependencies

You can visualize the dependency graph using common formats like DOT, GraphML, or JSON:

```python
# Export to DOT format (for use with tools like Graphviz)
analyzer.export_dependency_graph(
    output_format="dot",
    output_path="dependency_graph.dot"
)


# Generate a PNG visualization
analyzer.visualize_dependencies(
    output_path="dependency_visualization.png",
    format="png"  # supports 'png', 'svg', or 'pdf'
)
```

Note

Visualization requires the Graphviz software to be installed on your system.

## LLM Context Generation

One of the most powerful features of the dependency analyzer is its ability to generate concise, LLM-friendly context about your codebase structure:

```python
# Generate markdown context for LLMs
context = analyzer.generate_llm_context(
    output_format="markdown",
    output_path="dependency_context.md",
    max_tokens=4000  # approximate token limit
)
```

The generated context includes:

* Overall statistics (component count, type breakdown)
* Key components with high connectivity
* Circular dependency information
* Language-specific insights (e.g., import patterns for Python, resource relationships for Terraform)

## Language-Specific Features

### Python Dependency Analysis

The Python dependency analyzer maps import relationships between modules:

```python
# Get a Python-specific analyzer
python_analyzer = repo.get_dependency_analyzer('python')


# Build the graph
python_analyzer.build_dependency_graph()


# Find standard library vs. third-party dependencies
report = python_analyzer.generate_dependency_report()
print(f"Standard library imports: {len(report['standard_library_imports'])}")
print(f"Third-party imports: {len(report['third_party_imports'])}")
```

### Terraform Dependency Analysis

The Terraform dependency analyzer maps relationships between infrastructure resources:

```python
# Get a Terraform-specific analyzer
terraform_analyzer = repo.get_dependency_analyzer('terraform')


# Build the graph
terraform_analyzer.build_dependency_graph()


# Find all resources of a specific type
s3_buckets = terraform_analyzer.get_resource_by_type("aws_s3_bucket")
```

Each resource in the graph includes its absolute file path, making it easy to locate resources in complex infrastructure codebases:

```plaintext
aws_launch_template.app (aws_launch_template) [File: /path/to/your/project/compute.tf]
```

Tip

For complete API details, including all available methods and options, see the **[DependencyAnalyzer API Reference](/api/dependency-analyzer/)**.

## Advanced Usage

### Custom Dependency Analysis

If you have specific needs for your dependency analysis, you can extend the base `DependencyAnalyzer` class to create analyzers for other languages or frameworks:

```python
from kit.dependency_analyzer import DependencyAnalyzer


class CustomDependencyAnalyzer(DependencyAnalyzer):
    # Implement required abstract methods
    def build_dependency_graph(self):
        # Your custom logic here
        pass


    def export_dependency_graph(self, output_format="json", output_path=None):
        # Your custom export logic here
        pass


    def find_cycles(self):
        # Your custom cycle detection logic here
        pass


    def visualize_dependencies(self, output_path, format="png"):
        # Your custom visualization logic here
        pass
```

# Docstring-based Vector Indexing

Alpha Feature

The features described on this page, particularly symbol-level indexing and LLM-generated summaries, are currently in **alpha**. API and behavior may change in future releases. Please use with this in mind and report any issues or feedback.



`DocstringIndexer` builds a vector index using **LLM-generated summaries** of source files (“docstrings”) instead of the raw code. This often yields more relevant results because the embedded text focuses on *intent* rather than syntax or specific variable names.

## Why use it?

* **Cleaner embeddings** – Comments like *“Summary of retry logic”* embed better than nested `for`-loops.
* **Smaller index** – One summary per file (or symbol) is < 1 kB, while the file itself might be thousands of tokens.
* **Provider-agnostic** – Works with any LLM supported by `kit.Summarizer` (OpenAI, Anthropic, Google…).

## How it Works

1. **Configuration**: Instantiate `DocstringIndexer` with a `Repository` object and a `Summarizer` (configured with your desired LLM, e.g., OpenAI, Anthropic, Google). An embedding function (`embed_fn`) can also be provided if you wish to use a custom embedding model; otherwise, `DocstringIndexer` will use a default embedding function (based on `sentence-transformers`, which is included in the standard installation).

 

```python
from kit import Repository, DocstringIndexer, Summarizer
from kit.llms.openai import OpenAIConfig # For configuring the summarization LLM


# 1. Initialize your Repository
repo = Repository("/path/to/your/codebase")


# 2. Configure and initialize the Summarizer
# It's good practice to specify the model you want for summarization.
# Summarizer defaults to OpenAIConfig() if no config is passed, which then
# might use environment variables (OPENAI_MODEL) or a default model from OpenAIConfig.
llm_summarizer_config = OpenAIConfig(model="gpt-4o") # Or "gpt-4-turbo", etc.
summarizer = Summarizer(repo, config=llm_summarizer_config)


# 3. Initialize DocstringIndexer
# By default, DocstringIndexer now uses SentenceTransformer('all-MiniLM-L6-v2')
# for embeddings, so you don't need to provide an embed_fn for basic usage.
indexer = DocstringIndexer(repo, summarizer)


# 4. Build the index
# This will process the repository, generate summaries, and create embeddings.
indexer.build()


# After building, you can query the index using a SummarySearcher.


# Option 1: Manually create a SummarySearcher (traditional way)
# from kit import SummarySearcher
# searcher_manual = SummarySearcher(indexer)


# Option 2: Use the convenient get_searcher() method (recommended)
searcher = indexer.get_searcher()


# Now you can use the searcher
results = searcher.search("your query here", top_k=3)
for result in results:
    print(f"Found: {result.get('metadata', {}).get('file_path')}::{result.get('metadata', {}).get('symbol_name')}")
    print(f"Summary: {result.get('metadata', {}).get('summary')}")
    print(f"Score: {result.get('score')}")
    print("---")
```

### Using a Custom Embedding Function (Optional)

If you want to use a different embedding model or a custom embedding function, you can pass it to the `DocstringIndexer` during initialization. The function should take a string as input and return a list of floats (the embedding vector).

For example, if you wanted to use a different model from the `sentence-transformers` library:

```python
from kit import Repository, DocstringIndexer, Summarizer
from kit.llms.openai import OpenAIConfig
from sentence_transformers import SentenceTransformer # Make sure you have this installed


repo = Repository("/path/to/your/codebase")
llm_summarizer_config = OpenAIConfig(model="gpt-4o")
summarizer = Summarizer(repo, config=llm_summarizer_config)


# Load a specific sentence-transformer model
# You can find available models at https://www.sbert.net/docs/pretrained_models.html
custom_st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def custom_embedding_function(text_to_embed: str):
    embedding_vector = custom_st_model.encode(text_to_embed)
    return embedding_vector.tolist()


# Initialize DocstringIndexer with your custom embedding function
indexer_custom = DocstringIndexer(repo, summarizer, embed_fn=custom_embedding_function)


indexer_custom.build()
```

This approach gives you flexibility if the default embedding model doesn’t meet your specific needs.

## Inspecting the Indexed Data (Optional)

After building the index, you might want to inspect its raw contents to understand what was stored. This can be useful for debugging or exploration. The exact method depends on the `VectorDBBackend` being used.

If you’re using the default `ChromaDBBackend` (or have explicitly configured it), you can access the underlying ChromaDB collection and retrieve entries.

```python
# Assuming 'indexer' is your DocstringIndexer instance after 'indexer.build()' has run.
# And 'indexer.backend' is an instance of ChromaDBBackend.


if hasattr(indexer.backend, 'collection'):
    chroma_collection = indexer.backend.collection
    print(f"Inspecting ChromaDB collection: {chroma_collection.name}")
    print(f"Number of items: {chroma_collection.count()}")


    # Retrieve the first few items (e.g., 3)
    # We include 'metadatas' and 'documents' (which holds the summary text).
    # 'embeddings' are excluded for brevity.
    retrieved_data = chroma_collection.get(
        limit=3,
        include=['metadatas', 'documents']
    )


    if retrieved_data and retrieved_data.get('ids'):
        for i in range(len(retrieved_data['ids'])):
            item_id = retrieved_data['ids'][i]
            # The 'document' is the summary text that was embedded.
            summary_text = retrieved_data['documents'][i] if retrieved_data['documents'] else "N/A"
            # 'metadata' contains file_path, symbol_name, original summary, etc.
            metadata = retrieved_data['metadatas'][i] if retrieved_data['metadatas'] else {}


            print(f"\n--- Item {i+1} ---")
            print(f"  ID (in Chroma): {item_id}")
            print(f"  Stored Summary (Document): {summary_text}")
            print(f"  Metadata:")
            for key, value in metadata.items():
                print(f"    {key}: {value}")
    else:
        print("No items found in the collection or collection is empty.")
else:
    print("The configured backend does not seem to be ChromaDB or doesn't expose a 'collection' attribute for direct inspection this way.")
```

**Expected Output from Inspection:**

Running the inspection code above might produce output like this:

```text
Inspecting ChromaDB collection: kit_docstring_index
Number of items: 10 # Or however many items are in your test repo index


--- Item 1 ---
  ID (in Chroma): utils.py::greet
  Stored Summary (Document): The `greet` function in the `utils.py` file is designed to generate a friendly greeting message...
  Metadata:
    file_path: utils.py
    level: symbol
    summary: The `greet` function in the `utils.py` file is designed to generate a friendly greeting message...
    symbol_name: greet
    symbol_type: function


--- Item 2 ---
  ID (in Chroma): app.py::main
  Stored Summary (Document): The `main` function in `app.py` demonstrates a simple authentication workflow...
  Metadata:
    file_path: app.py
    level: symbol
    summary: The `main` function in `app.py` demonstrates a simple authentication workflow...
    symbol_name: main
    symbol_type: function


... (and so on)
```

This shows that each entry in the ChromaDB collection has:

* An `id` (often `file_path::symbol_name`).
* The `document` field, which is the text of the summary that was embedded.
* `metadata` containing details like `file_path`, `symbol_name`, `symbol_type`, `level`, and often a redundant copy of the `summary` itself.

Knowing the structure of this stored data can be very helpful when working with search results or debugging the indexing process.

### Symbol-Level Indexing

Alpha Feature: Symbol-Level Indexing

Symbol-level indexing is an advanced alpha feature. While powerful, it may require more resources and is undergoing active development. Feedback is highly appreciated.

For more granular search, you can instruct `DocstringIndexer` to create summaries for individual **functions and classes** within your files. This allows for highly specific semantic queries like “find the class that manages database connections” or “what function handles user authentication?”

To enable symbol-level indexing, pass `level="symbol"` to `build()`:

```python
# Build a symbol-level index
indexer.build(level="symbol", file_extensions=[".py"], force=True)
```

When `level="symbol"`:

* `DocstringIndexer` iterates through files, then extracts symbols (functions, classes) from each file using `repo.extract_symbols()`.

* It then calls `summarizer.summarize_function()` or `summarizer.summarize_class()` for each symbol.

* The resulting embeddings are stored with metadata including:

  * `file_path`: The path to the file containing the symbol.
  * `symbol_name`: The name of the function or class (e.g., `my_function`, `MyClass`, `MyClass.my_method`).
  * `symbol_type`: The type of symbol (e.g., “FUNCTION”, “CLASS”, “METHOD”).
  * `summary`: The LLM-generated summary of the symbol.
  * `level`: Set to `"symbol"`.

3. **Querying**: Use `SummarySearcher` to find relevant summaries.

   ```python
   from kit import SummarySearcher


   searcher = SummarySearcher(indexer) # Pass the built indexer
   results = searcher.search("user authentication logic", top_k=3)


   for res in results:
       print(f"Score: {res['score']:.4f}")
       if res.get('level') == 'symbol':
           print(f"  Symbol: {res['symbol_name']} ({res['symbol_type']}) in {res['file_path']}")
       else:
           print(f"  File: {res['file_path']}")
       print(f"  Summary: {res['summary'][:100]}...")
       print("---")
   ```

   The `results` will contain the summary and associated metadata, including the `level` and symbol details if applicable.

## Quick start

```python
import kit
from sentence_transformers import SentenceTransformer


repo = kit.Repository("/path/to/your/project")


# 1. LLM summarizer (make sure OPENAI_API_KEY / etc. is set)
summarizer = repo.get_summarizer()


# 2. Embedding function (any model that returns list[float])
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda txt: embed_model.encode(txt).tolist()


# 3. Build the index (stored in .kit/docstring_db)
indexer = kit.DocstringIndexer(repo, summarizer)
indexer.build()


# 4. Search
searcher = kit.SummarySearcher(indexer)
for hit in searcher.search("retry back-off", top_k=5):
    print(hit["file"], "→", hit["summary"])
```

### Storage details

`DocstringIndexer` delegates persistence to any `kit.vector_searcher.VectorDBBackend`. The default backend is [`Chroma`](https://docs.trychroma.com/) and lives in `.kit/docstring_db/` inside your repo.

## Use Cases

* **Semantic Code Search**: Find code by describing what it *does*, not just what keywords it contains. (e.g., “retry back-off logic” instead of trying to guess variable names like `exponential_delay` or `MAX_RETRIES`).
* **Onboarding**: Quickly understand what different parts of a codebase are for.
* **Automated Documentation**: Use the summaries as a starting point for API docs.
* **Codebase Q\&A**: As shown in the [Codebase Q\&A Bot tutorial](/docs/tutorials/codebase-qa-bot), combine `SummarySearcher` with an LLM to answer questions about your code, using summaries to find relevant context at either the file or symbol level.

## API reference

Check docs for [`DocstringIndexer`](/docs/api/docstring-indexer) and [`SummarySearcher`](/docs/api/summary-searcher) for full signatures.

# Incremental Analysis & Caching

> High-performance caching system for faster symbol extraction

kit’s incremental analysis system provides intelligent caching that improves performance for repeated operations. By caching symbol extraction results and using sophisticated invalidation strategies, kit achieves 25x performance improvements for warm cache scenarios.

## Overview

The incremental analysis system consists of two main components:

* **FileAnalysisCache**: Manages file-level caching with multiple invalidation strategies
* **IncrementalAnalyzer**: Orchestrates analysis with performance tracking and statistics

### Performance Benefits

Real-world performance improvements on the kit repository (60k+ symbols, 7,606 files):

* **Cold cache**: 27.8s (2,158 symbols/sec)
* **Warm cache**: 0.76s (78,823 symbols/sec)
* **Speedup**: 36.5x faster

Tip

The incremental analysis system is automatically integrated with git state detection, ensuring cache invalidation when switching branches or commits.

## Efficient Change Detection

**Key Insight**: Kit only analyzes files that have actually changed, avoiding expensive re-analysis of unchanged files.

When you call `extract_symbols_incremental()`, Kit performs these steps:

1. **File Discovery**: One-time walk to find all supported files (`.py`, `.js`, `.ts`, etc.)
2. **Change Detection**: For each file, quickly check if it changed using mtime, size, and content hash
3. **Selective Analysis**: Only analyze changed files using tree-sitter
4. **Cache Retrieval**: Return cached results instantly for unchanged files

### Real-World Impact

If you have 1,000 files in your repository and only 1 file changes:

```plaintext
Analyzing 1 changed files (skipping 999 cached)
```

This selective approach is the core reason for Kit’s dramatic performance improvements - it avoids redundant work entirely.

### Behind the Scenes

The `analyze_changed_files()` method demonstrates this filtering:

```python
def analyze_changed_files(self, file_paths):
    results = {}
    changed_files = []


    # Filter to only changed files
    for file_path in file_paths:
        if self.cache.is_file_changed(file_path):
            changed_files.append(file_path)
        else:
            # Use cached results instantly
            cached_symbols = self.cache.get_cached_symbols(file_path)
            if cached_symbols:
                results[str(file_path)] = cached_symbols
                self._stats["cache_hits"] += 1


    logger.info(f"Analyzing {len(changed_files)} changed files "
                f"(skipping {len(file_paths) - len(changed_files)} cached)")


    # Only analyze the changed files
    for file_path in changed_files:
        symbols = self.analyze_file(file_path)
        results[str(file_path)] = symbols


    return results
```

Note

This is fundamentally different from traditional approaches that re-parse all files on every analysis. Kit’s incremental system scales to large codebases because analysis time is proportional to changes, not repository size.

## Quick Start

### Basic Usage

```python
from kit.repository import Repository


# Create repository instance
repo = Repository("/path/to/your/project")


# First analysis - builds cache
symbols = repo.extract_symbols_incremental()
print(f"Found {len(symbols)} symbols")


# Second analysis - uses cache (much faster)
symbols = repo.extract_symbols_incremental()
print(f"Found {len(symbols)} symbols (cached)")


# Get performance statistics
stats = repo.get_incremental_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']}")
print(f"Files analyzed: {stats['files_analyzed']}")
```

### CLI Usage

The incremental analysis system is also available through the CLI:

```bash
# Extract symbols with caching
kit symbols /path/to/repo


# View cache statistics
kit cache status /path/to/repo


# Clear cache if needed
kit cache clear /path/to/repo


# Cleanup stale entries
kit cache cleanup /path/to/repo
```

## Cache Invalidation Strategies

The system uses multiple strategies to detect file changes:

### 1. Modification Time (mtime)

Fast check using file system metadata:

```python
if cached_metadata.get("mtime") != current_metadata.get("mtime"):
    return True  # File changed
```

### 2. File Size

Quick validation of file size changes:

```python
if cached_metadata.get("size") != current_metadata.get("size"):
    return True  # File changed
```

### 3. Content Hash

Definitive change detection using SHA-256:

```python
if cached_metadata.get("hash") != current_metadata.get("hash"):
    return True  # File changed
```

### 4. Git State Detection

Automatic cache invalidation when git state changes:

```python
# Automatically detects branch switches, commits, etc.
if self._check_git_state_changed():
    self._invalidate_caches()
```

## Advanced Features

### LRU Eviction

The cache implements Least Recently Used (LRU) eviction to manage memory:

```python
from kit.incremental_analyzer import FileAnalysisCache


# Create cache with custom size limit
cache = FileAnalysisCache(
    repo_path=Path("/path/to/repo"),
    max_cache_size=5000  # Limit to 5000 files
)
```

### Batch Analysis

Analyze multiple files efficiently with automatic change filtering:

```python
from pathlib import Path


# Get incremental analyzer
analyzer = repo.incremental_analyzer


# Analyze specific files - only changed ones will be processed
files = [Path("src/main.py"), Path("src/utils.py"), Path("src/models.py")]
results = analyzer.analyze_changed_files(files)


# Results contain symbols for all files, but only changed files were analyzed
for file_path, symbols in results.items():
    print(f"{file_path}: {len(symbols)} symbols")
```

The `analyze_changed_files()` method automatically:

* ✅ Filters input files to only analyze those that changed
* ✅ Returns cached results for unchanged files
* ✅ Logs exactly how many files were skipped vs. analyzed
* ✅ Maintains the same output format regardless of cache hits/misses

### Performance Tracking

Monitor analysis performance:

```python
# Get detailed statistics
stats = repo.get_incremental_stats()


print(f"Cache hit rate: {stats['cache_hit_rate']}")
print(f"Files analyzed: {stats['files_analyzed']}")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
print(f"Average analysis time: {stats['avg_analysis_time']:.3f}s")
print(f"Cache size: {stats['cache_size_mb']:.1f}MB")
```

## Cache Management

### Cache Status

Check cache health and statistics:

```python
# Get cache statistics
cache_stats = repo.incremental_analyzer.cache.get_cache_stats()


print(f"Cached files: {cache_stats['cached_files']}")
print(f"Total symbols: {cache_stats['total_symbols']}")
print(f"Cache size: {cache_stats['cache_size_mb']:.1f}MB")
print(f"Cache directory: {cache_stats['cache_dir']}")
```

### Cache Cleanup

Remove stale entries for deleted files:

```python
# Clean up stale cache entries
removed_count = repo.cleanup_incremental_cache()
print(f"Removed {removed_count} stale entries")
```

### Cache Clearing

Clear all cached data:

```python
# Clear all cache data
repo.clear_incremental_cache()
print("Cache cleared")
```

## CLI Commands

### Cache Status

```bash
# View cache statistics
kit cache status /path/to/repo
```

Output:

```plaintext
Cache Statistics:
  Cached files: 1,234
  Total symbols: 45,678
  Cache size: 12.3MB
  Cache directory: /path/to/repo/.kit/incremental_cache
  Hit rate: 85.2%
```

### Cache Management

```bash
# Clear all cache data
kit cache clear /path/to/repo


# Clean up stale entries
kit cache cleanup /path/to/repo


# View detailed statistics
kit cache stats /path/to/repo
```

## Integration with Git

The incremental analysis system automatically integrates with git operations:

### Automatic Invalidation

Cache is automatically invalidated when:

* Switching branches (`git checkout`)
* Making commits (`git commit`)
* Merging branches (`git merge`)
* Rebasing (`git rebase`)
* Any operation that changes the git SHA

### Example Workflow

```python
# Initial analysis on main branch
repo = Repository("/path/to/repo")
symbols_main = repo.extract_symbols_incremental()
print(f"Main branch: {len(symbols_main)} symbols")


# Switch to feature branch (cache automatically invalidated)
# git checkout feature-branch


# Analysis on feature branch (cache rebuilt)
symbols_feature = repo.extract_symbols_incremental()
print(f"Feature branch: {len(symbols_feature)} symbols")


# Switch back to main (cache invalidated again)
# git checkout main


# Analysis back on main (cache rebuilt for main)
symbols_main_again = repo.extract_symbols_incremental()
print(f"Back to main: {len(symbols_main_again)} symbols")
```

## Best Practices

### 1. Use Incremental Analysis for Development

For development workflows where you’re repeatedly analyzing the same codebase:

```python
# Use incremental analysis for better performance
symbols = repo.extract_symbols_incremental()


# Instead of traditional analysis
# symbols = repo.extract_symbols()  # Slower
```

### 2. Monitor Cache Performance

Track cache effectiveness:

```python
def analyze_with_monitoring(repo):
    start_time = time.time()
    symbols = repo.extract_symbols_incremental()
    elapsed = time.time() - start_time


    stats = repo.get_incremental_stats()
    print(f"Analysis: {elapsed:.2f}s, Hit rate: {stats['cache_hit_rate']}")


    return symbols
```

### 3. Cleanup in CI/CD

Clean up cache in CI/CD environments:

```bash
# In CI/CD pipeline
kit cache cleanup /path/to/repo
kit symbols /path/to/repo --format json > symbols.json
```

### 4. Cache Size Management

Monitor and manage cache size:

```python
# Check cache size periodically
stats = repo.get_incremental_stats()
if stats['cache_size_mb'] > 100:  # 100MB limit
    repo.cleanup_incremental_cache()
```

## Troubleshooting

### Cache Not Working

If cache doesn’t seem to be working:

1. **Check git state**: Ensure you’re not switching branches frequently
2. **Verify file stability**: Check if files are being modified
3. **Monitor statistics**: Use `get_incremental_stats()` to debug

```python
# Debug cache behavior
stats = repo.get_incremental_stats()
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
print(f"Hit rate: {stats['cache_hit_rate']}")
```

### Performance Issues

If performance is still slow:

1. **Clear and rebuild cache**:

   ```python
   repo.clear_incremental_cache()
   symbols = repo.extract_symbols_incremental()
   ```

2. **Check cache size limits**:

   ```python
   # Increase cache size if needed
   analyzer = repo.incremental_analyzer
   analyzer.cache.max_cache_size = 20000
   ```

3. **Monitor file changes**:

   ```python
   # Check if files are changing unexpectedly
   for file_path in changed_files:
       if analyzer.cache.is_file_changed(file_path):
           print(f"File changed: {file_path}")
   ```

### Cache Corruption

If cache becomes corrupted:

```python
# Reset everything
repo.clear_incremental_cache()
repo.incremental_analyzer.cache.clear_cache()


# Rebuild from scratch
symbols = repo.extract_symbols_incremental()
```

## Implementation Details

### Cache Storage

Cache data is stored in the repository’s `.kit` directory:

```plaintext
.kit/
├── incremental_cache/
│   ├── analysis_metadata.json    # File metadata (mtime, size, hash)
│   └── symbols_cache.json        # Cached symbol data
```

### Memory Management

* **LRU eviction** prevents unlimited memory growth
* **Configurable cache size** (default: 10,000 files)
* **Automatic cleanup** of stale entries

### Thread Safety

The cache is designed for single-threaded use. For multi-threaded scenarios:

```python
# Create separate instances per thread
def worker_thread():
    repo = Repository("/path/to/repo")  # New instance
    symbols = repo.extract_symbols_incremental()
```

The incremental analysis system makes kit’s symbol extraction dramatically faster for development workflows while maintaining complete accuracy and reliability.

# LLM Best Practices

Providing the right context to a Large Language Model (LLM) is critical for getting accurate and relevant results when building AI developer tools with `kit`. This guide outlines best practices for assembling context using `kit` features.

### 1. File Tree (`repo.get_file_tree`)

* **Context:** Provides the overall structure of the repository or specific directories.
* **Use Cases:** Understanding project layout, locating relevant modules.
* **Prompting Tip:** Include the file tree when asking the LLM about high-level architecture or where to find specific functionality.

```yaml
# Example Context Block
Repository File Tree (partial):
src/
  __init__.py
  core/
    repo.py
    search.py
  utils/
    parsing.py
tests/
  test_repo.py
README.md
```

Caution

Use depth limits or filtering for large projects to avoid overwhelming the LLM.

### 2. Symbols (`repo.get_symbols`)

* **Context:** Lists functions, classes, variables, etc., within specified files.
* **Use Cases:** Understanding the code within a file, finding specific definitions, providing context for code generation/modification tasks.
* **Prompting Tip:** Clearly label the file path associated with the symbols.

```yaml
# Example Context Block
Symbols in src/core/repo.py:
- class Repo:
  - def __init__(self, path):
  - def get_symbols(self, file_paths):
  - def search_semantic(self, query):
- function _validate_path(path):
```

Note

Filter symbols to relevant files/modules when possible.

### 3. Code Snippets (via Symbols or `get_file_content`)

* **Context:** The actual source code of specific functions, classes, or entire files.
* **Use Cases:** Detailed code review, bug finding, explanation, modification.
* **Prompting Tip:** Provide the code for symbols identified as relevant by other context methods (e.g., symbols mentioned in a diff, search results).

```python
# Example Context Block
Code for Repo.search_semantic in src/core/repo.py:


def search_semantic(self, query):
    # ... implementation ...
    pass
```

Note

Clearly identify chunks in the prompt and prefer symbol-based chunking over line-based chunking when appropriate.

### 4. Text Search Results (`repo.search_text`)

* **Context:** Lines of code matching a specific text query.
* **Use Cases:** Finding specific variable names, API calls, error messages.
* **Prompting Tip:** Include the search query and clearly label the results.

```yaml
# Example Context Block
Text search results for "database connection":
- src/db/connect.py:15: conn = connect_database()
- src/config.py:8: DATABASE_URL = "..."
```

Note

Clearly specify the search query used to generate the results.

### 5. Symbol Usages (`repo.find_symbol_usages`)

* **Context:** Where a specific symbol (function, class) is used or called throughout the codebase. This method finds definitions and textual occurrences.
* **Use Cases:** Understanding the impact of changing a function, finding examples of how an API is used, basic dependency analysis.
* **Prompting Tip:** Specify the symbol whose usages are being listed.

```yaml
# Example Context Block
Usages of function connect_database (defined in src/db/connect.py):
- src/app.py:50: db_conn = connect_database()
- tests/test_db.py:12: mock_connect = mock(connect_database)
```

Note

Clearly indicate the symbol whose usages are being shown.

### 6. Semantic Search Results (`repo.search_semantic`)

* **Context:** Code chunks semantically similar to a natural language query.
* **Use Cases:** Finding code related to a concept (e.g., “user authentication logic”), exploring related functionality.
* **Prompting Tip:** Include the semantic query and label the results clearly.

```plaintext
# Example Context Block
Semantic search results for "user login handling":
- Chunk from src/auth/login.py (lines 25-40):
    def handle_login(username, password):
        # ... validation logic ...


- Chunk from src/models/user.py (lines 10-15):
    class User:
        # ... attributes ...
```

Note

Indicate that the results are from a semantic search, as the matches might not be exact text matches.

### 7. Diff Content

* **Context:** The specific lines added, removed, or modified in a changeset (e.g., a Git diff).
* **Use Cases:** Code review, understanding specific changes in a PR or commit.
* **Prompting Tip:** Clearly mark the diff section in the context.

```diff
# Example Context Block
Code Diff:
--- a/src/utils/parsing.py
+++ b/src/utils/parsing.py
@@ -10,5 +10,6 @@
 def parse_data(raw_data):
     # Extended parsing logic
+    data = preprocess(raw_data)
     return json.loads(data)
```

Note

Pair this context with specific line numbers for targeted analysis.

### 8. Vector Search Results (`repo.search_vectors`)

* **Context:** Code chunks similar to a given vector representation.
* **Use Cases:** Finding code related to a concept (e.g., “user authentication logic”), exploring related functionality.
* **Prompting Tip:** Include the vector query and label the results clearly.

```plaintext
# Example Context Block
Vector search results for "user login handling":
- Chunk from src/auth/login.py (lines 25-40):
    def handle_login(username, password):
        # ... validation logic ...


- Chunk from src/models/user.py (lines 10-15):
    class User:
        # ... attributes ...
```

Note

Indicate that the results are from a vector search, as the matches might not be exact text matches.

# Tree-sitter Plugin System

kit includes a plugin system that allows you to extend and customize symbol extraction for any programming language. This system enables you to:

* **Extend existing languages** with additional query patterns (e.g., detect FastAPI routes in Python)
* **Register completely new languages** with custom parsers and queries
* **Load multiple query files** per language for modular organization
* **Use custom query directories** for team-specific or project-specific patterns

## Supported Languages

kit comes with built-in support for 12+ programming languages:

* **Python** (`.py`) - Functions, classes, methods, decorators
* **JavaScript** (`.js`) - Functions, classes, variables, imports
* **TypeScript** (`.ts`, `.tsx`) - Types, interfaces, functions, classes
* **Go** (`.go`) - Functions, methods, structs, interfaces
* **Rust** (`.rs`) - Functions, structs, enums, traits
* **Java** (`.java`) - Classes, methods, fields, interfaces
* **C/C++** (`.c`) - Functions, structs, typedefs
* **Ruby** (`.rb`) - Classes, methods, modules
* **Dart** (`.dart`) - Classes, functions, mixins, enums, extensions
* **HCL/Terraform** (`.hcl`, `.tf`) - Resources, variables, modules

Each language supports comprehensive symbol extraction including:

* **Classes and interfaces** with inheritance relationships
* **Functions and methods** with parameter information
* **Variables and constants** with type information
* **Language-specific constructs** (decorators, generics, etc.)

## Table of Contents

* [Quick Start](#quick-start)
* [API Reference](#api-reference)
* [Extending Existing Languages](#extending-existing-languages)
* [Registering New Languages](#registering-new-languages)
* [Real-World Examples](#real-world-examples)
* [Best Practices](#best-practices)
* [Troubleshooting](#troubleshooting)

## Quick Start

### Extending Python for Custom Patterns

```python
from kit.tree_sitter_symbol_extractor import TreeSitterSymbolExtractor


# Define a custom query for detecting test functions
test_query = '''
(function_definition
  name: (identifier) @name
  (#match? @name "^test_")
) @definition.test_function
'''


# Save query to a file
with open('test_patterns.scm', 'w') as f:
    f.write(test_query)


# Extend Python language
TreeSitterSymbolExtractor.extend_language("python", "test_patterns.scm")


# Extract symbols with new patterns
code = '''
def test_user_login():
    pass


def regular_function():
    pass
'''


symbols = TreeSitterSymbolExtractor.extract_symbols(".py", code)
for symbol in symbols:
    print(f"{symbol['type']}: {symbol['name']}")
# Output:
# test_function: test_user_login
# function: test_user_login
# function: regular_function
```

### Working with Dart

kit includes comprehensive Dart support out of the box:

```python
from kit.tree_sitter_symbol_extractor import TreeSitterSymbolExtractor


dart_code = '''
class Calculator {
  int add(int a, int b) => a + b;


  Calculator.named(String name) : this();


  String get displayName => 'Calculator';
}


mixin Flyable {
  void fly() => print('Flying...');
}


enum Status { pending, completed }


extension StringExtension on String {
  String get reversed => split('').reversed.join('');
}
'''


symbols = TreeSitterSymbolExtractor.extract_symbols(".dart", dart_code)
for symbol in symbols:
    print(f"{symbol['type']}: {symbol['name']}")
# Output:
# class: Calculator
# function: add
# constructor: Calculator
# constructor: named
# getter: displayName
# function: fly
# enum: Status
# extension: StringExtension
# getter: reversed
```

## API Reference

### Core Methods

#### `extend_language(language: str, query_file: str) -> None`

Extend an existing language with additional query patterns.

* **language**: Language name (e.g., ‘python’, ‘javascript’, ‘dart’)
* **query\_file**: Path to .scm query file (absolute or relative to queries directory)

#### `register_language(name: str, extensions: List[str], query_files: List[str], query_dirs: Optional[List[str]] = None) -> None`

Register a completely new language.

* **name**: Language name (should match tree-sitter-language-pack)
* **extensions**: File extensions (e.g., \[‘.kt’, ‘.kts’])
* **query\_files**: List of .scm query files to load
* **query\_dirs**: Optional custom directories to search for queries

#### `list_supported_languages() -> Dict[str, List[str]]`

Returns mapping of language names to their supported file extensions.

```python
languages = TreeSitterSymbolExtractor.list_supported_languages()
print(languages)
# Output:
# {
#   'python': ['.py'],
#   'javascript': ['.js'],
#   'dart': ['.dart'],
#   'go': ['.go'],
#   ...
# }
```

#### `reset_plugins() -> None`

Reset all custom languages and extensions. Useful for testing and cleanup.

### Symbol Structure

Extracted symbols have this structure:

```python
{
    "name": "function_name",      # Symbol name
    "type": "function",           # Symbol type (function, class, method, etc.)
    "start_line": 0,             # Starting line number (0-indexed)
    "end_line": 5,               # Ending line number (0-indexed)
    "code": "def function_name():\n    pass",  # Full symbol code
    "subtype": "optional"        # Optional subtype for specialized symbols
}
```

## Extending Existing Languages

### Python: FastAPI Routes

Detect FastAPI route decorators:

```python
fastapi_query = '''
; FastAPI route handlers
(decorated_definition
  (decorator_list
    (decorator
      (call
        (attribute
          object: (identifier) @app_name
          attribute: (identifier) @http_method
        )
        arguments: (argument_list
          (string) @route_path
        )
      )
    )
  )
  definition: (function_definition
    name: (identifier) @name
  )
) @definition.route_handler
'''


TreeSitterSymbolExtractor.extend_language("python", "/path/to/fastapi.scm")
```

### Dart: Flutter Widgets

Detect Flutter widget classes:

```python
flutter_query = '''
; Flutter StatelessWidget classes
(class_definition
  name: (identifier) @name
  superclass: (identifier) @superclass
  (#match? @superclass "StatelessWidget")
) @definition.stateless_widget


; Flutter StatefulWidget classes
(class_definition
  name: (identifier) @name
  superclass: (identifier) @superclass
  (#match? @superclass "StatefulWidget")
) @definition.stateful_widget


; Build methods in widgets
(class_definition
  body: (class_body
    (method_signature
      name: (identifier) @name
      (#match? @name "build")
    )
  )
) @definition.build_method
'''


TreeSitterSymbolExtractor.extend_language("dart", "/path/to/flutter.scm")
```

### Python: Django Models

Detect Django model fields:

```python
django_query = '''
; Django model fields
(assignment
  target: (identifier) @field_name
  value: (call
    function: (attribute
      object: (identifier) @models
      attribute: (identifier) @field_type
    )
  )
) @definition.model_field


; Django Meta classes
(class_definition
  name: (identifier) @name
  (#match? @name "Meta")
) @definition.meta_class
'''


TreeSitterSymbolExtractor.extend_language("python", "/path/to/django.scm")
```

### JavaScript: React Components

Detect React functional components:

```python
react_query = '''
; React functional components
(function_declaration
  name: (identifier) @name
  (#match? @name "^[A-Z]")
  body: (block
    (return_statement
      argument: (jsx_element)
    )
  )
) @definition.react_component


; React hooks
(call_expression
  function: (identifier) @hook_name
  (#match? @hook_name "^use[A-Z]")
) @definition.hook_usage
'''


TreeSitterSymbolExtractor.extend_language("javascript", "/path/to/react.scm")
```

## Registering New Languages

### Example: Kotlin Support

```python
# Kotlin query patterns
kotlin_query = '''
; Function declarations
(function_declaration
  name: (identifier) @name
) @definition.function


; Class declarations
(class_declaration
  name: (identifier) @name
) @definition.class


; Property declarations
(property_declaration
  name: (identifier) @name
) @definition.property


; Data classes
(class_declaration
  modifiers: (modifiers
    (modifier) @data_modifier
    (#match? @data_modifier "data")
  )
  name: (identifier) @name
) @definition.data_class
'''


# Register the language
TreeSitterSymbolExtractor.register_language(
    name="kotlin",
    extensions=[".kt", ".kts"],
    query_files=["kotlin.scm"],
    query_dirs=["/path/to/custom/queries"]
)
```

### Example: Custom DSL

```python
# Register a custom domain-specific language
TreeSitterSymbolExtractor.register_language(
    name="my_dsl",
    extensions=[".mydsl"],
    query_files=["base.scm", "advanced.scm"],
    query_dirs=[
        "/company/shared/queries",
        "/project/local/queries"
    ]
)
```

## Real-World Examples

### Team Coding Standards

Enforce naming conventions across your codebase:

naming\_standards.scm

```python
(function_definition
  name: (identifier) @name
  (#match? @name "^(get|set|create|update|delete)_")
) @definition.crud_function


(class_definition
  name: (identifier) @name
  (#match? @name ".*Service$")
) @definition.service_class


(class_definition
  name: (identifier) @name
  (#match? @name ".*Repository$")
) @definition.repository_class
```

### API Documentation Generation

Extract API endpoints for documentation:

api\_patterns.scm

```python
(decorated_definition
  (decorator_list
    (decorator
      (call
        (attribute
          attribute: (identifier) @http_method
          (#match? @http_method "(get|post|put|delete|patch)")
        )
      )
    )
  )
  definition: (function_definition
    name: (identifier) @name
  )
) @definition.api_endpoint
```

### Testing Pattern Detection

Identify test functions and test classes:

test\_patterns.scm

```python
(function_definition
  name: (identifier) @name
  (#match? @name "^test_")
) @definition.test_function


(class_definition
  name: (identifier) @name
  (#match? @name "^Test")
) @definition.test_class


(function_definition
  decorators: (decorator_list
    (decorator
      (identifier) @decorator
      (#match? @decorator "pytest.fixture")
    )
  )
  name: (identifier) @name
) @definition.test_fixture
```

## Best Practices

### Query Organization

1. **Separate by Purpose**: Create different .scm files for different concerns

   ```plaintext
   queries/
   ├── python/
   │   ├── tags.scm          # Base language patterns
   │   ├── django.scm        # Django-specific patterns
   │   ├── fastapi.scm       # FastAPI-specific patterns
   │   └── testing.scm       # Testing patterns
   ├── dart/
   │   ├── tags.scm          # Base Dart patterns
   │   ├── flutter.scm       # Flutter-specific patterns
   │   └── testing.scm       # Dart testing patterns
   ```

2. **Use Descriptive Names**: Make symbol types self-documenting

   ```python
   # Good
   @definition.api_endpoint
   @definition.model_field
   @definition.test_fixture
   @definition.flutter_widget


   # Avoid
   @definition.thing
   @definition.item
   ```

3. **Comment Your Queries**: Explain complex patterns

   ```scheme
   ; Match Django model fields with specific field types
   ; Captures both the field name and field type for analysis
   (assignment
     target: (identifier) @field_name
     value: (call
       function: (attribute
         object: (identifier) @models
         attribute: (identifier) @field_type
       )
     )
   ) @definition.model_field
   ```

### Performance Considerations

1. **Use Specific Patterns**: More specific queries are faster

   ```scheme
   ; Better - specific pattern
   (function_definition
     name: (identifier) @name
     (#match? @name "^handle_")
   ) @definition.handler


   ; Slower - overly broad pattern
   (function_definition
     name: (identifier) @name
   ) @definition.function
   ```

2. **Combine Related Patterns**: Group similar patterns in one file

3. **Test Query Performance**: Use logging to monitor query compilation time

### Version Control

1. **Include Query Files**: Check .scm files into version control
2. **Document Extensions**: Maintain a README explaining custom queries
3. **Team Sharing**: Use shared query directories for team standards

## Troubleshooting

### Common Query Errors

1. **Invalid Field Name**: Field doesn’t exist in grammar

   ```plaintext
   Error: Invalid field name at row 5, column 10: slice
   ```

   **Solution**: Check the tree-sitter grammar documentation for valid field names

2. **Query Compilation Failed**: Syntax error in query

   ```plaintext
   Error: Query compile error for ext .py
   ```

   **Solution**: Validate query syntax, check parentheses matching

### Debugging Tips

1. **Enable Debug Logging**:

   ```python
   import logging
   logging.getLogger('kit.tree_sitter_symbol_extractor').setLevel(logging.DEBUG)
   ```

2. **Test Queries Incrementally**: Start with simple patterns and add complexity

3. **Check Language Support**: Verify the language is available in tree-sitter-language-pack

### Reset and Recovery

If you encounter issues with cached queries:

```python
# Reset all plugins and start fresh
TreeSitterSymbolExtractor.reset_plugins()


# Re-register your extensions
TreeSitterSymbolExtractor.extend_language("python", "your_query.scm")
```

## Advanced Usage

### Multiple Query Directories

Load queries from multiple locations with fallback priority:

```python
TreeSitterSymbolExtractor.register_language(
    name="python",
    extensions=[".py"],
    query_files=["base.scm", "company.scm", "project.scm"],
    query_dirs=[
        "/project/queries",           # Highest priority
        "/company/shared/queries",    # Medium priority
        "/home/user/.kit/queries"     # Lowest priority
    ]
)
```

### Dynamic Query Loading

Load queries based on project configuration:

```python
import yaml


def load_project_queries():
    with open('.kit-config.yml') as f:
        config = yaml.safe_load(f)


    for lang_config in config.get('languages', []):
        TreeSitterSymbolExtractor.extend_language(
            language=lang_config['name'],
            query_file=lang_config['query_file']
        )


# Usage in project setup
load_project_queries()
```

### Integration with CI/CD

Use plugins to enforce coding standards:

check\_standards.py

```python
def check_naming_conventions(file_path: str) -> List[str]:
    violations = []


    with open(file_path) as f:
        code = f.read()


    symbols = TreeSitterSymbolExtractor.extract_symbols(
        file_path.suffix, code
    )


    for symbol in symbols:
        if symbol['type'] == 'function' and not symbol['name'].startswith(('get_', 'set_', 'create_')):
            violations.append(f"Function {symbol['name']} doesn't follow naming convention")


    return violations
```

This plugin system makes Kit’s symbol extraction completely customizable while maintaining excellent performance and backward compatibility. You can now adapt Kit to work with any codebase’s specific patterns and conventions!

# The Repository Interface

The `kit.Repository` object is the backbone of the library. It serves as your primary interface for accessing, analyzing, and understanding codebases, regardless of their language or location (local path or remote Git URL).

## Why the `Repository` Object?

Interacting directly with code across different languages, file structures, and potential locations (local vs. remote) can be cumbersome. The `Repository` object provides a **unified and consistent abstraction layer** to handle this complexity.

Key benefits include:

* **Unified Access:** Provides a single entry point to read files, extract code structures (symbols), perform searches, and more.
* **Location Agnostic:** Works seamlessly with both local file paths and remote Git repository URLs (handling cloning and caching automatically when needed).
* **Language Abstraction:** Leverages `tree-sitter` parsers under the hood to understand the syntax of various programming languages, allowing you to work with symbols (functions, classes, etc.) in a standardized way.
* **Foundation for Tools:** Acts as the foundation upon which you can build higher-level developer tools and workflows, such as documentation generators, AI code reviewers, or semantic search engines.

## What Can You Do with a `Repository`?

Once you instantiate a `Repository` object pointing to your target codebase:

```python
from kit import Repository


# Point to a local project
my_repo = Repository("/path/to/local/project")


# Or point to a remote GitHub repo
# github_repo = Repository("https://github.com/owner/repo-name")


# Or analyze a specific version
# versioned_repo = Repository("https://github.com/owner/repo-name", ref="v1.2.3")
```

You can perform various code intelligence tasks:

* **Explore Structure:** Get the file tree (`.get_file_tree()`).
* **Read Content:** Access the raw content of specific files (`.get_file_content()`).
* **Understand Code:** Extract detailed information about functions, classes, and other symbols (`.extract_symbols()`).
* **Access Git Metadata:** Get current commit SHA, branch, and remote URL (`.current_sha`, `.current_branch`, `.remote_url`).
* **Search & Navigate:** Find text patterns (`.search_text()`) or semantically similar code (`.search_semantic()`).
* **Analyze Dependencies:** Find where symbols are defined and used (`.find_symbol_usages()`).
* **Prepare for LLMs:** Chunk code intelligently by lines or symbols (`.chunk_file_by_lines()`, `.chunk_file_by_symbols()`) and get code context around specific lines (`.extract_context_around_line()`).
* **Integrate with AI:** Obtain configured summarizers (`.get_summarizer()`) or vector searchers (`.get_vector_searcher()`) for advanced AI workflows.
* **Export Data:** Save the file tree, symbol information, or full repository index to structured formats like JSON (`.write_index()`, `.write_symbols()`, etc.).

The following table lists some of the key classes and tools you can access through the `Repository` object:

| Class/Tool         | Description                                    |
| ------------------ | ---------------------------------------------- |
| `Summarizer`       | Generate summaries of code using LLMs          |
| `VectorSearcher`   | Query vector index of code for semantic search |
| `DocstringIndexer` | Build vector index of LLM-generated summaries  |
| `SummarySearcher`  | Query that index                               |

Tip

For a complete list of methods, parameters, and detailed usage examples, please refer to the **[Repository Class API Reference](/api/repository/)**.

Note

## File and Directory Exclusion (.gitignore support)

By default, kit automatically ignores files and directories listed in your `.gitignore` as well as `.git/` and its contents. This ensures your indexes, symbol extraction, and searches do not include build artifacts, dependencies, or version control internals.

**Override:**

* This behavior is the default. If you want to include ignored files, you can override this by modifying the `RepoMapper` logic (see `src/kit/repo_mapper.py`) or subclassing it with custom exclusion rules.

# Repository Versioning

> Working with specific commits, tags, and branches

One of kit’s most powerful features is the ability to analyze repositories at specific points in their history. Whether you’re debugging an issue that appeared in a particular release, comparing code evolution over time, or ensuring reproducible analysis results, kit’s versioning capabilities provide the foundation for sophisticated historical code analysis.

## Why Analyze Different Versions?

### Release Analysis and Debugging

When reviewing issues that appeared in specific releases, analyzing the exact codebase state at that time is crucial:

```python
from kit import Repository


# Analyze the codebase at a specific release
repo_v1 = Repository("https://github.com/owner/project", ref="v1.2.0")
repo_v2 = Repository("https://github.com/owner/project", ref="v1.3.0")


# Compare symbol extraction between versions
symbols_v1 = repo_v1.extract_symbols("src/core/api.py")
symbols_v2 = repo_v2.extract_symbols("src/core/api.py")


# Find new or removed functions
v1_functions = {s["name"] for s in symbols_v1 if s["type"] == "function"}
v2_functions = {s["name"] for s in symbols_v2 if s["type"] == "function"}


new_functions = v2_functions - v1_functions
removed_functions = v1_functions - v2_functions


print(f"Functions added in v1.3.0: {new_functions}")
print(f"Functions removed in v1.3.0: {removed_functions}")
```

### Reproducible Code Analysis

For documentation generation, CI/CD pipelines, or research purposes, you often need reproducible results:

```python
# Always analyze the exact same version
repo = Repository("https://github.com/owner/project", ref="abc123def456")


# This will always return the same results, regardless of when you run it
file_tree = repo.get_file_tree()
symbols = repo.extract_symbols()


# Perfect for generating consistent documentation or reports
```

### Historical Trend Analysis

Understanding how codebases evolve over time reveals important patterns:

```python
# Analyze multiple releases to track complexity growth
releases = ["v1.0.0", "v1.1.0", "v1.2.0", "v1.3.0"]
complexity_data = []


for release in releases:
    repo = Repository("https://github.com/owner/project", ref=release)
    symbols = repo.extract_symbols()


    # Count functions and classes as a simple complexity metric
    function_count = sum(1 for s in symbols if s["type"] == "function")
    class_count = sum(1 for s in symbols if s["type"] == "class")


    complexity_data.append({
        "release": release,
        "functions": function_count,
        "classes": class_count,
        "total_symbols": len(symbols)
    })


print("Complexity evolution:")
for data in complexity_data:
    print(f"{data['release']}: {data['total_symbols']} symbols "
          f"({data['functions']} functions, {data['classes']} classes)")
```

### Pre-Production Analysis

Analyze feature branches or specific commits before they reach production:

```python
# Analyze a feature branch before merging
feature_repo = Repository("https://github.com/owner/project", ref="feature/new-api")
main_repo = Repository("https://github.com/owner/project", ref="main")


# Check for breaking changes
feature_symbols = feature_repo.extract_symbols("src/api/")
main_symbols = main_repo.extract_symbols("src/api/")


# Identify changes in public API
# (left up to you!)
```

## Working with Different Reference Types

### Commit SHAs

The most precise way to reference a specific state:

```python
# Full SHA (40 characters)
repo = Repository(".", ref="8cf426abe80f6cd3ab02ffc6fb11b00dd60995c8")


# Short SHA (typically 7+ characters)
repo = Repository(".", ref="8cf426a")


# Access the current commit information
print(f"Analyzing commit: {repo.current_sha}")
print(f"Short SHA: {repo.current_sha_short}")
```

### Tags and Releases

Useful for analyzing specific releases:

```python
# Semantic version tags
repo = Repository("https://github.com/owner/project", ref="v1.2.3")


# Other tag formats
repo = Repository("https://github.com/owner/project", ref="release-2024-01")
repo = Repository("https://github.com/owner/project", ref="stable")


# The tag information is preserved
print(f"Analyzing version: {repo.ref}")
```

### Branches

Analyze specific development branches:

```python
# Main development branch
repo = Repository("https://github.com/owner/project", ref="main")


# Feature branches
repo = Repository("https://github.com/owner/project", ref="develop")
repo = Repository("https://github.com/owner/project", ref="feature/user-auth")


# Release branches
repo = Repository("https://github.com/owner/project", ref="release/v2.0")
```

## Accessing Git Metadata

Kit provides access to basic git repository metadata:

```python
repo = Repository("https://github.com/owner/project", ref="v1.2.3")


# Basic git information
print(f"Current SHA: {repo.current_sha}")
print(f"Short SHA: {repo.current_sha_short}")
print(f"Branch: {repo.current_branch}")
print(f"Remote URL: {repo.remote_url}")


# Check if we're on a specific ref
print(f"Requested ref: {repo.ref}")
```

This metadata is especially useful for:

* **Logging and tracking**: Record exactly what version was analyzed
* **Cache invalidation**: Use SHA as cache keys for computed results
* **Audit trails**: Maintain records of what code was analyzed when
* **Validation**: Ensure you’re analyzing the expected version

## Practical Examples

### Documentation Generation Workflow

```python
def generate_api_docs(repo_url: str, version: str):
    """Generate API documentation for a specific version."""
    repo = Repository(repo_url, ref=version)


    # Extract all public API symbols
    api_symbols = []
    for symbol in repo.extract_symbols():
        if symbol["type"] in ["function", "class"] and not symbol["name"].startswith("_"):
            api_symbols.append(symbol)


    # Generate documentation
    docs = {
        "version": version,
        "commit": repo.current_sha,
        "generated_at": datetime.now().isoformat(),
        "api_reference": api_symbols
    }


    return docs


# Generate docs for multiple versions
for version in ["v1.0.0", "v1.1.0", "v1.2.0"]:
    docs = generate_api_docs("https://github.com/owner/project", version)
    with open(f"docs/api-{version}.json", "w") as f:
        json.dump(docs, f, indent=2)
```

### Security Audit Across Versions

```python
def audit_security_patterns(repo_url: str, versions: list):
    """Audit security patterns across multiple versions."""
    security_patterns = [
        r"password\s*=",
        r"api_key\s*=",
        r"secret\s*=",
        r"eval\s*\(",
        r"exec\s*\("
    ]


    results = {}


    for version in versions:
        repo = Repository(repo_url, ref=version)
        version_results = []


        for pattern in security_patterns:
            matches = repo.search_text(pattern, file_pattern="*.py")
            if matches:
                version_results.extend(matches)


        results[version] = {
            "commit": repo.current_sha,
            "issues_found": len(version_results),
            "details": version_results
        }


    return results


# Audit recent releases
audit_results = audit_security_patterns(
    "https://github.com/owner/project",
    ["v1.0.0", "v1.1.0", "v1.2.0"]
)
```

### Migration Impact Analysis

```python
def analyze_migration_impact(repo_url: str, before_ref: str, after_ref: str):
    """Analyze the impact of a major change or migration."""


    before_repo = Repository(repo_url, ref=before_ref)
    after_repo = Repository(repo_url, ref=after_ref)


    # Compare file structures
    before_files = {f["path"] for f in before_repo.get_file_tree() if not f["is_dir"]}
    after_files = {f["path"] for f in after_repo.get_file_tree() if not f["is_dir"]}


    # Compare symbols
    before_symbols = {s["name"] for s in before_repo.extract_symbols()}
    after_symbols = {s["name"] for s in after_repo.extract_symbols()}


    return {
        "files": {
            "added": after_files - before_files,
            "removed": before_files - after_files,
            "total_before": len(before_files),
            "total_after": len(after_files)
        },
        "symbols": {
            "added": after_symbols - before_symbols,
            "removed": before_symbols - after_symbols,
            "total_before": len(before_symbols),
            "total_after": len(after_symbols)
        },
        "metadata": {
            "before_commit": before_repo.current_sha,
            "after_commit": after_repo.current_sha
        }
    }


# Analyze impact of a major refactoring
impact = analyze_migration_impact(
    "https://github.com/owner/project",
    "v1.x-legacy",
    "v2.0-rewrite"
)
```

## Best Practices

### Choosing the Right Reference Type

* **Use commit SHAs** for maximum precision and immutability
* **Use tags** for analyzing specific releases or versions
* **Use branches** for analyzing ongoing development work
* **Avoid branch names** for long-term storage/caching (they move over time)

### Performance Considerations

```python
# Cache Repository instances when analyzing multiple aspects of the same version
repo = Repository("https://github.com/owner/project", ref="v1.2.3")


# Do multiple operations on the same repo instance
file_tree = repo.get_file_tree()
symbols = repo.extract_symbols()
search_results = repo.search_text("TODO")


# Rather than creating separate instances for each operation
```

### Error Handling

```python
def safe_repo_analysis(repo_url: str, ref: str):
    """Safely analyze a repository with proper error handling."""
    try:
        repo = Repository(repo_url, ref=ref)


        # Verify we got the expected ref
        if repo.current_sha is None:
            raise ValueError("Repository has no git metadata")


        return {
            "success": True,
            "sha": repo.current_sha,
            "symbols": repo.extract_symbols()
        }


    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid ref '{ref}': {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Analysis failed: {e}"
        }
```

Tip

**Repository Identity**: When using the REST API, repositories are identified by a deterministic ID that includes both path and ref. This means `repo@main` and `repo@v1.0.0` are treated as separate repositories, enabling precise caching and avoiding confusion between versions.

Note

**Git Requirements**: To use ref parameters with local repositories, the directory must be a git repository. For remote repositories, kit will clone the repository at the specified ref automatically.

## Integration with Other Kit Features

### CLI Usage with Versions

All kit CLI commands support the `--ref` parameter:

```bash
# Analyze symbols at a specific version
kit symbols https://github.com/owner/repo --ref v1.2.3


# Compare file trees between versions
kit file-tree https://github.com/owner/repo --ref v1.0.0 --output v1-files.json
kit file-tree https://github.com/owner/repo --ref v2.0.0 --output v2-files.json


# Export data for external analysis
kit export https://github.com/owner/repo symbols v1-symbols.json --ref v1.0.0
```

### REST API Versioning

The REST API maintains separate repository instances for different refs:

```bash
# Create repository instances for different versions
curl -X POST localhost:8000/repository \
  -d '{"path_or_url": "https://github.com/owner/repo", "ref": "v1.0.0"}'
# Returns: {"id": "abc123"}


curl -X POST localhost:8000/repository \
  -d '{"path_or_url": "https://github.com/owner/repo", "ref": "v2.0.0"}'
# Returns: {"id": "def456"}  // Different ID for different ref


# Access git metadata
curl localhost:8000/repository/abc123/git-info
```

### MCP Server Capabilities

The MCP server exposes versioning capabilities to AI agents:

```json
{
  "tool": "open_repository",
  "arguments": {
    "path_or_url": "https://github.com/owner/repo",
    "ref": "v1.2.3"
  }
}
```

This enables AI agents to perform sophisticated historical analysis and version comparison tasks automatically.

Repository versioning in kit provides the foundation for building sophisticated code analysis tools that can work across time, enabling everything from bug archaeology to compliance auditing to automated documentation generation.

# Searching

Not sure **which `kit` feature to reach for**? Use this page as a mental map of search-and-discovery tools – from plain-text grep all the way to LLM-powered semantic retrieval.

## Decision table

| Your goal                                         | Best tool                              | One-liner                           | Docs                                                             |
| ------------------------------------------------- | -------------------------------------- | ----------------------------------- | ---------------------------------------------------------------- |
| Find an exact string or regex                     | `repo.search_text()`                   | `repo.search_text("JWT", "*.go")`   | [Text search](/docs/core-concepts/semantic-search#exact-keyword) |
| List symbols in a file                            | `repo.extract_symbols()`               | `repo.extract_symbols("src/db.py")` | [Repository API](/docs/core-concepts/repository-api)             |
| See where a function is used                      | `repo.find_symbol_usages()`            | `repo.find_symbol_usages("login")`  | ^                                                                |
| Get a concise overview of a file / function       | `Summarizer`                           | `summarizer.summarize_file(path)`   | [Code summarization](/docs/core-concepts/code-summarization)     |
| Semantic search over **raw code chunks**          | `VectorSearcher`                       | `repo.search_semantic()`            | [Semantic search](/docs/core-concepts/semantic-search)           |
| Semantic search over **LLM summaries**            | `DocstringIndexer` + `SummarySearcher` | see below                           | [Docstring index](/docs/core-concepts/docstring-indexing)        |
| Build an LLM prompt with only the *relevant* code | `ContextAssembler`                     | `assembler.from_chunks(chunks)`     | [Context assembly](/docs/core-concepts/context-assembly)         |

> **Tip:** You can mix-and-match. For instance, run a docstring search first, then feed the matching files into `ContextAssembler` for an LLM chat.

## Approaches in detail

### 1. Plain-text / regex search

Fast, zero-setup, works everywhere. Use when you *know* what string you’re looking for.

```python
repo.search_text("parse_jwt", file_pattern="*.py")
```

### 2. Symbol indexing

`extract_symbols()` uses **tree-sitter** queries (Python, JS, Go, etc.) to list functions, classes, variables – handy for nav trees or refactoring tools.

### 3. LLM summarization

Generate natural-language summaries for files, classes, or functions with `Summarizer`. Great for onboarding or API docs.

### 4. Vector search (raw code)

`VectorSearcher` chunks code (symbols or lines) → embeds chunks → stores them in a local vector database. Good when wording of the query is *similar* to the code.

### 5. Docstring vector search

`DocstringIndexer` first *summarizes* code, then embeds the summary. The resulting vectors capture **intent**, not syntax; queries like “retry back-off logic” match even if the code uses exponential delays without those words.

***

Still unsure? Start with text-search (cheap), move to vector search (smart), and layer summaries when you need *meaning* over *matching*.

# Semantic Searching

Experimental

Vector / semantic search is an early feature. APIs, CLI commands, and index formats may change in future releases without notice.



Semantic search allows you to find code based on meaning rather than just keywords. Kit supports semantic code search using vector embeddings and ChromaDB, enabling you to search for code using natural language queries.

## How it works

* Chunks your codebase (by symbols or lines)
* Embeds each chunk using your chosen model (OpenAI, HuggingFace, etc)
* Stores embeddings in a local ChromaDB vector database
* Lets you search for code using natural language or code-like queries

## Quick Start

```python
from kit import Repository
from sentence_transformers import SentenceTransformer


# Use any embedding model you like
model = SentenceTransformer("all-MiniLM-L6-v2")
def embed_fn(texts):
    return model.encode(texts).tolist()


repo = Repository("/path/to/codebase")
vs = repo.get_vector_searcher(embed_fn=embed_fn)
vs.build_index()  # Index all code chunks (run once, or after code changes)


results = repo.search_semantic("How is authentication handled?", embed_fn=embed_fn)
for hit in results:
    print(hit["file"], hit.get("name"), hit.get("type"), hit.get("code"))
# Example output:
# src/kit/auth.py login function def login(...): ...
# src/kit/config.py AUTH_CONFIG variable AUTH_CONFIG = {...}
```

## Configuration

### Required: Embedding Function

You must provide an embedding function (`embed_fn`) when first accessing semantic search features via `repo.get_vector_searcher()` or `repo.search_semantic()`.

This function takes a list of text strings and returns a list of corresponding embedding vectors.

```python
from kit import Repository


repo = Repository("/path/to/repo")


# Define the embedding function wrapper
def embed_fn(texts: list[str]) -> list[list[float]]:
    # Adapt this to your specific embedding library/API
    return get_embeddings(texts)


# Pass the function when searching
results = repo.search_semantic("database connection logic", embed_fn=embed_fn)


# Or when getting the searcher explicitly
vector_searcher = repo.get_vector_searcher(embed_fn=embed_fn)
```

### Choosing an Embedding Model

`kit` is model-agnostic: pass any function `List[str] -> List[List[float]]`.

#### Local (Open-Source) Models

Use [`sentence-transformers`](https://www.sbert.net/) models for fast, local inference:

```python
from sentence_transformers import SentenceTransformer


# Popular lightweight model (100 MB-ish download)
model = SentenceTransformer("all-MiniLM-L6-v2")
def embed_fn(texts: list[str]) -> list[list[float]]:
    return model.encode(texts).tolist()


# Or try larger, more accurate models
model = SentenceTransformer("all-mpnet-base-v2")  # ~420MB, better quality
```

#### Cloud API Models

Use OpenAI or other cloud embedding services:

```python
import openai


def embed_fn(texts: list[str]) -> list[list[float]]:
    """OpenAI embedding function with batching support."""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [data.embedding for data in response.data]


# Alternative: single text fallback for simple APIs
def embed_fn_single(texts: list[str]) -> list[list[float]]:
    """If your API only supports single strings."""
    embeddings = []
    for text in texts:
        resp = openai.embeddings.create(model="text-embedding-3-small", input=[text])
        embeddings.append(resp.data[0].embedding)
    return embeddings
```

#### Batching Support

`VectorSearcher` will attempt to call your `embed_fn` with a *list* of texts for efficiency. If your function only supports single strings, it still works (falls back internally).

### Backend Configuration

`kit`’s `VectorSearcher` uses a pluggable backend system for storing and querying vector embeddings. Currently, the primary supported and default backend is **ChromaDB**.

#### ChromaDB (Default)

When you initialize `VectorSearcher` without specifying a `backend` argument, `kit` automatically uses an instance of `ChromaDBBackend`.

**Configuration Options:**

* **`persist_dir` (Optional\[str]):** Specifies where the ChromaDB index will be stored on disk.

  * If you provide a path: `repo.get_vector_searcher(persist_dir="./my_index")`
  * If no `persist_dir` is specified, defaults to `YOUR_REPO_PATH/.kit/vector_db/`
  * Persisting the index allows you to reuse it across sessions without re-indexing

```python
# Example: Initialize with custom persist directory
vector_searcher = repo.get_vector_searcher(
    embed_fn=my_embedding_function,
    persist_dir="./my_custom_kit_vector_index"
)


# Building the index (first time or to update)
vector_searcher.build_index()


# Later, to reuse the persisted index:
vector_searcher_reloaded = repo.get_vector_searcher(
    embed_fn=my_embedding_function,
    persist_dir="./my_custom_kit_vector_index"
)
results = vector_searcher_reloaded.search("my query")
```

#### Other Backends

While the `VectorDBBackend` interface is designed to support other vector databases, ChromaDB is the primary focus for now. If you need other backends like Faiss, please raise an issue on the kit GitHub repository.

## Usage Patterns

### Chunking Strategy

Control how your code is broken into searchable chunks:

```python
# Default: chunk by symbols (functions, classes, variables)
vs.build_index(chunk_by="symbols")


# Alternative: chunk by lines (~50-line blocks)
vs.build_index(chunk_by="lines")  # Useful for unsupported languages
```

`chunk_by="symbols"` (default) extracts functions/classes/variables via the existing AST parser. This is usually what you want.

You can re-index at any time; the previous collection is cleared automatically.

### Persisting & Re-using an Index

The index lives under `.kit/vector_db` by default (one Chroma collection per path).

```python
vs = repo.get_vector_searcher(embed_fn, persist_dir=".kit/my_index")
vs.build_index()
# … later …
searcher = repo.get_vector_searcher(embed_fn, persist_dir=".kit/my_index")
results = searcher.search("add user authentication")
```

### Docstring Index

Prefer *meaning-first* search? Instead of embedding raw code you can build an index of LLM-generated summaries:

```text
DocstringIndexer → SummarySearcher
```

See **[Docstring-Based Vector Index](/docs/core-concepts/docstring-indexing)** for details.

### Feeding Results to an LLM

Combine `VectorSearcher` with `ContextAssembler` to build an LLM prompt containing only *relevant* code:

```python
from kit import ContextAssembler


chunks = repo.search_semantic("jwt auth flow", embed_fn=embed_fn, top_k=10)
assembler = ContextAssembler(max_chars=12_000)
context = assembler.from_chunks(chunks)
llm_response = my_llm.chat(prompt + context)
```

### Advanced Usage Examples

#### Multi-Query Search

```python
queries = [
    "database connection setup",
    "user authentication logic",
    "error handling patterns"
]


all_results = []
for query in queries:
    results = repo.search_semantic(query, embed_fn=embed_fn, top_k=5)
    all_results.extend(results)


# Deduplicate by file path
unique_files = {r["file"]: r for r in all_results}
```

#### Filtering Results

```python
# Search only in specific directories
results = repo.search_semantic("api endpoints", embed_fn=embed_fn)
api_results = [r for r in results if "src/api/" in r["file"]]


# Search only for functions
function_results = [r for r in results if r.get("type") == "function"]
```

## Best Practices

### Performance Tips

* **Index size**: Indexing a very large monorepo may take minutes. Consider running on CI and committing `.kit/vector_db`.
* **Chunking**: Use `chunk_by="symbols"` for better semantic boundaries
* **Model selection**: Balance model size vs. quality based on your needs
* **Batch embedding**: Use APIs that support batch embedding for better performance

### Search Quality

* **Clean code**: Embeddings are language-agnostic – comments & docs influence similarity too. Clean code/comments improve search.
* **Query formulation**: Use natural language descriptions of what you’re looking for
* **Combine approaches**: Exact-keyword search (`repo.search_text()`) can still be faster for quick look-ups; combine both techniques.

### Production Considerations

```python
# Example: Production-ready setup with error handling
import logging


def safe_semantic_search(repo_path: str, query: str, top_k: int = 5):
    try:
        repo = Repository(repo_path)


        # Check if index exists
        vector_searcher = repo.get_vector_searcher(embed_fn=embed_fn)


        # Build index if needed (check if collection is empty)
        try:
            test_results = vector_searcher.search("test", top_k=1)
            if not test_results:
                logging.info("Building semantic index...")
                vector_searcher.build_index()
        except Exception:
            logging.info("Building semantic index...")
            vector_searcher.build_index()


        return repo.search_semantic(query, embed_fn=embed_fn, top_k=top_k)


    except Exception as e:
        logging.error(f"Semantic search failed: {e}")
        # Fallback to text search
        return repo.search_text(query)
```

## Limitations & Future Plans

* **CLI support**: The CLI (`kit search` / `kit serve`) currently performs **text** search only. A semantic variant is planned.
* **Language support**: Works with any language that kit can parse, but quality depends on symbol extraction
* **Index management**: Future versions may include index cleanup, optimization, and migration tools

Note

For complex production deployments, consider running embedding models on dedicated infrastructure and using the REST API or MCP server for distributed access.

# Tool-Calling with kit

Modern LLM runtimes (OpenAI *function-calling*, Anthropic *tools*, etc.) let you hand the model a **menu of functions** in JSON-Schema form. The model then plans its own calls – no hard-coded if/else trees required.

With `kit` you can expose its code-intelligence primitives (search, symbol extraction, summaries, etc) and let the LLM decide which one to execute in each turn. Your app stays declarative: *“Here are the tools, here is the user’s question, please help.”*

In practice that means you can drop `kit` into an existing chat agent with by just registering the schema and making the calls as needed. The rest of this guide shows the small amount of glue code needed and the conversational patterns that emerge.

You may also be interested in kit’s [MCP integration](../mcp/using-kit-with-mcp), which can achieve similar goals.

`kit` exposes its code-intelligence primitives as **callable tools**. Inside Python you can grab the JSON-Schema list with a single helper (`kit.get_tool_schemas()`) and hand that straight to your LLM runtime. Once the schema is registered the model can decide *when* to call:

* `open_repository`
* `search_code`
* `extract_symbols`
* `find_symbol_usages`
* `get_file_tree`, `get_file_content`
* `get_code_summary`

This page shows the minimal JSON you need, the decision patterns the model will follow, and a multi-turn example.

## 1 Register the tools

### OpenAI Chat Completions

```python
from openai import OpenAI
from kit import get_tool_schemas


client = OpenAI()


# JSON-Schema for every kit tool
functions = get_tool_schemas()


messages = [
    {"role": "system", "content": "You are an AI software engineer, some refer to as the 'Scottie Scheffler of Programming'. Feel free to call tools when you need context."},
]
```

`functions` is a list of JSON-Schema objects. Pass it directly as the `tools`/`functions` parameter to `client.chat.completions.create()`.

### Anthropic (messages-v2)

```python
from anthropic import Anthropic
anthropic = Anthropic()


# JSON-Schema for every kit tool
functions = get_tool_schemas()


response = anthropic.messages.create(
    model="claude-3-7-sonnet-20250219",
    system="You are an AI software engineer…",
    tools=functions,
    messages=[{"role": "user", "content": "I got a test failure around FooBar.  Help me."}],
)
```

## 2 When should the model call which tool?

Below is the heuristic kit’s own prompts (and our internal dataset) encourage. You **don’t** need to hard-code this logic—the LLM will pick it up from the tool names / descriptions—but understanding the flow helps you craft better conversation instructions.

| Situation                                             | Suggested tool(s)                                           |
| ----------------------------------------------------- | ----------------------------------------------------------- |
| No repo open yet                                      | `open_repository` (first turn)                              |
| “What files mention X?”                               | `search_code` (fast regex)                                  |
| “Show me the function/class definition”               | `get_file_content` *or* `extract_symbols`                   |
| ”Where else is `my_func` used?“                       | 1) `extract_symbols` (file-level) → 2) `find_symbol_usages` |
| ”Summarize this file/function for the PR description” | `get_code_summary`                                          |
| IDE-like navigation                                   | `get_file_tree` + `get_file_content`                        |

A **typical multi-turn session**:

```plaintext
User: I keep getting KeyError("user_id") in prod.


Assistant (tool call): search_code {"repo_id": "42", "query": "KeyError(\"user_id\")"}


Tool result → 3 hits returned (files + line numbers)


Assistant: The error originates in `auth/session.py` line 88.  Shall I show you that code?


User: yes, show me.


Assistant (tool call): get_file_content {"repo_id": "42", "file_path": "auth/session.py"}


Tool result → file text


Assistant: Here is the snippet … (explanatory text)
```

## 3 Prompt orchestration: system / developer messages

Tool-calling conversations have **three channels of intent**:

1. **System prompt** – your immutable instructions (e.g. *“You are an AI software-engineer agent.”*)
2. **Developer prompt** – *app-level* steering: *“If the user asks for code you have not seen, call `get_file_content` first.”*
3. **User prompt** – the human’s actual message.

`kit` does *not* impose a format—you simply include the JSON-schema from `kit.get_tool_schemas()` in your `tools` / `functions` field and add whatever system/developer guidance you choose. A common pattern is:

```python
system = """You are an AI software-engineer.
Feel free to call tools when they help you answer precisely.
When showing code, prefer the smallest snippet that answers the question.
"""


developer = """Available repos are already open in the session.
Call `search_code` before you attempt to answer questions like
  *"Where is X defined?"* or *"Show references to Y"*.
Use `get_code_summary` before writing long explanations of unfamiliar files.
"""


messages = [
    {"role": "system", "content": system},
    {"role": "system", "name": "dev-instructions", "content": developer},
    {"role": "user", "content": user_query},
]
```

Because the developer message is separate from the user’s content it can be updated dynamically by your app (e.g. after each tool result) without contaminating the visible chat transcript.

## 4 Streaming multi-tool conversations

Nothing prevents the LLM from chaining calls:

1. `extract_symbols` on the failing file.
2. Pick the function, then `get_code_summary` for a concise explanation.

Frameworks like **LangChain**, **LlamaIndex** or **CrewAI** can route these calls automatically when you surface kit’s tool schema as their “tool” object.

```python
from langchain_community.tools import Tool
kit_tools = [Tool.from_mcp_schema(t, call_kit_server) for t in functions]
```

## 5 Security considerations

`get_file_content` streams raw code. If you expose kit to an external service:

* Restrict `open_repository` to a safe path.
* Consider stripping secrets from returned text.
* Disable `get_file_content` for un-trusted queries and rely on `extract_symbols` + `get_code_summary` instead.

## 6 In-process (no extra server)

If your application is **already written in Python** you don’t have to spawn any servers at all—just keep a `Repository` instance in memory and expose thin wrappers as tools/functions:

```python
from typing import TypedDict
from kit import Repository


repo = Repository("/path/to/repo")


class SearchArgs(TypedDict):
    query: str
    pattern: str


def search_code(args: SearchArgs):
    return repo.search_text(args["query"], file_pattern=args.get("pattern", "*.py"))
```

Then register the wrapper with your tool-calling framework:

```python
from openai import OpenAI
client = OpenAI()


functions = [
    {
        "name": "search_code",
        "description": "Regex search across the active repository",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "pattern": {"type": "string", "default": "*.py"},
            },
            "required": ["query"],
        },
    },
]


client.chat.completions.create(
    model="gpt-4o",
    tools=functions,
    messages=[...]
)
```

Because everything stays in the same process you avoid IPC/JSON overhead and can share caches (e.g. the `RepoMapper` symbol index) across calls.

**Tip:** if you later need multi-language tooling or a separate sandbox you can still swap in the MCP server without touching your prompt logic—the function schema stays the same.

***

# Kit Project Roadmap

This document outlines the current capabilities of the `kit` library and a potential roadmap for its future development. It’s a living document and will evolve as the project progresses.

## Core Philosophy

`kit` aims to be a comprehensive Python toolkit for advanced code understanding, analysis, and interaction, with a strong emphasis on leveraging Large Language Models (LLMs) where appropriate. It’s designed to be modular, extensible, and developer-friendly.

## Current Capabilities

As of now, `kit` provides the following core functionalities:

Repository Interaction

The `Repository` class acts as a central hub for accessing various code analysis features for a given codebase.

Code Mapping & Symbols

`RepoMapper` provides structural and symbol information from code files, using Tree-sitter for multi-language support and incremental updates.

Code Summarization

The `Summarizer` class, supporting multiple LLM providers (e.g., OpenAI, Anthropic, Google), generates summaries for code files, functions, and classes.

Docstring Indexing & Search

The `DocstringIndexer` generates and embeds AI-powered summaries (dynamic docstrings) for code elements. The `SummarySearcher` queries this index for semantic understanding and retrieval based on code intent.

Code Search

Includes `CodeSearcher` for literal/regex searches, and `VectorSearcher` for semantic search on raw code embeddings. For semantic search on AI-generated summaries, see “Docstring Indexing & Search”.

LLM Context Building

`LLMContext` helps in assembling relevant code snippets and information into effective prompts for LLMs.

## Planned Enhancements & Future Directions

Here are some areas we’re looking to improve and expand upon:

### 1. Enhanced Code Intelligence

* **`RepoMapper` & Symbol Extraction:**

  * **Deeper Language Insights:** Beyond basic symbol extraction, explore richer semantic information (e.g., variable types, function signatures in more detail).
  * **Custom Symbol Types:** Allow users to define and extract custom symbol types relevant to their specific frameworks or DSLs.
  * **Robustness:** Continue to improve `.gitignore` handling and parsing of various project structures.
  * **Performance:** Optimize scanning for very large repositories.

* **`CodeSearcher`:**

  * **Full File Exclusion:** Implement robust `.gitignore` and other ignore file pattern support.
  * **Advanced Search Options:** Add features like whole-word matching, and consider more powerful query syntax.
  * **Performance:** Explore integration with native search tools (e.g., `ripgrep`) as an optional backend for speed.

* **`VectorSearcher` (Semantic Search):**

  * **Configurability:** Offer more choices for embedding models, chunking strategies, and vector database backends for raw code embeddings.
  * **Hybrid Search:** Explore combining keyword and semantic search for optimal results.
  * **Index Management:** Tools for easier creation, updating, and inspection of semantic search indexes.

* **Docstring Indexing & Search Enhancements:**

  * Explore advanced indexing strategies (e.g., hierarchical summaries, metadata filtering for summary search).
  * Improve management and scalability of summary vector stores.
  * Investigate hybrid search techniques combining summary semantics with keyword precision.

### 2. Advanced LLM Integration

* **`Summarizer`:**

  * **Granular Summaries Refinement:** Refine and expand granular summaries for functions and classes, ensuring broad language construct coverage and exploring different summary depths.
  * **Multi-LLM Support Expansion:** Expand and standardize multi-LLM support, facilitating easier integration of new cloud providers, local models, and enhancing common configuration interfaces.
  * **Customizable Prompts:** Allow users more control over the prompts used for summarization.

* **`LLMContext`:**

  * **Smarter Context Retrieval:** Develop more sophisticated strategies for selecting the most relevant context for different LLM tasks (e.g., using call graphs, semantic similarity, and historical data).
  * **Token Optimization:** Implement techniques to maximize information density within LLM token limits.

### 3. Code Transformation & Generation

* **Refactoring Tools:** Leverage `kit`’s understanding of code to suggest or perform automated refactoring.
* **Code Generation:** Explore LLM-powered code generation based on existing codebase patterns or natural language descriptions.
* **Documentation Generation:** Automate the creation or updating of code documentation using `kit`’s analysis and LLM capabilities.

### 4. Broader Language & Framework Support

* **Tree-sitter Queries:** Continuously expand and refine Tree-sitter queries for robust support across more programming languages and to address specific parsing challenges (e.g., HCL resource extraction noted previously).
* **Framework Awareness:** Develop extensions or plugins that provide specialized understanding for popular frameworks (e.g., Django, React, Spring).

### 5. Usability & Developer Experience

* **Comprehensive Testing:** Ensure high test coverage for all modules and functionalities.
* **Documentation:** Maintain high-quality, up-to-date documentation, including API references, tutorials, and practical recipes.
* **CLI Development:** Develop a more feature-rich and user-friendly command-line interface for common `kit` operations.
* ✅ **IDE Integration:** Explore possibilities for integrating `kit`’s features into popular IDEs via plugins, MCP, or Language Server Protocol (LSP) extensions.
* **REST API Service:** Develop a comprehensive REST API service to make `kit`’s capabilities accessible to non-Python users and applications. This would allow developers using any programming language to leverage `kit`’s code intelligence features through standard HTTP requests.

### 6. Cross-Language & Cross-Platform Support

* **REST API & Service Layer:** Expand the REST API service to provide comprehensive access to all `kit` features:

  * **Containerized Deployment:** Provide Docker images and deployment templates for easy self-hosting.
  * **Client Libraries:** Develop official client libraries for popular languages (TypeScript, Go, Rust) to interact with the `kit` API.
  * **Authentication & Multi-User Support:** Implement secure authentication and multi-user capabilities for shared deployments.
  * **Webhooks & Events:** Support webhook integrations for code events and analysis results.

### 7. Community & Extensibility

* **Plugin Architecture:** Design `kit` with a clear plugin architecture to allow the community to easily add new languages, analysis tools, or LLM integrations.

This roadmap is ambitious, and priorities will be adjusted based on user feedback and development progress.

# Running Tests

To run tests using uv and pytest, first ensure you have the development dependencies installed:

```sh
# Install all deps
uv pip install -e .
```

Then, run the full test suite using:

```sh
uv run pytest
```

Or to run a specific test file:

```sh
uv run pytest tests/test_hcl_symbols.py
```

## Code Style and Formatting

Kit uses [Ruff](https://docs.astral.sh/ruff/) for linting, formatting, and import sorting with a line length of 120 characters. Our configuration can be found in `pyproject.toml`.

To check your code against our style guidelines:

```sh
# Run linting checks
ruff check .


# Check format (doesn't modify files)
ruff format --check .
```

To automatically fix linting issues and format your code:

```sh
# Fix linting issues
ruff check --fix .


# Format code
ruff format .
```

These checks are enforced in CI, so we recommend running them locally before pushing changes.

# Adding New Languages

* To add a new language:

  1. Add a tree-sitter grammar and build it (see [tree-sitter docs](https://tree-sitter.github.io/tree-sitter/creating-parsers)).
  2. Add a `queries/<lang>/tags.scm` file with queries for symbols you want to extract.
  3. Add the file extension to `TreeSitterSymbolExtractor.LANGUAGES`.
  4. Write/expand tests for the new language.

**Why?**

* This approach lets you support any language with a tree-sitter grammar—no need to change core logic.
* `tags.scm` queries make symbol extraction flexible and community-driven.

# Command Line Interface

> Complete guide to kit's CLI commands for repository analysis, symbol extraction, and AI-powered workflows

kit provides a comprehensive command-line interface for repository analysis, symbol extraction, and AI-powered development workflows. All commands support both human-readable and machine-readable output formats for seamless integration with other tools.

## Installation & Setup

```bash
# Install kit
pip install cased-kit


# Verify installation
kit --version


# Get help for any command
kit --help
kit <command> --help
```

## Core Commands

### Repository Analysis

#### `kit symbols`

Extract symbols (functions, classes, variables) from a repository with intelligent caching for 25-36x performance improvements.

```bash
kit symbols <repository-path> [OPTIONS]
```

**Options:**

* `--format, -f <format>`: Output format (`table`, `json`, `names`, `plain`)
* `--pattern, -p <pattern>`: File pattern filter (e.g., `*.py`, `src/**/*.js`)
* `--output, -o <file>`: Save output to file
* `--type, -t <type>`: Filter by symbol type (`function`, `class`, `variable`, etc.)

**Examples:**

```bash
# Extract all symbols (uses incremental analysis for speed)
kit symbols /path/to/repo


# Get only Python functions as JSON
kit symbols /path/to/repo --pattern "*.py" --type function --format json


# Export to file for further analysis
kit symbols /path/to/repo --output symbols.json


# Quick symbol names for scripting
kit symbols /path/to/repo --format names | grep "test_"
```

#### `kit file-tree`

Display repository structure with file type indicators and statistics.

```bash
kit file-tree <repository-path> [OPTIONS]
```

**Options:**

* `--path, -p <subpath>`: Subdirectory path to show tree for (relative to repo root)
* `--output, -o <file>`: Save output to file

**Examples:**

```bash
# Show full repository structure
kit file-tree /path/to/repo


# Show structure for specific subdirectory
kit file-tree /path/to/repo --path src
kit file-tree /path/to/repo -p src/components


# Export structure as JSON
kit file-tree /path/to/repo --output structure.json


# Analyze specific directory and export
kit file-tree /path/to/repo --path src/api --output api-structure.json
```

#### `kit search`

Fast text search across repository files with regex support.

```bash
kit search <repository-path> <query> [OPTIONS]
```

**Options:**

* `--pattern, -p <pattern>`: File pattern filter
* `--output, -o <file>`: Save output to file

**Examples:**

```bash
# Search for function definitions
kit search /path/to/repo "def.*login"


# Search in specific files
kit search /path/to/repo "TODO" --pattern "*.py"
```

#### `kit grep`

Perform fast literal grep search on repository files using system grep. By default, excludes common build directories, cache folders, and hidden directories for optimal performance.

```bash
kit grep <repository-path> <pattern> [OPTIONS]
```

**Options:**

* `--case-sensitive/--ignore-case, -c/-i`: Case sensitive search (default: case-sensitive)
* `--include <pattern>`: Include files matching glob pattern (e.g., ‘\*.py’)
* `--exclude <pattern>`: Exclude files matching glob pattern
* `--max-results, -n <count>`: Maximum number of results to return (default: 1000)
* `--directory, -d <path>`: Limit search to specific directory within repository
* `--include-hidden`: Include hidden directories in search (default: false)
* `--output, -o <file>`: Save output to JSON file

**Automatic Exclusions:** By default, kit grep excludes common directories for better performance:

* **Build/Cache**: `__pycache__`, `node_modules`, `dist`, `build`, `target`, `.cache`
* **Hidden**: `.git`, `.vscode`, `.idea`, `.github`, `.terraform`
* **Virtual Environments**: `.venv`, `venv`, `.tox`
* **Language-Specific**: `vendor` (Go/PHP), `deps` (Elixir), `_build` (Erlang)

**Examples:**

```bash
# Basic literal search (excludes common directories automatically)
kit grep /path/to/repo "TODO"


# Case insensitive search
kit grep /path/to/repo "function" --ignore-case


# Search only in specific directory
kit grep /path/to/repo "class" --directory "src/api"


# Search only Python files in src directory
kit grep /path/to/repo "class" --include "*.py" --directory "src"


# Include hidden directories (search .github, .vscode, etc.)
kit grep /path/to/repo "workflow" --include-hidden


# Exclude test files but include hidden dirs
kit grep /path/to/repo "import" --exclude "*test*" --include-hidden


# Complex filtering with directory limitation
kit grep /path/to/repo "api_key" --include "*.py" --exclude "*test*" --directory "src" --ignore-case


# Limit results and save to file
kit grep /path/to/repo "error" --max-results 50 --output errors.json
```

**Performance Tips:**

* Use `--directory` to limit search scope in large repositories
* Default exclusions make searches 3-5x faster in typical projects
* Use `--include-hidden` only when specifically searching configuration files
* For regex patterns, use `kit search` instead

**Note:** Requires system `grep` command to be available in PATH. For cross-platform regex search, use `kit search` instead.

#### `kit dependencies`

Analyze and visualize code dependencies within a repository. Supports Python and Terraform dependency analysis with features including dependency graph generation, circular dependency detection, module-specific analysis, and visualization generation.

```bash
kit dependencies <repository-path> --language <language> [OPTIONS]
```

**Options:**

* `--language, -l <language>`: Language to analyze (required: `python`, `terraform`)
* `--output, -o <file>`: Output to file instead of stdout
* `--format, -f <format>`: Output format (`json`, `dot`, `graphml`, `adjacency`)
* `--visualize, -v`: Generate visualization (requires Graphviz)
* `--viz-format <format>`: Visualization format (`png`, `svg`, `pdf`)
* `--cycles, -c`: Show only circular dependencies
* `--llm-context`: Generate LLM-friendly context description
* `--module, -m <name>`: Analyze specific module/resource
* `--include-indirect, -i`: Include indirect dependencies (for module analysis)

**Examples:**

```bash
# Analyze Python dependencies with JSON output
kit dependencies /path/to/repo --language python


# Generate DOT format for Graphviz
kit dependencies /path/to/repo --language python --format dot --output deps.dot


# Create visualization directly
kit dependencies /path/to/repo --language terraform --visualize --viz-format svg


# Check for circular dependencies
kit dependencies /path/to/repo --language python --cycles


# Analyze specific Python module
kit dependencies /path/to/repo --language python --module "myproject.auth" --include-indirect


# Generate LLM context for AI analysis
kit dependencies /path/to/repo --language python --llm-context --output context.md


# Terraform infrastructure analysis
kit dependencies /path/to/repo --language terraform --format graphml --output infrastructure.graphml


# Combined analysis with visualization
kit dependencies /path/to/repo --language python --cycles --visualize --output analysis.json
```

**Sample Output:**

```plaintext
🔍 Analyzing python dependencies...
📊 Found 127 components in the dependency graph
📈 Summary: 45 internal, 82 external dependencies
✅ No circular dependencies found!
```

**Visualization Requirements:**

* Install Graphviz: `brew install graphviz` (macOS) or `apt-get install graphviz` (Linux)
* Supports PNG, SVG, and PDF output formats

#### `kit usages`

Find all usages of a specific symbol across the repository.

```bash
kit usages <repository-path> <symbol-name> [OPTIONS]
```

**Options:**

* `--symbol-type, -t <type>`: Filter by symbol type
* `--output, -o <file>`: Save output to file
* `--format, -f <format>`: Output format

**Examples:**

```bash
# Find all usages of a function
kit usages /path/to/repo "calculate_total"


# Find class usages with JSON output
kit usages /path/to/repo "UserModel" --symbol-type class --format json
```

### Cache Management

#### `kit cache`

Manage the incremental analysis cache for optimal performance.

```bash
kit cache <action> <repository-path> [OPTIONS]
```

**Actions:**

* `status`: Show cache statistics and health
* `clear`: Clear all cached data
* `cleanup`: Remove stale entries for deleted files
* `stats`: Show detailed performance statistics

**Examples:**

```bash
# Check cache status
kit cache status /path/to/repo


# Clear cache if needed
kit cache clear /path/to/repo


# Clean up stale entries
kit cache cleanup /path/to/repo


# View detailed statistics
kit cache stats /path/to/repo
```

**Sample Output:**

```plaintext
Cache Statistics:
  Cached files: 1,234
  Total symbols: 45,678
  Cache size: 12.3MB
  Cache directory: /path/to/repo/.kit/incremental_cache
  Hit rate: 85.2%
  Files analyzed: 156
  Cache hits: 1,078
  Cache misses: 156
  Average analysis time: 0.023s
```

### AI-Powered Workflows

#### `kit summarize`

Generate AI-powered pull request summaries with optional PR body updates.

```bash
kit summarize <pr-url> [OPTIONS]
```

**Options:**

* `--plain, -p`: Output raw summary content (no formatting)
* `--dry-run, -n`: Generate summary without posting to GitHub
* `--model, -m <model>`: Override LLM model for this summary
* `--config, -c <file>`: Use custom configuration file
* `--update-pr-body`: Add summary to PR description

**Examples:**

```bash
# Generate and post PR summary
kit summarize https://github.com/owner/repo/pull/123


# Dry run (preview without posting)
kit summarize --dry-run https://github.com/owner/repo/pull/123


# Update PR body with summary
kit summarize --update-pr-body https://github.com/owner/repo/pull/123


# Use specific model
kit summarize --model claude-3-5-haiku-20241022 https://github.com/owner/repo/pull/123


# Clean output for piping
kit summarize --plain https://github.com/owner/repo/pull/123
```

#### `kit commit`

Generate intelligent commit messages from staged git changes.

```bash
kit commit [repository-path] [OPTIONS]
```

**Options:**

* `--dry-run, -n`: Show generated message without committing
* `--model, -m <model>`: Override LLM model
* `--config, -c <file>`: Use custom configuration file

**Examples:**

```bash
# Generate and commit with AI message
git add .
kit commit


# Preview message without committing
git add src/auth.py
kit commit --dry-run


# Use specific model
kit commit --model gpt-4o-mini-2024-07-18


# Specify repository path
kit commit /path/to/repo --dry-run
```

**Sample Output:**

```plaintext
Generated commit message:
feat(auth): add JWT token validation middleware


- Implement JWTAuthMiddleware for request authentication
- Add token validation with signature verification
- Include error handling for expired and invalid tokens
- Update middleware registration in app configuration


Commit? [y/N]: y
```

### Content Processing

#### `kit context`

Extract contextual code around specific lines for LLM analysis.

```bash
kit context <repository-path> <file-path> <line-number> [OPTIONS]
```

**Options:**

* `--lines, -n <count>`: Context lines around target (default: 10)
* `--output, -o <file>`: Save output to JSON file

**Examples:**

```bash
# Get context around a specific line
kit context /path/to/repo src/main.py 42


# Export context for analysis
kit context /path/to/repo src/utils.py 15 --output context.json
```

#### `kit chunk-lines`

Split file content into line-based chunks for LLM processing.

```bash
kit chunk-lines <repository-path> <file-path> [OPTIONS]
```

**Options:**

* `--max-lines, -n <count>`: Maximum lines per chunk (default: 50)
* `--output, -o <file>`: Save output to JSON file

**Examples:**

```bash
# Default chunking (50 lines)
kit chunk-lines /path/to/repo src/large-file.py


# Smaller chunks for detailed analysis
kit chunk-lines /path/to/repo src/main.py --max-lines 20


# Export chunks for LLM processing
kit chunk-lines /path/to/repo src/main.py --output chunks.json
```

#### `kit chunk-symbols`

Split file content by code symbols (functions, classes) for semantic chunking.

```bash
kit chunk-symbols <repository-path> <file-path> [OPTIONS]
```

**Options:**

* `--output, -o <file>`: Save output to JSON file

**Examples:**

```bash
# Chunk by symbols (functions, classes)
kit chunk-symbols /path/to/repo src/main.py


# Export symbol-based chunks
kit chunk-symbols /path/to/repo src/api.py --output symbol-chunks.json
```

### Export Operations

#### `kit export`

Export repository data to structured JSON files for external tools and analysis.

```bash
kit export <repository-path> <data-type> <output-file> [OPTIONS]
```

**Data Types:**

* `index`: Complete repository index (files + symbols)
* `symbols`: All extracted symbols
* `file-tree`: Repository file structure
* `symbol-usages`: Usages of a specific symbol

**Options:**

* `--symbol <name>`: Symbol name (required for `symbol-usages`)
* `--symbol-type <type>`: Symbol type filter (for `symbol-usages`)

**Examples:**

```bash
# Export complete repository analysis
kit export /path/to/repo index complete-analysis.json


# Export only symbols
kit export /path/to/repo symbols symbols.json


# Export file structure
kit export /path/to/repo file-tree structure.json


# Export symbol usage analysis
kit export /path/to/repo symbol-usages api-usages.json --symbol "ApiClient"


# Export specific symbol type usage
kit export /path/to/repo symbol-usages class-usages.json --symbol "User" --symbol-type class
```

### Server Operations

#### `kit serve`

Run the kit REST API server for web integrations and remote access.

```bash
kit serve [OPTIONS]
```

**Options:**

* `--host <host>`: Server host (default: 0.0.0.0)
* `--port <port>`: Server port (default: 8000)
* `--reload/--no-reload`: Auto-reload on changes (default: True)

**Examples:**

```bash
# Start development server
kit serve


# Production configuration
kit serve --host 127.0.0.1 --port 9000 --no-reload


# Custom port for testing
kit serve --port 3000
```

### PR Review Operations

Note

**Want to build a custom PR reviewer?** This section covers kit’s production-ready PR reviewer. For a tutorial on building your own custom reviewer using kit’s components, see [Build an AI PR Reviewer](/tutorials/ai_pr_reviewer).

#### `kit review`

AI-powered GitHub pull request reviewer that provides comprehensive code analysis with full repository context. The reviewer clones repositories, analyzes symbol relationships, and generates intelligent reviews using Claude or GPT-4.

```bash
kit review <pr-url> [OPTIONS]
```

**Options:**

* `--plain, -p`: Output raw review content for piping (no formatting)
* `--dry-run, -n`: Generate review without posting to GitHub (shows formatted preview)
* `--model, -m <model>`: Override LLM model for this review
* `--config, -c <file>`: Use custom configuration file
* `--init-config`: Create default configuration file
* `--agentic`: Use multi-turn agentic analysis (higher cost, deeper analysis)
* `--agentic-turns <count>`: Number of analysis turns for agentic mode

**Examples:**

```bash
# Review and post comment
kit review https://github.com/owner/repo/pull/123


# Dry run (formatted preview without posting)
kit review --dry-run https://github.com/owner/repo/pull/123


# Clean output for piping to other tools
kit review --plain https://github.com/owner/repo/pull/123
kit review -p https://github.com/owner/repo/pull/123


# Override model for specific review
kit review --model gpt-4.1-nano https://github.com/owner/repo/pull/123
kit review -m claude-3-5-haiku-20241022 https://github.com/owner/repo/pull/123


# Pipe to Claude Code for implementation
kit review -p https://github.com/owner/repo/pull/123 | \
  claude "Implement these code review suggestions"


# Use agentic mode for complex PRs
kit review --agentic --agentic-turns 15 https://github.com/owner/repo/pull/123


# Initialize configuration
kit review --init-config
```

#### Quick Setup

**1. Install and configure:**

```bash
# Install kit
pip install cased-kit


# Set up configuration
kit review --init-config


# Set API keys
export KIT_GITHUB_TOKEN="ghp_your_token"
export KIT_ANTHROPIC_TOKEN="sk-ant-your_key"
```

**2. Review a PR:**

```bash
kit review https://github.com/owner/repo/pull/123
```

#### Configuration

The reviewer uses `~/.kit/review-config.yaml` for configuration:

```yaml
github:
  token: ghp_your_token_here
  base_url: https://api.github.com.com


llm:
  provider: anthropic  # or "openai"
  model: claude-sonnet-4-20250514
  api_key: sk-ant-your_key_here
  max_tokens: 4000
  temperature: 0.1


review:
  analysis_depth: standard  # quick, standard, thorough
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
  cache_directory: ~/.kit/repo-cache
  cache_ttl_hours: 24
```

#### Model Selection

**Frontier Tier ($15-75/MTok)**

* `claude-opus-4-20250514`: Latest flagship, world’s best coding model, superior complex reasoning
* `claude-sonnet-4-20250514`: High-performance with exceptional reasoning and efficiency

**Premium Tier ($3-15/MTok)**

* `claude-3-7-sonnet-20250219`: Extended thinking capabilities
* `claude-3-5-sonnet-20241022`: Proven excellent balance

**Balanced Tier ($0.80-4/MTok)**

* `gpt-4o-mini-2024-07-18`: Excellent value model
* `claude-3-5-haiku-20241022`: Fastest responses

#### Cache Management

```bash
# Check cache status
kit review-cache status


# Clean up old repositories
kit review-cache cleanup


# Clear all cached repositories
kit review-cache clear
```

#### Enterprise Usage

**Batch Review:**

```bash
# Review multiple PRs
for pr in 123 124 125; do
  kit review https://github.com/company/repo/pull/$pr
done
```

**CI/CD Integration:**

.github/workflows/pr-review\.yml

```yaml
name: AI PR Review
on:
  pull_request:
    types: [opened, synchronize]


jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - name: AI Review
        run: |
          pip install cased-kit
          kit review --dry-run ${{ github.event.pull_request.html_url }}
        env:
          KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

#### Cost Analysis

**Real-world costs by team size:**

*Small Team (20 PRs/month):*

* Standard mode: $0.20-1.00/month
* Mixed usage: $1.00-5.00/month

*Enterprise (500 PRs/month):*

* Standard mode: $5.00-25.00/month
* Mixed usage: $25.00-150.00/month

**Cost per PR by complexity:**

* Simple (1-2 files): $0.005-0.025
* Medium (3-5 files): $0.01-0.05
* Complex (6+ files): $0.025-0.10

#### Features

**Intelligent Analysis:**

* Repository cloning with caching for 5-10x faster repeat reviews
* Symbol extraction and cross-codebase impact analysis
* Security, architecture, and performance assessment
* Multi-language support for any language kit supports

**Cost Transparency:**

* Real-time cost tracking with exact LLM usage
* Token breakdown (input/output) for cost optimization
* Model information and pricing details

**Enterprise Features:**

* GitHub integration with classic and fine-grained tokens
* Multiple LLM provider support (Anthropic Claude, OpenAI GPT-4)
* Configurable analysis depth and review modes
* Repository caching and batch operations

#### Example Output

```markdown
## 🛠️ Kit AI Code Review


### Summary & Implementation
This PR introduces a new authentication middleware that validates JWT tokens...


### Code Quality Assessment
The implementation follows clean code principles with appropriate error handling...


### Cross-Codebase Impact Analysis
- **AuthMiddleware**: Used in 15 other places across the codebase
- **validateToken**: New function will be called by 8 existing routes
- Breaking change risk: Low (additive changes only)


### Security & Reliability
✅ Proper JWT validation with signature verification
⚠️ Consider adding rate limiting for failed authentication attempts


### Specific Issues & Recommendations
1. **Line 42 in auth.py**: Consider using constant-time comparison
2. **Line 67 in middleware.py**: Add input validation for token format


---
*Generated by kit v0.3.3 with claude-sonnet-4 analysis*
```

## Output Formats

### Human-Readable Formats

* **Table**: Structured columns for easy reading
* **Plain Text**: Simple text output for basic parsing
* **Icons**: File type indicators (📄 for files, 📁 for directories)

### Machine-Readable Formats

* **JSON**: Structured data perfect for further processing
* **Names**: Simple lists for Unix pipeline operations

### Piping & Integration

All commands work seamlessly with Unix tools:

```bash
# Count Python files
kit file-tree /path/to/repo | grep "\.py" | wc -l


# Find large functions (over 50 lines)
kit symbols /path/to/repo --format json | jq '.[] | select(.end_line - .start_line > 50)'


# Get unique function names
kit symbols /path/to/repo --format names | sort | uniq


# Find files with many symbols
kit symbols /path/to/repo --format json | jq -r '.[] | .file' | sort | uniq -c | sort -nr
```

## Practical Workflows

### Development Workflow with Caching

```bash
#!/bin/bash
REPO_PATH="/path/to/repo"


# First analysis (builds cache)
echo "Initial analysis..."
time kit symbols $REPO_PATH > /dev/null


# Subsequent analyses (use cache - much faster)
echo "Cached analysis..."
time kit symbols $REPO_PATH > /dev/null


# Check cache performance
kit cache stats $REPO_PATH
```

### Focused Directory Analysis

The new subpath and grep capabilities enable focused analysis of specific parts of large repositories:

```bash
#!/bin/bash
REPO_PATH="/path/to/repo"
FOCUS_DIR="src/api"


# Analyze specific directory structure
echo "📁 Analyzing directory structure for $FOCUS_DIR"
kit file-tree $REPO_PATH --path $FOCUS_DIR


# Find TODOs only in API code
echo "🔍 Finding TODOs in API code"
kit grep $REPO_PATH "TODO" --include "$FOCUS_DIR/*.py"


# Find security-related code
echo "🔒 Finding security patterns"
kit grep $REPO_PATH "auth\|token\|password" --include "$FOCUS_DIR/*" --ignore-case


# Export focused analysis
kit file-tree $REPO_PATH --path $FOCUS_DIR --output api-structure.json
kit grep $REPO_PATH "class.*API" --include "$FOCUS_DIR/*.py" --output api-classes.json
```

### Large Repository Optimization

For large repositories, use subpath analysis to work efficiently:

```bash
#!/bin/bash
REPO_PATH="/path/to/large-repo"


# Instead of analyzing the entire repository
# kit symbols $REPO_PATH  # This could be slow


# Focus on specific components
for component in "src/auth" "src/api" "src/models"; do
  echo "Analyzing $component..."


  # Get component structure
  kit file-tree $REPO_PATH --path "$component"


  # Find component-specific patterns
  kit grep $REPO_PATH "class\|def\|import" --include "$component/*.py" --max-results 100


  # Extract symbols from component files only
  # (Note: symbols command doesn't support subpath yet, but you can filter)
  kit symbols $REPO_PATH --format json | jq ".[] | select(.file | startswith(\"$component\"))"
done
```

### AI-Powered Development

```bash
#!/bin/bash
# Stage changes
git add .


# Generate intelligent commit message
kit commit --dry-run


# If satisfied, commit
kit commit


# Create PR and get AI summary
gh pr create --title "Feature: New auth system" --body "Initial implementation"
PR_URL=$(gh pr view --json url -q .url)
kit summarize --update-pr-body $PR_URL
```

### Code Review Preparation

```bash
#!/bin/bash
REPO_PATH="/path/to/repo"
OUTPUT_DIR="./analysis"


mkdir -p $OUTPUT_DIR


# Generate comprehensive analysis (uses caching)
kit export $REPO_PATH index $OUTPUT_DIR/repo-index.json


# Find all TODO items with grep (faster than search for literal strings)
kit grep $REPO_PATH "TODO\|FIXME\|XXX" --ignore-case --output $OUTPUT_DIR/todos.json


# Analyze specific areas that changed
CHANGED_DIRS=$(git diff --name-only HEAD~1 | cut -d'/' -f1 | sort -u)
for dir in $CHANGED_DIRS; do
  echo "Analyzing changed directory: $dir"
  kit file-tree $REPO_PATH --path "$dir" --output "$OUTPUT_DIR/${dir}-structure.json"
  kit grep $REPO_PATH "test\|spec" --include "$dir/*" --output "$OUTPUT_DIR/${dir}-tests.json"
done
```

### Documentation Generation

```bash
#!/bin/bash
REPO_PATH="/path/to/repo"
DOCS_DIR="./docs"


mkdir -p $DOCS_DIR


# Extract all public APIs (uses incremental analysis)
kit symbols $REPO_PATH --format json | \
  jq '.[] | select(.type=="function" and (.name | startswith("_") | not))' \
  > $DOCS_DIR/public-api.json


# Generate symbol usage reports
for symbol in $(kit symbols $REPO_PATH --format names | head -10); do
  kit usages $REPO_PATH "$symbol" --output "$DOCS_DIR/usage-$symbol.json"
done
```

### Migration Analysis

```bash
#!/bin/bash
OLD_REPO="/path/to/old/repo"
NEW_REPO="/path/to/new/repo"


# Compare symbol counts (both use caching)
echo "Old repo symbols:"
kit symbols $OLD_REPO --format names | wc -l


echo "New repo symbols:"
kit symbols $NEW_REPO --format names | wc -l


# Find deprecated patterns
kit search $OLD_REPO "deprecated\|legacy" --pattern "*.py" > deprecated-code.txt


# Export both for detailed comparison
kit export $OLD_REPO symbols old-symbols.json
kit export $NEW_REPO symbols new-symbols.json
```

### CI/CD Integration

.github/workflows/code-analysis.yml

```yaml
name: Code Analysis
on: [push, pull_request]


jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3


      - name: Install kit
        run: pip install cased-kit


      - name: Analyze codebase
        run: |
          # Generate repository analysis (uses incremental caching)
          kit export . index analysis.json


          # Check for complexity issues
          COMPLEX_FUNCTIONS=$(kit symbols . --format json | jq '[.[] | select(.type=="function" and (.end_line - .start_line) > 50)] | length')


          if [ $COMPLEX_FUNCTIONS -gt 10 ]; then
            echo "Warning: $COMPLEX_FUNCTIONS functions are longer than 50 lines"
          fi


          # Find security-related patterns
          kit search . "password\|secret\|key" --pattern "*.py" > security-review.txt


          # Show cache performance
          kit cache stats .


      - name: Upload analysis
        uses: actions/upload-artifact@v3
        with:
          name: code-analysis
          path: |
            analysis.json
            security-review.txt
```

## Best Practices

### Performance

* Use incremental analysis (`kit symbols`) for repeated operations - 25-36x faster
* Use `--format names` for large repositories when you only need symbol names
* Leverage file patterns (`--pattern`) to limit search scope
* Monitor cache performance with `kit cache stats`

### Caching

* Use `kit cache cleanup` periodically to remove stale entries
* Monitor cache size with `kit cache status`
* Clear cache (`kit cache clear`) if switching between very different git states frequently

### AI Workflows

* Use `--dry-run` to preview AI-generated content before posting
* Combine `kit commit` with `kit summarize` for complete AI-powered development workflow
* Use `--plain` output for piping to other AI tools

### Scripting

* Always check command exit codes (`$?`) in scripts
* Use `--output` to save data persistently rather than relying on stdout capture
* Combine with `jq`, `grep`, `sort`, and other Unix tools for powerful analysis

### Integration

* Export JSON data for integration with external tools and databases
* Use the CLI in CI/CD pipelines for automated code quality checks
* Combine with language servers and IDEs for enhanced development workflows

# Overview

## kit: Code Intelligence Toolkit

A modular, production-grade toolkit for codebase mapping, symbol extraction, code search, and LLM-powered developer workflows. Supports multi-language codebases via `tree-sitter`.

`kit` features a “mid-level API” to build your own custom tools, applications, agents, and workflows: easily build code review bots, semantic code search, documentation generators, and more.

`kit` is **free and open source** with a permissive MIT license. Check it out on [GitHub](https://github.com/cased/kit).

## Installation

### Install from PyPI

```bash
# Basic installation (includes PR reviewer, no ML dependencies)
pip install cased-kit


# With semantic search features (includes PyTorch, sentence-transformers)
pip install cased-kit[ml]


# Everything (all features)
pip install cased-kit[all]
```

### Install from Source

```bash
git clone https://github.com/cased/kit.git
cd kit
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

## Why Use kit?

`kit` helps with:

* **Unifying Code Access:** Provides a single, consistent `Repository` object to interact with files, symbols, and search across diverse codebases, regardless of language.
* **Deep Code Understanding:** Leverages `tree-sitter` for accurate, language-specific parsing, enabling reliable symbol extraction and structural analysis across an entire codebase.
* **Bridging Code and LLMs:** Offers tools specifically designed to chunk code effectively and retrieve relevant context for large language models, powering smarter AI developer tools.

## Core Philosophy

`kit` aims to be a **toolkit** for building applications, agents, and workflows. It handles the low-level parsing and indexing complexity, and allows you to adapt these components to your specific needs.

We believe the building blocks for code intelligence and LLM workflows for developer tools should be free and open source, so you can build amazing products and experiences.

## Where to Go Next

* **Dive into the API:** Explore the [Core Concepts](/core-concepts/repository-api) to understand the `Repository` object and its capabilities.
* **Build Something:** Follow the [Tutorials](/tutorials/ai_pr_reviewer) for step-by-step guides on creating practical tools.

## LLM Documentation

This documentation site provides generated text files suitable for LLM consumption:

* [`/llms.txt`](/llms.txt): Entrypoint file following the llms.txt standard.
* [`/llms-full.txt`](/llms-full.txt): Complete documentation content concatenated into a single file.
* [`/llms-small.txt`](/llms-small.txt): Minified documentation content for models with smaller context windows.

# Quickstart

```bash
git clone https://github.com/cased/kit.git
cd kit
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Now, you can use kit! kit ships with a demonstration repository at `tests/fixtures/` you can use to get started.

Try this simple Python script (e.g., save as `test_kit.py` in the `kit` directory you cloned):

```python
import kit
import os


# Path to the demo repository
repo_path = "tests/fixtures/realistic_repo"


print(f"Loading repository at: {repo_path}")
# Ensure you have cloned the 'kit' repository and are in its root directory
# for this relative path to work correctly.
repo = kit.Repository(repo_path)


# Print the first 5 Python files found in the demo repo
print("\nFound Python files in the demo repo (first 5):")
count = 0
for file in repo.files('*.py'):
    print(f"- {file.path}")
    count += 1
    if count >= 5:
        break


if count == 0:
    print("No Python files found in the demo repository.")


# Extract symbols from a specific file in the demo repo (e.g., app.py)
target_file = 'app.py'
print(f"\nExtracting symbols from {target_file} in the demo repo (first 5):")
try:
    symbols = repo.extract_symbols(target_file)
    if symbols:
        for i, symbol in enumerate(symbols):
            print(f"- {symbol.name} ({symbol.kind}) at line {symbol.range.start.line}")
            if i >= 4:
                break
    else:
        print(f"No symbols found or file not parseable: {target_file}")
except FileNotFoundError:
    print(f"File not found: {target_file}")
except Exception as e:
    print(f"An error occurred extracting symbols: {e}")
```

Run it with `python test_kit.py`.

Next, explore the [Usage Guide](/introduction/usage-guide) to understand the core concepts.

# Usage Guide

This guide provides practical examples of how to use the core `Repository` object in `kit` to interact with your codebase.

## Initializing a `Repository`

First, create an instance of the `Repository` class, pointing it to your code. `kit` can work with local directories or clone remote Git repositories. This is the starting point for any analysis, giving `kit` access to the codebase.

### Local Directory

If your code is already on your machine:

```python
from kit import Repository


repo = Repository("/path/to/your/local/project")
```

### Remote Git Repository

`kit` can clone a public or private Git repository. For private repos, provide a GitHub token.

```python
# Public repo
repo = Repository("https://github.com/owner/repo-name")


# Private repo (requires token)
# Ensure the token has appropriate permissions
github_token = "your_github_pat_here"
repo = Repository("https://github.com/owner/private-repo-name", github_token=github_token)
```

### Caching

When cloning remote repositories, `kit` caches them locally to speed up subsequent initializations. By default, caches are stored in a temporary directory. You can specify a persistent cache directory:

```python
repo = Repository(
    "https://github.com/owner/repo-name",
    cache_dir="/path/to/persistent/cache"
)
```

## Basic Exploration

Once initialized, you can explore the codebase. Use these methods to get a high-level overview of the repository’s structure and key code elements, or to gather foundational context for an LLM.

### Getting the File Tree

List all files and directories:

```python
file_tree = repo.get_file_tree()
# Returns a list of dicts: [{'path': '...', 'is_dir': False, ...}, ...]
```

### Extracting Symbols

Identify functions, classes, etc., across the whole repo or in a specific file:

```python
# All symbols
all_symbols = repo.extract_symbols()


# Symbols in a specific file
specific_symbols = repo.extract_symbols("src/my_module.py")
# Returns a list of dicts: [{'name': '...', 'type': 'function', ...}, ...]
```

### Searching Text

Perform simple text or regex searches:

```python
matches = repo.search_text("my_function_call", file_pattern="*.py")
# Returns a list of dicts: [{'file': '...', 'line_number': 10, ...}, ...]
```

## Preparing Code for LLMs

`kit` provides utilities to prepare code snippets for large language models. These methods help break down large codebases into manageable pieces suitable for LLM context windows or specific analysis tasks.

### Chunking

Split files into manageable chunks, either by line count or by symbol definition:

```python
# Chunk by lines
line_chunks = repo.chunk_file_by_lines("src/long_file.py", max_lines=100)


# Chunk by symbols (functions, classes)
symbol_chunks = repo.chunk_file_by_symbols("src/long_file.py")
```

### Extracting Context

Get the specific function or class definition surrounding a given line number:

```python
context = repo.extract_context_around_line("src/my_module.py", line=42)
# Returns a dict like {'name': 'my_function', 'type': 'function', 'code': 'def my_function(...): ...'}
```

## Generating Code Summaries (Alpha)

`kit` includes an alpha feature for generating natural language summaries (like dynamic docstrings) for code elements (files, functions, classes) using a configured Large Language Model (LLM). This can be useful for:

* Quickly understanding the purpose of a piece of code.
* Providing context to other LLM-powered tools.
* Powering semantic search based on generated summaries rather than just raw code.

**Note:** This feature is currently in **alpha**. The API may change, and it requires an LLM (e.g., via OpenAI, Anthropic) to be configured for `kit` to use for summarization.

### Using the `DocstringIndexer`

The `DocstringIndexer` is responsible for managing the summarization process and storing/retrieving these generated “docstrings.”

```python
from kit import Repository
from kit.docstring_indexer import DocstringIndexer
from kit.summaries import Summarizer, OpenAIConfig
from sentence_transformers import SentenceTransformer  # or any embedder of your choice


# 1. Initialize your Repository
repo = Repository("tests/fixtures/realistic_repo")  # Or your project path


# 2. Configure the LLM-powered summarizer
# Make sure the relevant API key (e.g., OPENAI_API_KEY) is set in your environment
summarizer = Summarizer(repo, OpenAIConfig(model="gpt-4o"))


# 3. Provide an embedding function (str -> list[float]) for the vector index
st_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda text: st_model.encode(text).tolist()


# 4. Create the DocstringIndexer
#    You can specify where on disk to persist the vector DB via `persist_dir`.
indexer = DocstringIndexer(
    repo,
    summarizer,
    embed_fn,
    persist_dir="kit_docstring_cache",
)


# 5. Build the index (generates summaries for new/changed files/symbols)
#    This may take some time depending on repository size and LLM speed.
indexer.build(force=True)  # `level="symbol"` by default


# 6. Retrieve a summary – use the built-in SummarySearcher
searcher = indexer.get_searcher()
hits = searcher.search("utils.greet", top_k=1)  # Search by symbol or natural language
if hits:
    print("Summary:", hits[0]["summary"])
else:
    print("No summary found (yet).")
```

This generated summary can then be used for various purposes, including enhancing semantic search or providing contextual information for code generation tasks. Refer to the [Core Concepts: Docstring Indexing](/core-concepts/docstring-indexing) page for more details on configuration and advanced usage.

## Semantic Code Search

Perform vector-based semantic search (requires configuration). Go beyond keyword search to find code related by meaning or concept, useful for discovery and understanding.

```python
# NOTE: Requires prior setup - see Core Concepts > Configuring Semantic Search
results = repo.search_semantic("find code related to database connections", top_k=3)
```

## Finding Symbol Usages

Locate all definitions and references of a specific symbol: Track down where functions or classes are defined and used throughout the codebase for impact analysis or refactoring.

```python
usages = repo.find_symbol_usages("MyClass", symbol_type="class")
# Returns a list of dicts showing definitions and text matches across the repo.
```

## Exporting Data

`kit` can export the gathered information (file tree, symbols, index, usages) to JSON files for use in other tools or offline analysis. Persist the results of your analysis or integrate `kit`’s findings into other development workflows.

```python
# Export the full index (files + symbols)
repo.write_index("repo_index.json")


# Export only symbols
repo.write_symbols("symbols.json")


# Export file tree
repo.write_file_tree("file_tree.json")


# Export usages of a symbol
repo.write_symbol_usages("MyClass", "my_class_usages.json", symbol_type="class")
```

# Using kit with MCP

> Learn how to use kit with the Model Context Protocol (MCP) for AI-powered code understanding

Note: MCP support is currently in alpha.

The Model Context Protocol (MCP) provides a unified API for codebase operations, making it easy to integrate kit’s capabilities with AI tools and IDEs. This guide will help you set up and use kit with MCP.

Kit provides a MCP server implementation that exposes its code intelligence capabilities through a standardized protocol. When using kit as an MCP server, you gain access to:

* **Code Search**: Perform text-based and semantic code searches
* **Code Analysis**: Extract symbols, find symbol usages, and analyze dependencies
* **Code Summarization**: Create natural language summaries of code
* **File Navigation**: Explore file trees and repository structure

This document guides you through setting up and using `kit` with MCP-compatible tools like Cursor or Claude Desktop.

## What is MCP?

MCP (Model Context Protocol) is a specification that allows AI agents and development tools to interact with your codebase programmatically via a local server. `kit` implements an MCP server to expose its code intelligence features.

## Available MCP Tools in `kit`

Currently, `kit` exposes the following functionalities via MCP tools:

* `open_repository`: Opens a local or remote Git repository. Supports `ref` parameter for specific commits, tags, or branches.
* `get_file_tree`: Retrieves the file and directory structure of the open repository.
* `get_file_content`: Reads the content of a specific file.
* `search_code`: Performs text-based search across repository files.
* `grep_code`: Fast literal string search with directory filtering and smart exclusions.
* `extract_symbols`: Extracts functions, classes, and other symbols from a file.
* `find_symbol_usages`: Finds where a specific symbol is used across the repository.
* `get_code_summary`: Provides AI-generated summaries for files, functions, or classes.
* `get_git_info`: Retrieves git metadata including current SHA, branch, and remote URL.

### Opening Repositories with Specific Versions

The `open_repository` tool supports analyzing specific versions of repositories using the optional `ref` parameter:

```json
{
  "tool": "open_repository",
  "arguments": {
    "path_or_url": "https://github.com/owner/repo",
    "ref": "v1.2.3"
  }
}
```

The `ref` parameter accepts:

* **Commit SHAs**: `"abc123def456"`
* **Tags**: `"v1.2.3"`, `"release-2024"`
* **Branches**: `"main"`, `"develop"`, `"feature-branch"`

### Accessing Git Metadata

Use the `get_git_info` tool to access repository metadata:

```json
{
  "tool": "get_git_info",
  "arguments": {
    "repo_id": "your-repo-id"
  }
}
```

This returns information like current commit SHA, branch name, and remote URL - useful for understanding what version of code you’re analyzing.

### Automatic GitHub Token Handling

For convenience when working with private repositories, the MCP server automatically checks for GitHub tokens in environment variables:

1. First checks `KIT_GITHUB_TOKEN`
2. Falls back to `GITHUB_TOKEN` if `KIT_GITHUB_TOKEN` is not set
3. Uses no authentication if neither environment variable is set

This means you can set up your MCP client with:

```json
{
  "mcpServers": {
    "kit-mcp": {
      "command": "python",
      "args": ["-m", "kit.mcp"],
      "env": {
        "KIT_GITHUB_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

And then simply open private repositories without needing to specify the `github_token` parameter:

```json
{
  "tool": "open_repository",
  "arguments": {
    "path_or_url": "https://github.com/your-org/private-repo"
  }
}
```

More MCP features are coming soon.

## Setup

1. After installing `kit`, configure your MCP-compatible client by adding a stanza like this to your settings:

Available environment variables for the `env` section:

* `OPENAI_API_KEY`
* `KIT_MCP_LOG_LEVEL`
* `KIT_GITHUB_TOKEN` - Automatically used for private repository access
* `GITHUB_TOKEN` - Fallback for private repository access

```json
{
  "mcpServers": {
    "kit-mcp": {
      "command": "python",
      "args": ["-m", "kit.mcp"],
      "env": {
        "KIT_MCP_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

The `python` executable invoked must be the one where `cased-kit` is installed. If you see `ModuleNotFoundError: No module named 'kit'`, ensure the Python interpreter your MCP client is using is the correct one.

# PR Reviews & Summaries

> Production-ready AI-powered code reviewer and PR summarization with full repository context, transparent pricing, and CI/CD integration

# Kit AI PR Reviewer & Summarizer

Kit includes a **production-ready AI PR reviewer and summarizer** that provides professional-grade code analysis with full repository context. Use almost any LLM and pay just for tokens. High-quality reviews with SOTA models like Claude Sonnet 4 generally cost about 10 cents, while summaries cost just pennies.

Use per-organization profiles and prioritization for further customization. Use kit’s local output to pipe to other unix tools.

Tip

**Want to build a completely custom reviewer?** See the [Build an AI PR Reviewer tutorial](/tutorials/ai_pr_reviewer) to create your own using kit’s components.

## 🚀 Quick Start

```bash
# 1. Install kit (lightweight - no ML dependencies needed for PR review!)
pip install cased-kit


# 2. Set up configuration
kit review --init-config


# 3. Set API keys
export KIT_GITHUB_TOKEN="ghp_your_token"
export KIT_ANTHROPIC_TOKEN="sk-ant-your_key"
export KIT_OPENAI_TOKEN="sk-openai-your_key"
export KIT_GOOGLE_API_KEY="AIzaSy-your_google_key"


# 4. Review any GitHub PR
kit review https://github.com/owner/repo/pull/123


# 5. Test without posting (dry run with full formatting)
kit review --dry-run https://github.com/owner/repo/pull/123


# 6. Use custom context profiles for organization standards
kit review --profile company-standards https://github.com/owner/repo/pull/123


# 7. Focus on specific priority levels
kit review --priority=high,medium https://github.com/owner/repo/pull/123


# 8. Quick PR summaries for triage (5-10x cheaper)
kit summarize https://github.com/owner/repo/pull/123


# 9. Add summary to PR description for team visibility
kit summarize --update-pr-body https://github.com/owner/repo/pull/123
```

Tip

**Just want the PR reviewer?** The base `pip install cased-kit` gives you everything needed for PR reviews without heavy ML dependencies like PyTorch. If you need semantic search features later, install with `pip install cased-kit[ml]`.

## 💰 Transparent Pricing

Some examples based on real-world testing on production open source PRs:

| Model                   | Typical Cost      | Quality | Best For                                       |
| ----------------------- | ----------------- | ------- | ---------------------------------------------- |
| **gemini-1.5-flash-8b** | **$0.003**        | ⭐⭐⭐     | Ultra-budget, high volume                      |
| **gpt-4.1-nano**        | **$0.0015-0.004** | ⭐⭐⭐     | High-volume, ultra-budget                      |
| **gpt-4.1-mini**        | **$0.005-0.015**  | ⭐⭐⭐⭐    | Budget-friendly, often very good for the price |
| **gemini-2.5-flash**    | **$0.007**        | ⭐⭐⭐⭐    | Excellent value, fast                          |
| **claude-sonnet-4**     | **0.08-$0.14**    | ⭐⭐⭐⭐⭐   | **Recommended for most**                       |

**PR Summaries** (for triage): \~$0.005-0.02 per summary (5-10x cheaper than reviews)

### In Practice

Even without optimizing your model mix, a team doing 500 large PRs a month will generally pay under $50 a month total for reviews with SOTA models.

Tip

Don’t underestimate the smaller models. gpt-4.1-mini delivers surprisingly useful reviews *when given the right context via kit*. For simple projects, you can get decent AI code reviews for **less than $1/month**. Here’s an example [against kit itself](https://github.com/cased/kit/pull/56#issuecomment-2928399599). This review cost *half a cent*.

## 🎯 Key Features

### Intelligent Analysis

* **Repository Context**: Full codebase understanding, not just diff analysis
* **Symbol Analysis**: Identifies when functions/classes are used elsewhere
* **Cross-Impact Assessment**: Understands how changes affect the broader system
* **Multi-Language Support**: Works with any language kit supports

### Professional Output

* **Priority-Based Issues**: High/Medium/Low issue categorization with filtering options
* **Specific Recommendations**: Concrete code suggestions with examples
* **GitHub Integration**: Clickable links to all referenced files
* **Quality Scoring**: Objective metrics for review effectiveness

### Cost & Transparency

* **Real-Time Cost Tracking**: See exact LLM usage and costs
* **Token Breakdown**: Understand what drives costs
* **Model Information**: Know which AI provided the analysis
* **No Hidden Fees**: Pay only for actual LLM usage

## 📄 PR Summaries for Quick Triage

For teams that need to **quickly understand PRs before committing to full reviews**, kit includes fast, cost-effective PR summarization:

```bash
# Generate a quick summary
kit summarize https://github.com/owner/repo/pull/123


# Add the summary directly to the PR description
kit summarize --update-pr-body https://github.com/owner/repo/pull/123


# Use budget models for ultra-low-cost summaries
kit summarize --model gpt-4.1-nano https://github.com/owner/repo/pull/123
```

### Why Use PR Summaries?

* **5-10x cheaper** than full reviews (\~$0.005-0.02 vs $0.01-0.05+)
* **Perfect for triage**: Understand what a PR does before deciding on detailed review
* **Team visibility**: Add AI summaries directly to PR descriptions for everyone to see
* **Same repository intelligence**: Leverages symbol extraction and dependency analysis

### Summary Output Format

Summaries provide structured information in a consistent format:

* **What This PR Does**: 2-3 sentence overview of the main purpose
* **Key Changes**: Most important modifications (max 5 bullet points)
* **Impact**: Areas of codebase affected and potential risks/benefits

### PR Body Updates

The `--update-pr-body` option adds a marked AI summary section to the PR description:

```markdown
<!-- AI SUMMARY START -->


## What This PR Does
[AI-generated overview]


## Key Changes
- [Key modifications]


## Impact
- [Impact analysis]


*Generated by kit v0.7.1 • Model: claude-sonnet-4-20250514*
<!-- AI SUMMARY END -->
```

**Smart handling**: Re-running with `--update-pr-body` replaces the existing summary instead of duplicating it.

## 📋 Custom Context Profiles

Store and apply **organization-specific coding standards and review guidelines** through custom context profiles. Create profiles that automatically inject your company’s coding standards, security requirements, and style guidelines into every PR review.

```bash
# Create a profile from your existing coding guidelines
kit review-profile create --name company-standards \
  --file coding-guidelines.md \
  --description "Acme Corp coding standards"


# Use in any review
kit review --profile company-standards https://github.com/owner/repo/pull/123


# List all profiles
kit review-profile list
```

→ **[Complete Profiles Guide](/pr-reviewer/profiles)** - Profile management, team workflows, and examples

## 🔄 Output Modes & Integration

Kit provides different output modes for various workflows - from direct GitHub posting to piping output to CLI code writers:

```bash
# Standard mode - posts directly to GitHub
kit review https://github.com/owner/repo/pull/123


# Plain mode - clean output for piping to other tools
kit review --plain https://github.com/owner/repo/pull/123 | \
  claude "implement these suggestions"


# Priority filtering - focus on what matters
kit review --priority=high,medium https://github.com/owner/repo/pull/123
```

→ **[Integration Guide](/pr-reviewer/integration)** - Output modes, piping workflows, and multi-stage AI analysis

## 🚀 CI/CD Integration

Add AI code reviews to your GitHub Actions workflow:

```yaml
name: AI PR Review
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: AI Code Review
        run: |
          pip install cased-kit
          kit review ${{ github.event.pull_request.html_url }}
        env:
          KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

→ **[CI/CD Guide](/pr-reviewer/cicd)** - GitHub Actions, advanced workflows, and cost optimization strategies

## 🔧 Configuration

Quick configuration for common setups:

```bash
# Override model for specific review
kit review --model gpt-4.1-nano https://github.com/owner/repo/pull/123


# Free local AI with Ollama
kit review --model qwen2.5-coder:latest https://github.com/owner/repo/pull/123
```

→ **[Configuration Guide](/pr-reviewer/configuration)** - Model selection, API keys, and configuration files

## 📊 Examples

See real-world reviews with actual costs and analysis:

* **[FastAPI Packaging Change](https://github.com/cased/kit/blob/main/src/kit/pr_review/example_reviews/fastapi_11935_standard_dependencies.md)** ($0.034) - Architectural impact analysis
* **[React.dev UI Feature](https://github.com/cased/kit/blob/main/src/kit/pr_review/example_reviews/react_dev_6986_branding_menu.md)** ($0.012) - Accessibility-focused review
* **[Documentation Fix](https://github.com/cased/kit/blob/main/src/kit/pr_review/example_reviews/biopython_204_documentation_fix.md)** ($0.006) - Proportional response

→ **[More Examples](/pr-reviewer/examples)** - Real review examples and use cases

## 📈 What’s Next: Roadmap

#### Recently Shipped ✅

* **Custom Context Profiles**: Store and apply organization-specific coding standards and guidelines
* **Priority Filtering**: Focus reviews on what matters most

#### In Development

* **Feedback Learning**: Simple database to learn from review feedback and improve over time
* **Inline Comments**: Post comments directly on specific lines instead of summary comments
* **Follow-up Review Awareness**: Take previous reviews into account for better, more targeted feedback

#### Future Features

* **Multi-Model Consensus**: Compare reviews from multiple models for high-stakes changes
* **Smart Review Routing**: Automatically select the best model based on change type and team preferences

## 💡 Best Practices

### Cost Optimization

* **Use free local AI** for unlimited reviews with Ollama (requires self-hosted setup)
* Use budget models for routine changes, premium for breaking changes
* Use the `--model` flag to override models per PR
* Leverage caching - repeat reviews of same repo are 5-10x faster
* Set up profiles to avoid redundant context

### Team Adoption

* **Start with free local AI** to build confidence without costs
* Use budget models initially to control costs
* Create organization-specific guidelines for consistent reviews
* Add to CI/CD for all PRs or just high-impact branches

***

The kit AI PR reviewer provides **professional-grade code analysis** at costs accessible to any team size, from **$0.00/month with free local AI** to enterprise-scale deployment. With full repository context and transparent pricing, it’s designed to enhance your development workflow without breaking the budget.

# CI/CD Integration

> GitHub Actions workflows, advanced automation patterns, and cost optimization strategies for AI code reviews

# CI/CD Integration

Integrate AI code reviews seamlessly into your development workflow with GitHub Actions and other CI/CD platforms.

## Basic GitHub Actions

### Simple AI Review

Create `.github/workflows/pr-review.yml`:

```yaml
name: AI PR Review
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: AI Code Review
        run: |
          pip install cased-kit
          kit review ${{ github.event.pull_request.html_url }}
        env:
          KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

### With Custom Context Profiles

```yaml
name: AI PR Review with Company Standards
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: AI Code Review
        run: |
          pip install cased-kit
          kit review --profile company-standards ${{ github.event.pull_request.html_url }}
        env:
          KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Advanced Workflows

### Free Local AI Setup

For teams using self-hosted runners with Ollama:

```yaml
name: Free AI Review with Ollama
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review:
    runs-on: self-hosted  # Requires self-hosted runner with Ollama installed
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: AI Code Review
        run: |
          pip install cased-kit
          # Use completely free local AI
          kit review --model qwen2.5-coder:latest ${{ github.event.pull_request.html_url }}
        env:
          KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          # No LLM API keys needed - Ollama is free!
```

### Budget-Conscious Setup

Ultra-low cost with GPT-4.1-nano:

```yaml
name: Budget AI Review
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: Budget AI Review
        run: |
          pip install cased-kit
          # Configure for ultra-low cost
          kit review --model gpt-4.1-nano ${{ github.event.pull_request.html_url }}
        env:
          KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_OPENAI_TOKEN: ${{ secrets.OPENAI_API_KEY }}
```

### Smart Model Selection

Choose models based on PR size and complexity:

```yaml
name: Smart Model Selection
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: Smart Model Selection
        run: |
          pip install cased-kit


          # Use budget model for small PRs, premium for large ones
          FILES_CHANGED=$(gh pr view ${{ github.event.pull_request.number }} --json files --jq '.files | length')


          if [ "$FILES_CHANGED" -gt 20 ]; then
            MODEL="claude-sonnet-4-20250514"
            echo "🏗️ Large PR detected ($FILES_CHANGED files) - using premium model"
          elif [ "$FILES_CHANGED" -gt 5 ]; then
            MODEL="gpt-4.1"
            echo "📝 Medium PR detected ($FILES_CHANGED files) - using standard model"
          else
            MODEL="gpt-4.1-nano"
            echo "🔍 Small PR detected ($FILES_CHANGED files) - using budget model"
          fi


          kit review --model "$MODEL" ${{ github.event.pull_request.html_url }}
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
          KIT_OPENAI_TOKEN: ${{ secrets.OPENAI_API_KEY }}
```

## Conditional Reviews

### Skip Bot PRs and Drafts

```yaml
name: AI PR Review
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review:
    # Only review non-draft PRs from humans
    if: "!github.event.pull_request.draft && !contains(github.event.pull_request.user.login, 'bot')"
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: AI Code Review
        run: |
          pip install cased-kit
          kit review ${{ github.event.pull_request.html_url }}
        env:
          KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

### Label-Based Reviews

```yaml
name: Label-Based Reviews
on:
  pull_request:
    types: [opened, synchronize, reopened, labeled]


jobs:
  security-review:
    if: contains(github.event.pull_request.labels.*.name, 'security')
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: Security-Focused Review
        run: |
          pip install cased-kit
          kit review --profile security-standards --priority=high ${{ github.event.pull_request.html_url }}
        env:
          KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}


  breaking-change-review:
    if: contains(github.event.pull_request.labels.*.name, 'breaking-change')
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: Premium Review for Breaking Changes
        run: |
          pip install cased-kit
          kit review --model claude-opus-4-20250514 ${{ github.event.pull_request.html_url }}
        env:
          KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Priority-Based Workflows

### Priority Filtering by Branch

```yaml
name: Priority-Based Review
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: Smart Priority-Based Review
        run: |
          pip install cased-kit


          # Use high priority for main branch, all priorities for feature branches
          if [ "${{ github.event.pull_request.base.ref }}" == "main" ]; then
            PRIORITY="high,medium"
            echo "🎯 Main branch target - focusing on critical issues"
          else
            PRIORITY="high,medium,low"
            echo "🌿 Feature branch - comprehensive review"
          fi


          kit review --priority="$PRIORITY" ${{ github.event.pull_request.html_url }}
        env:
          KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

### Cost-Optimized Two-Stage Process

```yaml
name: Two-Stage Review Process
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: Two-Stage Review Process
        run: |
          pip install cased-kit


          # Stage 1: Quick high-priority scan with budget model
          HIGH_ISSUES=$(kit review -p --model gpt-4o-mini --priority=high ${{ github.event.pull_request.html_url }})


          # Stage 2: If critical issues found, do full review with premium model
          if echo "$HIGH_ISSUES" | grep -q "High Priority"; then
            echo "🚨 Critical issues detected - running comprehensive review"
            kit review --model claude-sonnet-4 ${{ github.event.pull_request.html_url }}
          else
            echo "✅ No critical issues found - posting quick scan results"
            echo "$HIGH_ISSUES" | gh pr comment ${{ github.event.pull_request.number }} --body-file -
          fi
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
          KIT_OPENAI_TOKEN: ${{ secrets.OPENAI_API_KEY }}
```

## Multi-Stage Processing

### Review with Implementation

```yaml
name: AI Review with Implementation
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review-and-process:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: AI Review with Multi-Stage Processing
        run: |
          pip install cased-kit


          # Stage 1: Generate review with kit's repository intelligence
          REVIEW=$(kit review -p --model claude-3-5-haiku-20241022 ${{ github.event.pull_request.html_url }})


          # Stage 2: Extract action items and post as separate comment
          echo "$REVIEW" | python scripts/extract-action-items.py | \
            gh pr comment ${{ github.event.pull_request.number }} --body-file -


          # Stage 3: Save review for later processing
          echo "$REVIEW" > review-${{ github.event.pull_request.number }}.md


          # Stage 4: Send to team notification system
          echo "$REVIEW" | python scripts/notify-team.py --channel engineering


          # Stage 5: Update metrics dashboard
          python scripts/update-metrics.py --pr ${{ github.event.pull_request.number }} --review "$REVIEW"
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

### External Tool Integration

```yaml
name: Review and Process
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review-integration:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: Checkout
        uses: actions/checkout@v4


      - name: Review and Process
        run: |
          pip install cased-kit


          # Get clean review output for processing
          kit review -p ${{ github.event.pull_request.html_url }} > raw-review.txt


          # Parse with custom tools
          python scripts/extract-security-issues.py raw-review.txt > security-issues.md
          python scripts/update-team-dashboard.py raw-review.txt
          python scripts/generate-metrics.py raw-review.txt > metrics.json


          # Post processed results back to PR
          if [ -s security-issues.md ]; then
            echo "## 🔒 Security Issues Detected" > processed-summary.md
            cat security-issues.md >> processed-summary.md
            gh pr comment ${{ github.event.pull_request.number }} --body-file processed-summary.md
          fi
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Smart Profile Selection

### File-Type Based Profiles

```yaml
name: Smart Profile Selection
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: Smart Profile Selection
        run: |
          pip install cased-kit


          # Check what type of files changed
          CHANGED_FILES=$(gh pr view ${{ github.event.pull_request.number }} --json files --jq -r '.files[].filename')


          if echo "$CHANGED_FILES" | grep -q "\.py$"; then
            PROFILE="python-backend"
            echo "🐍 Python files detected - using backend profile"
          elif echo "$CHANGED_FILES" | grep -q "\.(ts|tsx|js|jsx)$"; then
            PROFILE="frontend-react"
            echo "⚛️ React files detected - using frontend profile"
          elif echo "$CHANGED_FILES" | grep -q "security\|auth"; then
            PROFILE="security-focused"
            echo "🔒 Security-related files - using security profile"
          elif echo "$CHANGED_FILES" | grep -q "Dockerfile\|docker-compose\|\.yml$"; then
            PROFILE="infrastructure"
            echo "🏗️ Infrastructure files - using DevOps profile"
          else
            PROFILE="general-standards"
            echo "📋 General changes - using standard profile"
          fi


          kit review --profile "$PROFILE" ${{ github.event.pull_request.html_url }}
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

### Repository Optimization Workflows

Use `--repo-path` in CI/CD environments to optimize performance and handle special cases:

#### Pre-Cloned Repository Workflow

```yaml
name: Optimized Review with Pre-Cloned Repo
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for comprehensive analysis


      - name: AI Review with Local Repository
        run: |
          pip install cased-kit


          # Use the checked-out repository directly
          kit review --repo-path . ${{ github.event.pull_request.html_url }}
        env:
          KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

Tip

**Performance Benefit**: Using `--repo-path .` with a pre-checkout saves 30-90 seconds per review by skipping the clone step, especially valuable for large repositories or high-volume workflows.

## Cost Monitoring

### Review Cost Tracking

```yaml
name: AI Review with Cost Tracking
on:
  pull_request:
    types: [opened, synchronize, reopened]


jobs:
  ai-review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read


    steps:
      - name: AI Review with Cost Tracking
        run: |
          pip install cased-kit


          # Run review and capture cost information
          kit review --dry-run ${{ github.event.pull_request.html_url }} > review-output.txt


          # Extract cost information
          COST=$(grep "Total cost:" review-output.txt | awk '{print $3}')
          MODEL=$(grep "Model:" review-output.txt | awk '{print $2}')


          # Post actual review
          kit review ${{ github.event.pull_request.html_url }}


          # Log cost for monitoring
          echo "PR ${{ github.event.pull_request.number }}: $COST ($MODEL)" >> /tmp/review-costs.log


          # Alert if cost is unusually high
          if [ "$(echo "$COST > 0.50" | bc)" -eq 1 ]; then
            echo "⚠️ High review cost detected: $COST" >> $GITHUB_STEP_SUMMARY
          fi
        env:
          KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Platform-Specific Examples

### GitLab CI

.gitlab-ci.yml

```yaml
ai-review:
  stage: review
  image: python:3.9
  only:
    - merge_requests
  script:
    - pip install cased-kit
    - kit review --profile company-standards "$CI_MERGE_REQUEST_PROJECT_URL/-/merge_requests/$CI_MERGE_REQUEST_IID"
  variables:
    KIT_GITHUB_TOKEN: $GITLAB_TOKEN
    KIT_ANTHROPIC_TOKEN: $ANTHROPIC_API_KEY
```

### Azure DevOps

azure-pipelines.yml

```yaml
trigger:
  - none


pr:
  - main
  - develop


pool:
  vmImage: 'ubuntu-latest'


steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.9'


- script: |
    pip install cased-kit
    kit review --profile company-standards "$(System.PullRequest.SourceRepositoryURI)/pull/$(System.PullRequest.PullRequestNumber)"
  env:
    KIT_GITHUB_TOKEN: $(GitHubToken)
    KIT_ANTHROPIC_TOKEN: $(AnthropicToken)
  displayName: 'AI Code Review'
```

## Best Practices

### Error Handling

```yaml
- name: Robust AI Review
  run: |
    pip install cased-kit


    # Set error handling
    set +e  # Don't exit on error


    # Attempt review with timeout
    timeout 300 kit review ${{ github.event.pull_request.html_url }}
    EXIT_CODE=$?


    if [ $EXIT_CODE -eq 0 ]; then
      echo "✅ Review completed successfully"
    elif [ $EXIT_CODE -eq 124 ]; then
      echo "⏰ Review timed out after 5 minutes"
      gh pr comment ${{ github.event.pull_request.number }} --body "⏰ AI review timed out - PR may be too large for automated analysis"
    else
      echo "❌ Review failed with exit code $EXIT_CODE"
      gh pr comment ${{ github.event.pull_request.number }} --body "❌ AI review encountered an error - please check configuration"
    fi
  env:
    KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

### Resource Management

```yaml
- name: Resource-Efficient Review
  run: |
    pip install cased-kit


    # Check PR size before review
    FILES_CHANGED=$(gh pr view ${{ github.event.pull_request.number }} --json files --jq '.files | length')
    LINES_CHANGED=$(gh pr view ${{ github.event.pull_request.number }} --json additions,deletions --jq '.additions + .deletions')


    if [ "$FILES_CHANGED" -gt 100 ] || [ "$LINES_CHANGED" -gt 10000 ]; then
      echo "📊 Large PR detected ($FILES_CHANGED files, $LINES_CHANGED lines)"
      echo "Using focused review to manage costs"
      kit review --priority=high,medium --model gpt-4.1-mini ${{ github.event.pull_request.html_url }}
    else
      echo "📝 Standard PR size - full review"
      kit review --profile company-standards ${{ github.event.pull_request.html_url }}
    fi
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

### Notification Integration

```yaml
- name: Review with Notifications
  run: |
    pip install cased-kit


    # Run review and capture result
    if kit review --profile company-standards ${{ github.event.pull_request.html_url }}; then
      # Success notification
      curl -X POST "$SLACK_WEBHOOK" \
        -H 'Content-type: application/json' \
        --data '{
          "text": "✅ AI review completed for PR #${{ github.event.pull_request.number }}",
          "channel": "#code-reviews"
        }'
    else
      # Error notification
      curl -X POST "$SLACK_WEBHOOK" \
        -H 'Content-type: application/json' \
        --data '{
          "text": "❌ AI review failed for PR #${{ github.event.pull_request.number }}",
          "channel": "#engineering-alerts"
        }'
    fi
  env:
    KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
    SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
```

Tip

**Pro Tip**: Start with basic workflows and gradually add sophistication. Monitor costs and adjust model selection based on your team’s needs and budget.

***

[← Back to PR Reviewer Overview](/pr-reviewer/)

# Configuration

> Model selection, API keys, configuration files, and advanced setup options for AI code reviews

# Configuration

Configure kit’s AI PR reviewer for your team’s needs with flexible model selection, API key management, and configuration options.

## Model Override via CLI

Override the model for any specific review without modifying your configuration:

```bash
kit review --model gpt-4.1-nano https://github.com/owner/repo/pull/123
kit review --model gpt-4.1 https://github.com/owner/repo/pull/123


# Short flag also works
kit review -m claude-sonnet-4-20250514 https://github.com/owner/repo/pull/123
```

Tip

**Model validation**: Kit automatically validates model names and provides helpful suggestions if you mistype. Try `kit review --model gpt4` to see the validation in action!

## Available Models

### Free Local AI (Ollama)

Perfect for unlimited reviews without external API costs:

```bash
# Popular coding models
qwen2.5-coder:latest      # Excellent for code analysis
deepseek-r1:latest        # Strong reasoning capabilities
gemma3:latest             # Good general purpose
devstral:latest           # Mistral's coding model
llama3.2:latest           # Meta's latest model
codellama:latest          # Code-specialized Llama
```

**Setup:**

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh


# 2. Pull a model
ollama pull qwen2.5-coder:latest


# 3. Use with kit
kit review --model qwen2.5-coder:latest <pr-url>
```

### OpenAI Models

```bash
# Budget options
gpt-4.1-nano             # Ultra-budget: ~$0.0015-0.004
gpt-4.1-mini             # Budget-friendly: ~$0.005-0.015
gpt-4o-mini              # Newer mini model


# Standard options
gpt-4.1                  # Good balance: ~$0.02-0.10
gpt-4o                   # Latest GPT-4 model
gpt-4-turbo              # Fast GPT-4 variant
```

### Anthropic Claude

```bash
# Budget option
claude-3-5-haiku-20241022    # Fast and economical


# Recommended
claude-3-5-sonnet-20241022   # Excellent balance
claude-sonnet-4-20250514     # Latest Sonnet (recommended)


# Premium
claude-opus-4-20250514       # Highest quality
```

### Google Gemini

```bash
# Ultra-budget
gemini-1.5-flash-8b         # ~$0.003 per review


# Standard options
gemini-2.5-flash            # Excellent value: ~$0.007
gemini-1.5-flash            # Fast and efficient
gemini-1.5-pro              # More capable
gemini-2.5-pro              # Latest pro model
```

Tip

**Pro tip**: Use different models based on PR complexity. Save `claude-opus-4` for architectural changes and use `gpt-4.1-nano` for documentation/minor fixes.

## API Key Setup

### GitHub Token

Get from [GitHub Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens)

**Required permissions:**

* `repo` (for private repositories)
* `public_repo` (for public repositories)
* `pull_requests:write` (to post comments)

```bash
export KIT_GITHUB_TOKEN="ghp_your_token_here"
```

### LLM Provider API Keys

**Anthropic Claude (Recommended):**

```bash
export KIT_ANTHROPIC_TOKEN="sk-ant-your_key"
```

Get from: [Anthropic Console](https://console.anthropic.com/)

**OpenAI GPT Models:**

```bash
export KIT_OPENAI_TOKEN="sk-your_openai_key"
```

Get from: [OpenAI Platform](https://platform.openai.com/api-keys)

**Google Gemini:**

```bash
export KIT_GOOGLE_API_KEY="AIzaSy-your_google_key"
```

Get from: [Google AI Studio](https://aistudio.google.com/app/apikey)

**Ollama (Local - No API Key Required):**

```bash
# Just ensure Ollama is running
ollama serve
```

## Configuration Files

### Basic Configuration

Edit `~/.kit/review-config.yaml`:

```yaml
github:
  token: ghp_your_token_here
  base_url: https://api.github.com


llm:
  provider: anthropic  # or "openai", "google", "ollama"
  model: claude-sonnet-4-20250514
  api_key: sk-ant-your_key_here
  max_tokens: 4000
  temperature: 0.1


review:
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
  max_files: 50
```

### Provider-Specific Configurations

**Anthropic Claude:**

```yaml
github:
  token: ghp_your_token_here
  base_url: https://api.github.com


llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: sk-ant-your_key_here
  max_tokens: 4000
  temperature: 0.1


review:
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
  max_files: 50
```

**OpenAI GPT:**

```yaml
github:
  token: ghp_your_token_here
  base_url: https://api.github.com


llm:
  provider: openai
  model: gpt-4.1
  api_key: sk-your_openai_key
  max_tokens: 4000
  temperature: 0.1


review:
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
  max_files: 50
```

**Google Gemini:**

```yaml
github:
  token: ghp_your_token_here
  base_url: https://api.github.com


llm:
  provider: google
  model: gemini-2.5-flash  # or gemini-1.5-flash-8b for ultra-budget
  api_key: AIzaSy-your_google_key
  max_tokens: 4000
  temperature: 0.1


review:
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
  max_files: 50
```

**Free Local AI (Ollama):**

```yaml
github:
  token: ghp_your_token_here  # Still need GitHub API access
  base_url: https://api.github.com


llm:
  provider: ollama
  model: qwen2.5-coder:latest  # or deepseek-r1:latest
  api_base_url: http://localhost:11434
  api_key: ollama  # Placeholder (Ollama doesn't use API keys)
  max_tokens: 4000
  temperature: 0.1


review:
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
  max_files: 50
```

## Priority Filtering Configuration

### Default Priority Settings

```yaml
github:
  token: ghp_your_token_here
  base_url: https://api.github.com


llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: sk-ant-your_key_here
  max_tokens: 4000
  temperature: 0.1


review:
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
  max_files: 50
  # Optional: Set default priority filter
  priority_filter: ["high", "medium"]  # Only show important issues by default
```

### Priority Configuration Examples

**Security-focused configuration:**

```yaml
review:
  priority_filter: ["high"]  # Critical issues only
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
```

**General development workflow:**

```yaml
review:
  priority_filter: ["high", "medium"]  # Skip style suggestions
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
```

**Code quality/style reviews:**

```yaml
review:
  priority_filter: ["low"]  # Focus on improvements
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
```

**Default (show all priorities):**

```yaml
review:
  priority_filter: ["high", "medium", "low"]  # Same as omitting priority_filter
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
```

## Advanced Configuration Options

### Repository Analysis Settings

```yaml
review:
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
  max_files: 50


  # Advanced settings
  max_file_size: 1048576      # 1MB max file size
  exclude_patterns:           # Files to ignore
    - "*.lock"
    - "package-lock.json"
    - "yarn.lock"
    - "*.min.js"
    - "dist/"
    - "build/"


  include_patterns:           # Only analyze these files (if specified)
    - "*.py"
    - "*.js"
    - "*.ts"
    - "*.java"
    - "*.go"


  analysis_timeout: 300       # 5 minute timeout
  retry_attempts: 3           # Retry failed requests
```

### Multi-Provider Configuration

```yaml
github:
  token: ghp_your_token_here
  base_url: https://api.github.com


# Default provider
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: sk-ant-your_key_here
  max_tokens: 4000
  temperature: 0.1


# Alternative providers (for CLI override)
providers:
  openai:
    api_key: sk-your_openai_key
    api_base_url: https://api.openai.com/v1


  google:
    api_key: AIzaSy-your_google_key
    api_base_url: https://generativelanguage.googleapis.com/v1beta


  ollama:
    api_base_url: http://localhost:11434
    api_key: ollama


review:
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
  max_files: 50
```

### Custom Profile Defaults

```yaml
github:
  token: ghp_your_token_here
  base_url: https://api.github.com


llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: sk-ant-your_key_here
  max_tokens: 4000
  temperature: 0.1


review:
  post_as_comment: true
  clone_for_analysis: true
  cache_repos: true
  max_files: 50


  # Default profile for all reviews
  default_profile: "company-standards"


  # Repository-specific profiles
  profile_overrides:
    "cased/frontend-app": "frontend-react"
    "cased/api-service": "backend-python"
    "cased/security-lib": "security-focused"
```

## Environment Variable Overrides

You can override any configuration setting using environment variables:

```bash
# GitHub settings
export KIT_GITHUB_TOKEN="ghp_your_token"
export KIT_GITHUB_BASE_URL="https://api.github.com"


# LLM settings
export KIT_LLM_PROVIDER="anthropic"
export KIT_LLM_MODEL="claude-sonnet-4-20250514"
export KIT_ANTHROPIC_TOKEN="sk-ant-your_key"
export KIT_LLM_MAX_TOKENS="4000"
export KIT_LLM_TEMPERATURE="0.1"


# Review settings
export KIT_REVIEW_POST_AS_COMMENT="true"
export KIT_REVIEW_CACHE_REPOS="true"
export KIT_REVIEW_MAX_FILES="50"
export KIT_REVIEW_PRIORITY_FILTER="high,medium"
```

## Configuration Validation

Test your configuration:

```bash
# Initialize configuration with guided setup
kit review --init-config


# Validate current configuration
kit review --validate-config


# Test with dry run
kit review --dry-run --model claude-sonnet-4 https://github.com/owner/repo/pull/123
```

## Multiple Configuration Profiles

### Team-Specific Configs

```bash
# Create team-specific config directories
mkdir -p ~/.kit/profiles/frontend-team
mkdir -p ~/.kit/profiles/backend-team
mkdir -p ~/.kit/profiles/security-team


# Frontend team config
cat > ~/.kit/profiles/frontend-team/review-config.yaml << EOF
llm:
  provider: openai
  model: gpt-4.1-mini
  api_key: sk-frontend-team-key
review:
  default_profile: "frontend-react"
  priority_filter: ["high", "medium"]
EOF


# Use specific config
KIT_CONFIG_DIR=~/.kit/profiles/frontend-team kit review <pr-url>
```

### Project-Specific Configs

```bash
# In your project directory
mkdir .kit
cat > .kit/review-config.yaml << EOF
llm:
  provider: ollama
  model: qwen2.5-coder:latest
review:
  default_profile: "project-standards"
  max_files: 30
EOF


# Kit automatically uses project-local config if available
kit review <pr-url>
```

## Cost Management Configuration

### Budget Controls

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: sk-ant-your_key_here
  max_tokens: 4000
  temperature: 0.1


  # Cost controls
  cost_limit_per_review: 0.50    # Maximum $0.50 per review
  monthly_cost_limit: 100.00     # Maximum $100 per month


review:
  # Auto-downgrade model if cost limit exceeded
  fallback_model: "gpt-4.1-mini"


  # Skip review if PR is too large
  max_cost_estimate: 1.00        # Skip if estimated cost > $1.00
```

### Usage Tracking

```yaml
tracking:
  enabled: true
  log_file: ~/.kit/usage.log
  metrics_endpoint: https://your-metrics-server.com/api/usage
  team_id: "engineering-team"
```

## Troubleshooting

### Common Issues

**1. API Key Issues:**

```bash
# Test API key
curl -H "Authorization: Bearer sk-ant-your_key" \
  https://api.anthropic.com/v1/messages


# Check environment
echo $KIT_ANTHROPIC_TOKEN
```

**2. Model Availability:**

```bash
# List available models
kit review --list-models


# Test specific model
kit review --model claude-sonnet-4 --dry-run <pr-url>
```

**3. GitHub Permissions:**

```bash
# Test GitHub token
curl -H "Authorization: token ghp_your_token" \
  https://api.github.com/user


# Check permissions
gh auth status
```

### Debug Mode

```bash
# Enable debug logging
export KIT_DEBUG=true
kit review --dry-run <pr-url>


# Verbose output
kit review --verbose <pr-url>
```

### Configuration Reset

```bash
# Reset to defaults
rm ~/.kit/review-config.yaml
kit review --init-config


# Backup current config
cp ~/.kit/review-config.yaml ~/.kit/review-config.yaml.backup
```

Tip

**Pro Tip**: Start with a simple configuration and gradually add complexity. Use `--dry-run` mode to test changes before applying them to live reviews.

## Repository Configuration Options

### Using Existing Local Repositories

Skip cloning and use an existing local repository for analysis:

```bash
# Use existing repository instead of cloning
kit review --repo-path /path/to/existing/repo https://github.com/owner/repo/pull/123


# Works with any other flags
kit review --repo-path ~/projects/myproject --model gpt-4.1-nano https://github.com/owner/repo/pull/123


# Can be combined with configuration files
kit review --repo-path /workspace/repo --config custom-config.yaml https://github.com/owner/repo/pull/123
```

Caution

**Important**: When using `--repo-path`, the analysis is performed against the current state of your local repository, which may not reflect the main branch. Kit will display a warning to remind you of this.

**Benefits of using existing repositories:**

* **Faster analysis**: Skip cloning time for large repositories
* **Local development**: Analyze PRs against your working copy with local changes
* **Network efficiency**: No need to download repositories you already have
* **Bandwidth savings**: Useful for large repositories or limited internet connections

***

[← Back to PR Reviewer Overview](/pr-reviewer/)

# Examples & Use Cases

> Real-world review examples with actual costs and analysis across different project types and scenarios

# Examples & Use Cases

See real-world AI code reviews with actual costs, analysis depth, and practical outcomes across different project types and scenarios.

## Real-World Review Examples

### Large Framework Change

**[FastAPI Packaging Change](https://github.com/cased/kit/blob/main/src/kit/pr_review/example_reviews/fastapi_11935_standard_dependencies.md)** - Architectural impact analysis

* **Cost**: $0.034
* **Model**: claude-sonnet-4
* **Files Changed**: 12 files, 150+ lines
* **Focus**: Architectural impact, dependency management, breaking changes
* **Key Findings**: Identified potential breaking changes, suggested migration strategies

**Why this example matters**: Shows how kit handles complex architectural changes with full repository context, identifying cross-module impacts that diff-only tools miss.

### Frontend UI Enhancement

**[React.dev UI Feature](https://github.com/cased/kit/blob/main/src/kit/pr_review/example_reviews/react_dev_6986_branding_menu.md)** - Accessibility-focused review

* **Cost**: $0.012
* **Model**: gpt-4.1
* **Files Changed**: 6 files, 85 lines
* **Focus**: Accessibility, component design, user experience
* **Key Findings**: Accessibility improvements, component reusability suggestions

**Why this example matters**: Demonstrates kit’s ability to provide specialized feedback on UI/UX concerns, not just technical correctness.

### Documentation Update

**[BioPython Documentation Fix](https://github.com/cased/kit/blob/main/src/kit/pr_review/example_reviews/biopython_204_documentation_fix.md)** - Proportional response

* **Cost**: $0.006
* **Model**: gpt-4.1-mini
* **Files Changed**: 2 files, 15 lines
* **Focus**: Documentation clarity, example accuracy
* **Key Findings**: Minor suggestions for clarity, validation of examples

**Why this example matters**: Shows how kit provides proportional feedback - thorough but concise for documentation changes.

### Multi-Model Comparison

**[Model Comparison Analysis](https://github.com/cased/kit/blob/main/src/kit/pr_review/example_reviews/model_comparison_fastapi_11935.md)** - Cost vs quality analysis

Compares the same PR reviewed with different models:

* **GPT-4.1-nano**: $0.004 - High-level issues
* **GPT-4.1**: $0.034 - Detailed analysis
* **Claude Sonnet**: $0.087 - Comprehensive review
* **Claude Opus**: $0.156 - Architectural insights

**Why this example matters**: Helps teams choose the right model for their budget and quality requirements.

## Use Case Scenarios

### Security-Critical Changes

```bash
# Use security-focused profile with premium model
kit review --profile security-standards \
  --model claude-opus-4 \
  --priority=high \
  https://github.com/company/auth-service/pull/234
```

**Typical output focus**:

* Input validation vulnerabilities
* Authentication/authorization issues
* Secrets management problems
* Dependency security concerns
* Logging of sensitive data

### High-Volume Development

```bash
# Cost-optimized for daily reviews
kit review --model gpt-4.1-nano \
  --priority=high,medium \
  https://github.com/company/api/pull/456
```

**Benefits**:

* Reviews at \~$0.002-0.015 per PR
* Focus on important issues only
* Fast turnaround for daily workflow
* Sustainable for 100+ PRs/month

### Large Refactoring

```bash
# Comprehensive analysis for major changes
kit review --model claude-sonnet-4 \
  --profile architecture-standards \
  https://github.com/company/core/pull/789
```

**Typical output focus**:

* Cross-module impact analysis
* Breaking change identification
* Performance implications
* Backward compatibility concerns
* Migration strategy suggestions

### Code Quality Focus

```bash
# Emphasize style and improvements
kit review --priority=low \
  --profile code-quality \
  --model gpt-4.1-mini \
  https://github.com/company/utils/pull/101
```

**Typical output focus**:

* Code style improvements
* Refactoring opportunities
* Documentation enhancements
* Test coverage suggestions
* Performance optimizations

### Local Development Workflow

```bash
# Analyze PR against your local working copy
kit review --repo-path ~/projects/myapp \
  --model gpt-4.1-mini \
  https://github.com/company/myapp/pull/456
```

**Typical output focus**:

* Compatibility with your local changes
* Analysis against current branch state
* Integration testing considerations
* Merge conflict potential

Tip

**Local Development Pro Tip**: Use `--repo-path` when you want to analyze how a PR integrates with your local changes or when working on a feature branch that’s not yet pushed.

### Large Repository Optimization

```bash
# Skip cloning large repositories you already have locally
kit review --repo-path /workspace/large-monorepo \
  --model claude-sonnet-4 \
  --priority high,medium \
  https://github.com/company/monorepo/pull/789
```

**Benefits**:

* Save 5-15 minutes of cloning time for large repos
* Preserve bandwidth for remote/mobile development
* Use your existing repository cache
* Work with repositories that have complex setup requirements

### Offline Development Support

```bash
# Work offline with cached repositories
kit review --repo-path /local/cache/project \
  --model ollama:qwen2.5-coder:latest \
  --dry-run \
  https://github.com/company/project/pull/123
```

**Use case**:

* Limited or expensive internet connectivity
* Air-gapped development environments
* Local-only testing and validation
* Offline model usage with Ollama

## Team Workflow Examples

### Startup Team (Budget-Conscious)

.github/workflows/ai-review\.yml

```yaml
name: Budget AI Review
on:
  pull_request:
    types: [opened, synchronize]


jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - name: AI Review
        run: |
          pip install cased-kit
          # Use ultra-budget model for all PRs
          kit review --model gpt-4.1-nano \
            --priority=high,medium \
            ${{ github.event.pull_request.html_url }}
```

**Results**:

* **Cost**: \~$5-15/month for 500 PRs
* **Coverage**: Critical and important issues
* **Speed**: Fast reviews, good for rapid iteration

### Enterprise Team (Quality-Focused)

.github/workflows/ai-review\.yml

```yaml
name: Enterprise AI Review
on:
  pull_request:
    types: [opened, synchronize]


jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - name: AI Review with Smart Selection
        run: |
          pip install cased-kit


          # Different models based on target branch
          if [ "${{ github.event.pull_request.base.ref }}" == "main" ]; then
            MODEL="claude-sonnet-4"
            PROFILE="production-standards"
          else
            MODEL="gpt-4.1"
            PROFILE="development-standards"
          fi


          kit review --model "$MODEL" \
            --profile "$PROFILE" \
            ${{ github.event.pull_request.html_url }}
```

**Results**:

* **Cost**: \~$50-150/month for 500 PRs
* **Coverage**: Comprehensive analysis
* **Quality**: High-quality, detailed feedback

### Open Source Project

.github/workflows/ai-review\.yml

```yaml
name: Community AI Review
on:
  pull_request:
    types: [opened, synchronize]
    # Only review PRs from outside contributors
  if: github.event.pull_request.head.repo.full_name != github.repository


jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - name: Community PR Review
        run: |
          pip install cased-kit
          # Focus on contribution guidelines
          kit review --profile community-standards \
            --model gpt-4.1-mini \
            ${{ github.event.pull_request.html_url }}
```

**Results**:

* **Purpose**: Help external contributors
* **Focus**: Style, testing, documentation
* **Cost**: Minimal, only for external PRs

### DevSecOps Team (Security-First)

.github/workflows/security-review\.yml

```yaml
name: Security-Focused Review
on:
  pull_request:
    types: [opened, synchronize]
    paths:
      - 'src/auth/**'
      - 'src/api/**'
      - '**/*security*'
      - '**/*auth*'


jobs:
  security-review:
    runs-on: ubuntu-latest
    steps:
      - name: Security Review
        run: |
          pip install cased-kit
          # Premium model for security-critical code
          kit review --model claude-opus-4 \
            --profile security-hardening \
            --priority=high \
            ${{ github.event.pull_request.html_url }}
```

**Results**:

* **Focus**: Security vulnerabilities only
* **Quality**: Maximum thoroughness for critical code
* **Cost**: Higher per review, but targeted scope

## Cost Analysis Examples

### Monthly Budget Planning

**Small Team (10 developers, 200 PRs/month)**:

```bash
# Budget option: ~$10-30/month
kit review --model gpt-4.1-nano --priority=high,medium


# Balanced option: ~$20-60/month
kit review --model gpt-4.1-mini


# Premium option: ~$60-180/month
kit review --model claude-sonnet-4
```

**Large Team (50 developers, 1000 PRs/month)**:

```bash
# Smart tiering based on PR size
small_pr="gpt-4.1-nano"     # <5 files: ~$2-8/month per dev
medium_pr="gpt-4.1-mini"    # 5-20 files: ~$10-30/month per dev
large_pr="claude-sonnet-4"  # >20 files: ~$20-60/month per dev


# Total: ~$32-98/month per developer
```

### ROI Analysis Examples

**Bug Prevention**:

* **Cost**: $50/month for AI reviews
* **Prevented**: 2-3 production bugs/month
* **Savings**: $2000-15000 in bug fix costs
* **ROI**: 40-300x return on investment

**Code Quality Improvement**:

* **Cost**: $100/month for comprehensive reviews
* **Result**: 25% reduction in tech debt accumulation
* **Savings**: Faster development velocity
* **ROI**: Pays for itself in reduced maintenance time

## Integration Examples

### Slack Notifications

slack-integration.sh

```bash
#!/bin/bash
REVIEW=$(kit review -p --priority=high "$1")
CRITICAL_COUNT=$(echo "$REVIEW" | grep -c "High Priority")


if [ "$CRITICAL_COUNT" -gt 0 ]; then
  curl -X POST "$SLACK_WEBHOOK" \
    -H 'Content-type: application/json' \
    --data '{
      "text": "🚨 Critical issues found in PR '"$1"'",
      "attachments": [{
        "color": "danger",
        "text": "'"$(echo "$REVIEW" | head -500)"'"
      }]
    }'
else
  curl -X POST "$SLACK_WEBHOOK" \
    -H 'Content-type: application/json' \
    --data '{
      "text": "✅ PR '"$1"' looks good to go!"
    }'
fi
```

### Dashboard Metrics

metrics-collection.py

```python
#!/usr/bin/env python3
import subprocess
import json
import requests
from datetime import datetime


def collect_review_metrics(pr_url):
    # Get review with cost information
    result = subprocess.run([
        'kit', 'review', '--dry-run', '-p', pr_url
    ], capture_output=True, text=True)


    # Parse metrics
    lines = result.stderr.split('\n')
    cost = next((l for l in lines if 'Total cost:' in l), '').split('$')[-1]
    model = next((l for l in lines if 'Model:' in l), '').split(':')[-1].strip()


    # Extract issue counts
    issues = result.stdout.count('Priority:')
    high_priority = result.stdout.count('High Priority')


    # Send to dashboard
    metrics = {
        'pr_url': pr_url,
        'timestamp': datetime.now().isoformat(),
        'cost': float(cost) if cost else 0,
        'model': model,
        'total_issues': issues,
        'critical_issues': high_priority
    }


    requests.post('https://dashboard.company.com/api/reviews', json=metrics)
    return metrics


if __name__ == "__main__":
    import sys
    collect_review_metrics(sys.argv[1])
```

### Issue Tracker Integration

jira-integration.sh

```bash
#!/bin/bash
REVIEW=$(kit review -p --priority=high "$1")
SECURITY_ISSUES=$(echo "$REVIEW" | grep -i "security\|vulnerability" | wc -l)


if [ "$SECURITY_ISSUES" -gt 0 ]; then
  # Create security ticket
  jira issue create \
    --project="SEC" \
    --type="Security" \
    --summary="Security issues found in $1" \
    --description="$REVIEW" \
    --priority="High"
fi


PERFORMANCE_ISSUES=$(echo "$REVIEW" | grep -i "performance\|slow\|optimization" | wc -l)
if [ "$PERFORMANCE_ISSUES" -gt 0 ]; then
  # Create performance ticket
  jira issue create \
    --project="PERF" \
    --type="Task" \
    --summary="Performance issues found in $1" \
    --description="$REVIEW" \
    --priority="Medium"
fi
```

## Best Practices from Examples

### Model Selection Strategy

1. **Documentation/Small Changes**: `gpt-4.1-nano` or `gpt-4.1-mini`
2. **Regular Development**: `gpt-4.1` or `gemini-2.5-flash`
3. **Critical/Security Changes**: `claude-sonnet-4` or `claude-opus-4`
4. **Architectural Changes**: `claude-opus-4` for comprehensive analysis
5. **High-Volume Teams**: Mix of models based on PR complexity

### Priority Filtering Strategy

1. **Daily Development**: `--priority=high,medium` (focus on important issues)
2. **Pre-Release**: `--priority=high` (only critical blockers)
3. **Code Quality Reviews**: `--priority=low` (style and improvements)
4. **Security Audits**: `--priority=high` with security profile
5. **Architecture Reviews**: All priorities with premium model

### Profile Usage Patterns

1. **General Development**: `company-standards` profile
2. **Security-Critical**: `security-hardening` profile
3. **Frontend Work**: `frontend-react` or `ui-standards` profile
4. **Backend APIs**: `backend-api` or `microservice-standards` profile
5. **External Contributors**: `community-guidelines` profile

Tip

**Pro Tip**: Start simple with one model and basic priority filtering. Add complexity gradually as your team sees value and develops preferences for different scenarios.

***

[← Back to PR Reviewer Overview](/pr-reviewer/)

# Output Modes & Integration

> Different output modes for various workflows - from direct GitHub posting to piping output to CLI code writers

# Output Modes & Integration

Kit provides different output modes for various workflows - from direct GitHub posting to piping output to CLI code writers and custom automation systems.

## Output Modes

### Standard Mode

```bash
# Posts comment directly to GitHub PR
kit review https://github.com/owner/repo/pull/123
```

**Output includes**: Review comment posted to GitHub, status messages, cost breakdown, and quality metrics.

### Dry Run Mode (`--dry-run` / `-n`)

```bash
# Shows formatted preview without posting
kit review --dry-run https://github.com/owner/repo/pull/123
```

**Output includes**: Status messages, cost breakdown, quality metrics, and formatted review preview. Perfect for testing configurations and seeing full diagnostics.

### Plain Mode (`--plain` / `-p`)

```bash
# Clean output perfect for piping to other tools
kit review --plain https://github.com/owner/repo/pull/123
kit review -p https://github.com/owner/repo/pull/123
```

**Output**: Just the raw review content with no status messages or formatting. Ideal for automation and piping workflows.

## Repository Options

### Using Existing Repositories

```bash
# Skip cloning with existing local repository
kit review --repo-path /path/to/repo https://github.com/owner/repo/pull/123


# Combine with dry-run mode for local testing
kit review --repo-path ~/projects/myapp --dry-run https://github.com/company/myapp/pull/456


# Use with plain mode for piping workflows
kit review --repo-path /workspace/project -p https://github.com/company/project/pull/789
```

Note

**Repository State Warning**: When using `--repo-path`, kit analyzes the current state of your local repository, which may differ from the main branch. A warning message will be displayed to remind you of this.

**When to use `--repo-path`:**

* **Large repositories**: Skip cloning time for repos you already have locally
* **Local development**: Analyze PRs against your working copy with uncommitted changes
* **Bandwidth constraints**: Useful for large repositories or limited internet connections
* **Offline workflows**: Work with cached repositories when internet is unavailable

## Priority Filtering

Focus reviews on what matters most:

```bash
# Focus on critical issues only
kit review --priority=high https://github.com/owner/repo/pull/123


# Show important issues (high + medium priority)
kit review --priority=high,medium https://github.com/owner/repo/pull/123


# Get only style/improvement suggestions
kit review --priority=low https://github.com/owner/repo/pull/123


# Combine with other flags
kit review -p --priority=high https://github.com/owner/repo/pull/123 | \
  claude "Fix these critical security issues"
```

**Benefits**: Reduces noise, saves costs, and focuses attention on issues that matter to your workflow.

## Piping to CLI Code Writers

Combine kit’s repository intelligence with your favorite CLI code-generation tooling:

### Basic Piping Patterns

```bash
# Analyze with kit, implement with Claude Code
kit review -p https://github.com/owner/repo/pull/123 | \
  claude -p "Implement all the suggestions from this code review"


# Use specific models for cost optimization
kit review -p --model gpt-4.1-nano https://github.com/owner/repo/pull/123 | \
  claude -p "Fix the high-priority issues mentioned in this review"


# Focus on critical issues only
kit review -p --priority=high https://github.com/owner/repo/pull/123 | \
  claude -p "Implement these security and reliability fixes"
```

### Advanced Multi-Stage Workflows

```bash
# Stage 1: Kit analyzes codebase context
REVIEW=$(kit review -p --model claude-3-5-haiku-20241022 <pr-url>)


# Stage 2: Claude Code implements fixes
echo "$REVIEW" | claude -p "Implement these code review suggestions"


# Stage 3: Your custom tooling
echo "$REVIEW" | your-priority-tracker --extract-issues
```

Tip

**Why this workflow helps**: Kit excels at codebase understanding and context, while Claude Code excels at implementation. Combining them gives you AI-powered analysis → AI-powered fixes in seconds.

## Integration Examples

### Pipe to Any Tool

```bash
# Save review to file
kit review -p <pr-url> > review.md


# Send to Slack
kit review -p <pr-url> | curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"'"$(cat)"'"}' YOUR_SLACK_WEBHOOK


# Process with jq or other tools
kit review -p <pr-url> | your-custom-processor


# Extract specific sections
kit review -p <pr-url> | grep -A 5 "High Priority"


# Convert to different formats
kit review -p <pr-url> | pandoc -f markdown -t html > review.html
```

### Custom Processing Scripts

**Extract Security Issues:**

extract-security-issues.sh

```bash
#!/bin/bash
kit review -p --priority=high "$1" | \
  grep -i "security\|vulnerability\|injection\|auth" | \
  while read -r line; do
    echo "🔒 $line" >> security-report.md
  done
```

**Team Notification System:**

notify-team.sh

```bash
#!/bin/bash
REVIEW=$(kit review -p "$1")
CRITICAL_COUNT=$(echo "$REVIEW" | grep -c "High Priority")


if [ "$CRITICAL_COUNT" -gt 0 ]; then
  echo "$REVIEW" | your-slack-bot --channel security --urgent
else
  echo "$REVIEW" | your-slack-bot --channel code-review
fi
```

## CI/CD Integration Patterns

### Basic GitHub Actions Integration

```yaml
- name: AI Review with Plain Output
  run: |
    pip install cased-kit
    kit review -p ${{ github.event.pull_request.html_url }} > review.txt


    # Post to PR as comment
    gh pr comment ${{ github.event.pull_request.number }} --body-file review.txt
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

### Multi-Stage Processing

```yaml
- name: AI Review with Multi-Stage Processing
  run: |
    pip install cased-kit


    # Stage 1: Generate review with kit's repository intelligence
    REVIEW=$(kit review -p --model claude-3-5-haiku-20241022 ${{ github.event.pull_request.html_url }})


    # Stage 2: Extract action items and post as separate comment
    echo "$REVIEW" | your-issue-tracker --extract-priorities | \
      gh pr comment ${{ github.event.pull_request.number }} --body-file -


    # Stage 3: Save review for later processing
    echo "$REVIEW" > review-${{ github.event.pull_request.number }}.md


    # Stage 4: Send to team notification system
    echo "$REVIEW" | your-slack-notifier --channel engineering
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

### Integration with External Tools

```yaml
- name: Review and Process
  run: |
    pip install cased-kit


    # Get clean review output for processing
    kit review -p ${{ github.event.pull_request.html_url }} > raw-review.txt


    # Parse with your custom tools
    python scripts/extract-security-issues.py raw-review.txt
    python scripts/update-team-dashboard.py raw-review.txt
    python scripts/generate-metrics.py raw-review.txt


    # Optional: Post processed results back to PR
    gh pr comment ${{ github.event.pull_request.number }} --body-file processed-summary.md
```

## Cost Optimization Strategies

### Budget-Conscious Multi-Stage Analysis

```bash
# Stage 1: Quick high-priority scan with budget model
HIGH_ISSUES=$(kit review -p --model gpt-4o-mini --priority=high <pr-url>)


# Stage 2: If critical issues found, do full review with premium model
if echo "$HIGH_ISSUES" | grep -q "High Priority"; then
  kit review --model claude-sonnet-4 <pr-url>
else
  echo "$HIGH_ISSUES" | gh pr comment $PR_NUMBER --body-file -
fi
```

### Model Selection Based on Content

smart-review\.sh

```bash
#!/bin/bash
PR_URL=$1
PR_NUMBER=$(echo "$PR_URL" | grep -o '/pull/[0-9]*' | cut -d'/' -f3)


# Check PR size and select appropriate model
FILES_CHANGED=$(gh pr view "$PR_NUMBER" --json files --jq '.files | length')


if [ "$FILES_CHANGED" -gt 20 ]; then
  MODEL="claude-sonnet-4"    # Premium for large changes
elif [ "$FILES_CHANGED" -gt 5 ]; then
  MODEL="gpt-4.1"           # Mid-tier for medium changes
else
  MODEL="gpt-4.1-nano"      # Budget for small changes
fi


kit review --model "$MODEL" -p "$PR_URL"
```

## Best Practices

### Output Mode Selection

* **Standard mode**: For direct GitHub integration and team collaboration
* **Dry run (`--dry-run`)**: For testing configurations and seeing full diagnostics
* **Plain mode (`--plain` / `-p`)**: For piping to CLI code composers, custom tools, or automation
* **Tip**: Use plain mode in CI/CD for processing reviews with external systems

### Piping & Integration Workflows

* **Kit → Claude Code**: `kit review -p <pr-url> | claude "implement these suggestions"`
* **Multi-tool chains**: Use kit for analysis, pipe to specialized tools for processing
* **Cost optimization**: Use budget models for analysis, premium models for implementation
* **Automation**: Pipe plain output to issue trackers, notification systems, or custom processors

### Error Handling in Scripts

robust-review\.sh

```bash
#!/bin/bash
set -e  # Exit on error


PR_URL=$1
if [ -z "$PR_URL" ]; then
  echo "Usage: $0 <pr-url>"
  exit 1
fi


# Generate review with error handling
if REVIEW=$(kit review -p "$PR_URL" 2>/dev/null); then
  echo "✅ Review generated successfully"
  echo "$REVIEW" | your-processor
else
  echo "❌ Review failed, check configuration"
  exit 1
fi
```

## Real-World Integration Examples

### Slack Integration

slack-review\.sh

````bash
#!/bin/bash
REVIEW=$(kit review -p --priority=high,medium "$1")
WEBHOOK_URL="your-slack-webhook-url"


curl -X POST -H 'Content-type: application/json' \
  --data '{
    "text": "🤖 AI Code Review Complete",
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "```'"$REVIEW"'```"
        }
      }
    ]
  }' "$WEBHOOK_URL"
````

### Jira Integration

jira-review\.sh

```bash
#!/bin/bash
REVIEW=$(kit review -p --priority=high "$1")
CRITICAL_ISSUES=$(echo "$REVIEW" | grep -c "High Priority" || echo "0")


if [ "$CRITICAL_ISSUES" -gt 0 ]; then
  # Create Jira ticket for critical issues
  jira issue create \
    --project="CODE" \
    --type="Bug" \
    --summary="Critical issues found in PR review" \
    --description="$REVIEW"
fi
```

### Dashboard Integration

dashboard-integration.py

```python
#!/usr/bin/env python3
import sys
import requests
import subprocess


def get_review(pr_url):
    result = subprocess.run([
        'kit', 'review', '-p', '--priority=high,medium', pr_url
    ], capture_output=True, text=True)
    return result.stdout


def post_to_dashboard(review_data):
    response = requests.post('https://your-dashboard.com/api/reviews',
                           json={'content': review_data})
    return response.status_code == 200


if __name__ == "__main__":
    pr_url = sys.argv[1]
    review = get_review(pr_url)
    success = post_to_dashboard(review)
    print(f"Dashboard update: {'✅' if success else '❌'}")
```

[← Back to PR Reviewer Overview](/pr-reviewer/)

# Custom Context Profiles

> Store and apply organization-specific coding standards and review guidelines through custom context profiles

# Custom Context Profiles

Kit supports **organization-specific coding standards and review guidelines** through custom context profiles. Create profiles that automatically inject your company’s coding standards, security requirements, and style guidelines into every PR review, ensuring consistent and organization-aligned feedback.

Tip

**Why this matters**: Instead of manually specifying guidelines for each review, profiles encode your organization’s knowledge once and apply it consistently across all reviews. Perfect for teams with specific security requirements, architectural standards, or coding conventions.

## Quick Start

```bash
# Create a profile from your existing coding guidelines
kit review-profile create --name company-standards \
  --file coding-guidelines.md \
  --description "Acme Corp coding standards"


# Use in any review
kit review --profile company-standards https://github.com/owner/repo/pull/123


# List all profiles
kit review-profile list
```

## Profile Management

### Creating Profiles

**From a file (recommended for sharing):**

```bash
kit review-profile create \
  --name python-security \
  --file security-guidelines.md \
  --description "Python security best practices" \
  --tags "security,python"
```

**Interactive creation:**

```bash
kit review-profile create \
  --name company-standards \
  --description "Company coding standards"
# Then type your guidelines, press Enter for new lines, then Ctrl+D to finish
```

### Managing Profiles

```bash
# List all profiles with details
kit review-profile list --format table


# Show specific profile content
kit review-profile show --name company-standards


# Edit existing profile
kit review-profile edit --name company-standards \
  --file updated-guidelines.md


# Share profiles between team members
kit review-profile export --name company-standards \
  --file shared-standards.md
kit review-profile import --file shared-standards.md \
  --name imported-standards


# Clean up old profiles
kit review-profile delete --name old-profile
```

## Example Profile Content

Here’s an effective profile structure that provides concrete, actionable guidance:

### Security-Focused Profile

```markdown
**Security Review Guidelines:**


- **Input Validation**: All user inputs must be validated against expected formats
- **SQL Injection Prevention**: Use parameterized queries, never string concatenation
- **XSS Prevention**: Sanitize all user content before rendering
- **Authentication**: Verify all endpoints require proper authentication
- **Authorization**: Check that users can only access resources they own
- **Secrets Management**: No hardcoded API keys, tokens, or passwords
- **Logging**: Sensitive data must not appear in logs
- **Dependencies**: Flag any new dependencies for security review
```

### Code Quality Profile

```markdown
**Code Quality Standards:**


- **Documentation**: All public functions must have docstrings with examples
- **Type Safety**: Use type hints for all function parameters and returns
- **Error Handling**: Implement proper exception handling with specific error types
- **Testing**: New features require unit tests with 80%+ coverage
- **Performance**: Flag N+1 queries and inefficient algorithms
- **Architecture**: Follow SOLID principles, maintain loose coupling
```

## Using Profiles in Reviews

### Basic Usage

```bash
# Apply organization standards automatically
kit review --profile company-standards https://github.com/owner/repo/pull/123


# Combine with other options
kit review --profile security-focused \
  --priority=high \
  --model claude-sonnet-4 \
  https://github.com/owner/repo/pull/123


# Multiple contexts for different teams
kit review --profile backend-api https://github.com/owner/repo/pull/123    # API team
kit review --profile frontend-react https://github.com/owner/repo/pull/123  # UI team
```

### CI/CD Integration

```yaml
- name: AI Review with Company Standards
  run: |
    pip install cased-kit
    kit review --profile company-standards ${{ github.event.pull_request.html_url }}
  env:
    KIT_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Team Organization

### Team-Specific Profiles

```bash
# Different standards for different teams
kit review-profile create --name backend-api \
  --description "Backend API development standards" \
  --tags "backend,api,python"


kit review-profile create --name frontend-react \
  --description "React frontend standards" \
  --tags "frontend,react,typescript"


kit review-profile create --name security-hardening \
  --description "Security review guidelines" \
  --tags "security,compliance"
```

### Project-Type Profiles

```bash
# Different standards for different project types
kit review-profile create --name microservice-standards \
  --description "Microservice architecture guidelines"


kit review-profile create --name data-pipeline-standards \
  --description "Data processing best practices"


kit review-profile create --name mobile-app-standards \
  --description "Mobile development guidelines"
```

## Advanced Examples

### Multi-Modal Team Setup

```yaml
# In your CI/CD, use different profiles based on changed files
- name: Smart Profile Selection
  run: |
    pip install cased-kit


    # Check what type of files changed
    CHANGED_FILES=$(gh pr view ${{ github.event.pull_request.number }} --json files --jq -r '.files[].filename')


    if echo "$CHANGED_FILES" | grep -q "\.py$"; then
      PROFILE="python-backend"
    elif echo "$CHANGED_FILES" | grep -q "\.(ts|tsx|js|jsx)$"; then
      PROFILE="frontend-react"
    elif echo "$CHANGED_FILES" | grep -q "security\|auth"; then
      PROFILE="security-focused"
    else
      PROFILE="general-standards"
    fi


    kit review --profile "$PROFILE" ${{ github.event.pull_request.html_url }}
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    KIT_ANTHROPIC_TOKEN: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Profile Best Practices

### Writing Effective Profiles

1. **Be Specific**: Include concrete examples, not just general principles
2. **Focus on Intent**: Explain *why* certain practices are required
3. **Use Examples**: Show good and bad code patterns when possible
4. **Stay Current**: Regular review and update profiles as standards evolve
5. **Tag Appropriately**: Use tags for easy organization and discovery

### Team Workflow

1. **Start Small**: Begin with essential standards, expand over time
2. **Collaborate**: Involve team members in creating and updating profiles
3. **Version Control**: Export profiles and track them alongside your code
4. **Regular Reviews**: Schedule quarterly profile review meetings
5. **Share Success**: Use export/import to share effective profiles across teams

### Example Integration

```bash
# Morning routine: Update and sync team profiles
kit review-profile export --name company-standards --file standards.md
git add standards.md && git commit -m "Update coding standards"


# Review with latest standards
kit review --profile company-standards https://github.com/owner/repo/pull/123
```

## Storage and Sharing

* **Location**: Profiles stored in `~/.kit/profiles/` as human-readable YAML files
* **Format**: Includes metadata (name, description, tags, timestamps) and content
* **Sharing**: Export/import functionality for team collaboration and version control
* **Backup**: Include profile exports in your team’s configuration management

Tip

**Pro Tip**: When a profile is specified, the custom context is automatically injected into the review prompt as “Custom Review Guidelines”, ensuring the AI reviewer follows your organization’s standards while maintaining all kit’s repository intelligence features.

***

[← Back to PR Reviewer Overview](/pr-reviewer/)

# Documentation

> Documentation for kit.

This uses [Starlight](https://starlight.astro.build) to build the documentation.

## 🧞 Commands

All commands are run from the root of the project, from a terminal:

| Command                | Action                                           |
| :--------------------- | :----------------------------------------------- |
| `pnpm install`         | Installs dependencies                            |
| `pnpm dev`             | Starts local dev server at `localhost:4321`      |
| `pnpm build`           | Build your production site to `./dist/`          |
| `pnpm preview`         | Preview your build locally, before deploying     |
| `pnpm astro ...`       | Run CLI commands like `astro add`, `astro check` |
| `pnpm astro -- --help` | Get help using the Astro CLI                     |

## 👀 Want to learn more?

Check out [Starlight’s docs](https://starlight.astro.build/), read [the Astro documentation](https://docs.astro.build), or jump into the [Astro Discord server](https://astro.build/chat).

# Practical Recipes

Note

These snippets are *copy-paste-ready* solutions for common developer-productivity tasks with **kit**. Adapt them to scripts, CI jobs, or IDE plugins.

## 1. Rename every function `old_name` → `new_name`

```python
from pathlib import Path
from kit import Repository


repo = Repository("/path/to/project")


# Gather definitions & references (quick heuristic)
usages = repo.find_symbol_usages("old_name", symbol_type="function")


edits: dict[str, str] = {}
for u in usages:
    path, line = u["file"], u.get("line")
    if line is None:
        continue
    lines = repo.get_file_content(path).splitlines()
    lines[line] = lines[line].replace("old_name", "new_name")
    edits[path] = "\n".join(lines) + "\n"


# Apply edits – prompt the user first!
for rel_path, new_src in edits.items():
    Path(repo.repo_path, rel_path).write_text(new_src)


repo.mapper.scan_repo()  # refresh symbols if you’ll run more queries
```

***

## 2. Summarize a Git diff for an LLM PR review

```python
from kit import Repository
repo = Repository(".")
assembler = repo.get_context_assembler()
assembler.add_diff(diff_text)  # diff_text from `git diff`
summary = repo.get_summarizer().summarize(assembler.format_context())
print(summary)
```

***

## 3. Semantic search for authentication code

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embed = lambda text: model.encode([text])[0].tolist()


repo = Repository(".")
vs = repo.get_vector_searcher(embed_fn=embed)
vs.build_index()


hits = repo.search_semantic("How is user authentication handled?", embed_fn=embed)
for h in hits:
    print(h["file"], h.get("name"))
```

***

## 4. Export full repo index to JSON (file tree + symbols)

```python
repo = Repository("/path/to/project")
repo.write_index("repo_index.json")
```

***

## 5. Find All Callers of a Specific Function (Cross-File)

This recipe helps you understand where a particular function is being used throughout your entire codebase, which is crucial for impact analysis or refactoring.

```python
from kit import Repository


# Initialize the repository
repo = Repository("/path/to/your_project")


# Specify the function name and its type
function_name_to_trace = "my_target_function"


# Find all usages (definitions, calls, imports)
usages = repo.find_symbol_usages(function_name_to_trace, symbol_type="function")


print(f"Usages of function '{function_name_to_trace}':")
for usage in usages:
    file_path = usage.get("file")
    line_number = usage.get("line") # Assuming 'line' is the start line of the usage/symbol
    context_snippet = usage.get("context", "No context available")
    usage_type = usage.get("type", "unknown") # e.g., 'function' for definition, 'call' for a call site


    # We are interested in where it's CALLED, so we might filter out the definition itself if needed,
    # or differentiate based on the 'type' or 'context'.
    # For this example, we'll print all usages.
    if line_number is not None:
        print(f"- Found in: {file_path}:L{line_number + 1}") # (line is 0-indexed, display as 1-indexed)
    else:
        print(f"- Found in: {file_path}")
    print(f"    Type: {usage_type}")
    print(f"    Context: {context_snippet.strip()}\n")


# Example: Filtering for actual call sites (heuristic based on context or type if available)
# print(f"\nCall sites for function '{function_name_to_trace}':")
# for usage in usages:
#     # This condition might need refinement based on what 'find_symbol_usages' returns for 'type' of a call
#     if usage.get("type") != "function" and function_name_to_trace + "(" in usage.get("context", ""):
#         file_path = usage.get("file")
#         line_number = usage.get("line")
#         print(f"- Call in: {file_path}:L{line_number + 1 if line_number is not None else 'N/A'}")
```

***

## 6. Identify Potentially Unused Functions (Heuristic)

This recipe provides a heuristic to find functions that *might* be unused within the analyzed codebase. This can be a starting point for identifying dead code. Note that this is a heuristic because it might not catch dynamically called functions, functions part of a public API but not used internally, or functions used only in parts of the codebase not analyzed (e.g., separate test suites).

```python
from kit import Repository


repo = Repository("/path/to/your_project")


# Get all symbols from the repository index
# The structure of repo.index() might vary; assuming it's a dict like {'symbols': {'file_path': [symbol_dicts]}}
# or a direct way to get all function definitions.
# For this example, let's assume we can iterate through all symbols and filter functions.


# A more robust way might be to iterate files, then symbols within files from repo.index()
# index = repo.index()
# all_symbols_by_file = index.get("symbols", {})


print("Potentially unused functions:")


# First, get a list of all function definitions
defined_functions = []
repo_index = repo.index() # Assuming this fetches file tree and symbols
symbols_map = repo_index.get("symbols", {})


for file_path, symbols_in_file in symbols_map.items():
    for symbol_info in symbols_in_file:
        if symbol_info.get("type") == "function":
            defined_functions.append({
                "name": symbol_info.get("name"),
                "file": file_path,
                "line": symbol_info.get("line_start", 0) # or 'line'
            })


for func_def in defined_functions:
    function_name = func_def["name"]
    definition_file = func_def["file"]
    definition_line = func_def["line"]


    if not function_name: # Skip if name is missing
        continue


    usages = repo.find_symbol_usages(function_name, symbol_type="function")


    # Filter out the definition itself from the usages to count actual calls/references
    # This heuristic assumes a usage is NOT the definition if its file and line differ,
    # or if the usage 'type' (if available and detailed) indicates a call.
    # A simpler heuristic: if only 1 usage, it's likely just the definition.


    actual_references = []
    for u in usages:
        # Check if the usage is different from the definition site
        if not (u.get("file") == definition_file and u.get("line") == definition_line):
            actual_references.append(u)


    # If a function has no other references apart from its own definition site (or very few)
    # It's a candidate for being unused. The threshold (e.g., 0 or 1) can be adjusted.
    if len(actual_references) == 0:
        print(f"- Function '{function_name}' defined in {definition_file}:L{definition_line + 1} has no apparent internal usages.")


:::caution[Limitations of this heuristic:]
**Limitations of this heuristic:**


*   **Dynamic Calls:** Functions called dynamically (e.g., through reflection, or if the function name is constructed from a string at runtime) won't be detected as used.
*   **Public APIs:** Functions intended for external use (e.g., library functions) will appear unused if the analysis is limited to the library's own codebase.
*   **Test Code:** If your test suite is separate and not part of the `Repository` path being analyzed, functions used only by tests might be flagged.
*   **Object Methods:** The `symbol_type="function"` might need adjustment or further logic if you are also looking for unused *methods* within classes, as their usage context is different.
:::
```

# Build an AI PR Reviewer

Tip

**Looking for a ready-to-use PR reviewer?** Kit includes a production-ready AI PR reviewer available via `kit review`. See the [CLI documentation](/introduction/cli#pr-review-operations) for setup and usage. This tutorial shows you how to build your own custom reviewer using kit’s components.

`kit` shines when an LLM needs to *understand a change in the context of the **entire** code-base*—exactly what a human reviewer does. A good review often requires looking beyond the immediate lines changed to understand their implications, check for consistency with existing patterns, and ensure no unintended side-effects arise. This tutorial walks through a **minimal but complete** AI PR-review bot that demonstrates how `kit` provides this crucial whole-repo context.

1. Fetches a GitHub PR (diff + metadata).
2. Builds a `kit.Repository` for the **changed branch** so we can query *any* file, symbol or dependency as it exists in that PR.
3. Generates a focused context bundle with `kit.llm_context.ContextAssembler`, which intelligently combines the diff, the full content of changed files, relevant neighboring code, and even semantically similar code from elsewhere in the repository.
4. Sends the bundle to an LLM and posts the comments back to GitHub.

By the end you will see how a few dozen lines of Python—plus `kit`—give your LLM the *whole-repo* superpowers, enabling it to perform more insightful and human-like code reviews.

## 1. Fetch PR data

To start, our AI reviewer needs the raw materials of the pull request.

Use the GitHub REST API to grab the *diff* **and** the PR-head **commit SHA**:

```python
import os, requests


def fetch_pr(repo, pr_number):
    """Return the PR's unified diff **and** head commit SHA."""
    token = os.getenv("GITHUB_TOKEN")
    url   = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"


    # 1) Unified diff
    diff_resp = requests.get(
        url,
        headers={
            "Accept": "application/vnd.github.v3.diff",
            "Authorization": f"token {token}",
        },
        timeout=15,
    )
    diff_resp.raise_for_status()
    diff = diff_resp.text


    # 2) JSON metadata (for head SHA, title, description, …)
    meta_resp = requests.get(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"token {token}",
        },
        timeout=15,
    )
    meta_resp.raise_for_status()
    pr_info = meta_resp.json()


    head_sha = pr_info["head"]["sha"]
    return diff, head_sha
```

***

## 2. Create a `Repository` for the PR branch

With the `head_sha` obtained, we **ideally** want to load the repository *at that exact commit*. Today, `kit.Repository` will clone the **default branch** of a remote repository (usually `main`) when you pass a URL. If you need the precise PR-head commit you have two options:

1. Clone the repo yourself, `git checkout <head_sha>`, and then point `Repository` at that local path.
2. Call `Repository(url)` to fetch the default branch **and** apply the PR diff in memory (as we do later in this tutorial). For many review tasks this is sufficient because the changed files still exist on `main`, and the diff contains the exact edits.

Direct `ref=`/commit checkout support is coming shortly.

So for now we’ll simply clone the default branch and rely on the diff for any code that hasn’t been pushed upstream:

```python
from kit import Repository


repo = Repository(
    path_or_url="https://github.com/OWNER/REPO.git", # Replace with actual repo URL
    github_token=os.getenv("GITHUB_TOKEN"),
    cache_dir="~/.cache/kit",  # clones are cached for speed
)
```

The `cache_dir` parameter tells `kit` where to store parts of remote repositories it fetches. This caching significantly speeds up subsequent operations on the same repository or commit, which is very beneficial for a bot that might process multiple PRs or re-analyze a PR if it’s updated.

Now `repo` can *instantly* answer questions like: `repo.search_text("TODO")` (useful for checking if the PR resolves or introduces to-do items), `repo.extract_symbols('src/foo.py')` (to understand the structure of a changed file), `repo.find_symbol_usages('User')` (to see how a modified class or function is used elsewhere, helping to assess the impact of changes). These capabilities allow our AI reviewer to gather rich contextual information far beyond the simple diff.

***

## 3. Build context for the LLM

The `ContextAssembler` is the workhorse for preparing the input to the LLM. It orchestrates several `kit` features to build a comprehensive understanding of the PR:

```python
from kit import Repository
from unidiff import PatchSet
from sentence_transformers import SentenceTransformer


# Assume `repo`, `diff`, `pr_title`, `pr_description` are defined
# `diff` is the raw diff string
# `pr_title`, `pr_description` are strings from your PR metadata


# -------------------------------------------------
# 1) Build or load the semantic index so search_semantic works
# -------------------------------------------------
st_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda text: st_model.encode(text).tolist()


vs = repo.get_vector_searcher(embed_fn)
vs.build_index()  # do this once; subsequent runs can skip if cached


# -------------------------------------------------
# 2) Assemble context for the LLM
# -------------------------------------------------
assembler = repo.get_context_assembler()
patch = PatchSet(diff)


# Add the raw diff
assembler.add_diff(diff)


# Add full content of changed / added files – with safety guards
LOCK_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Cargo.lock",
    "composer.lock",
}


for p_file in patch:
    if p_file.is_removed_file:
        continue  # nothing to embed


    assembler.add_file(
        p_file.path,
        max_lines=400,              # Inline only if the file is reasonably small
        skip_if_name_in=LOCK_FILES, # Skip bulky lock files entirely (diff already added)
    )


# Semantic search for related code using PR title/description
for q in filter(None, [pr_title, pr_description]):
    q = q.strip()
    if not q:
        continue
    hits = repo.search_semantic(q, top_k=3, embed_fn=embed_fn)
    assembler.add_search_results(hits, query=f"Code semantically related to: '{q}'")


context_blob = assembler.format_context()
```

The `ContextAssembler` is used as follows:

1. **`assembler.add_diff(diff)`**: This provides the LLM with the direct changes from the PR.

2. **`assembler.add_file(p_file.path)`**: Supplying the full content of changed files allows the LLM to see modifications in their complete original context, not just the diff hunks.

3. **Augment with Semantic Search (`assembler.add_search_results(...)`)**: This is a key step where `kit` truly empowers the AI reviewer. Beyond direct code connections, `kit`’s `repo.search_semantic()` method can unearth other code sections that are *conceptually related* to the PR’s intent, even if not directly linked by calls or imports.

   You can use queries derived from the PR’s title or description to find examples of similar functionality, relevant design patterns, or areas that might require parallel updates.

   **The Power of Summaries**: While `repo.search_semantic()` can operate on raw code, its effectiveness is significantly amplified when your `Repository` instance is configured with a `DocstringIndexer`. The `DocstringIndexer` (see the [Docstring Search Tutorial](/tutorials/docstring_search)) preprocesses your codebase, generating AI summaries for files or symbols. When `repo.search_semantic()` leverages this index, it matches based on the *meaning and purpose* captured in these summaries, leading to far more relevant and insightful results than simple keyword or raw-code vector matching. This allows the AI reviewer to understand context like “find other places where we handle user authentication” even if the exact phrasing or code structure varies.

   The Python code snippet above illustrates how you might integrate this. Remember to ensure your `repo` object is properly set up with an embedding function and, for best results, a `DocstringIndexer`. Refer to the “[Docstring Search](/tutorials/docstring_search)” and “[Semantic Code Search](/tutorials/semantic_code_search)” tutorials for detailed setup guidance.

Finally, `assembler.format_context()` consolidates all the added information into a single string (`context_blob`), ready to be sent to the LLM. This step might also involve applying truncation or specific formatting to optimise for the LLM’s input requirements.

***

## 4. Prepare the LLM Prompt

With the meticulously assembled `context_blob` from `kit`, we can now prompt an LLM. The quality of the prompt—including the system message that sets the LLM’s role and the user message containing the context—is vital. Because `kit` has provided such comprehensive and well-structured context, the LLM is significantly better equipped to act like an “expert software engineer” and provide a nuanced, insightful review.

````python
from openai import OpenAI


client = OpenAI()
msg = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.2,
    messages=[
        {"role": "system", "content": "You are an expert software engineer …"},
        {"role": "user",   "content": f"PR context:\n```\n{context_blob}\n```\nGive a review."},
    ],
)
review = msg.choices[0].message.content.strip()
````

***

## 5. Post the review back to GitHub

This final step completes the loop by taking the LLM’s generated review and posting it as a comment on the GitHub pull request. This delivers the AI’s insights directly to the developers, integrating the AI reviewer into the existing development workflow.

```python
requests.post(
    f"https://api.github.com/repos/{repo_full}/issues/{pr_number}/comments",
    headers={
        "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github.v3+json",
    },
    json={"body": review},
    timeout=10,
).raise_for_status()
```

***

## Where to go next?

This tutorial provides a foundational AI PR reviewer. `kit`’s components can help you extend it further:

* **Chunk large diffs or files** – If a PR is very large, the `ContextAssembler` currently adds full content. You might need strategies to chunk very large files (e.g. `repo.chunk_file_by_symbols`) or diffs, or implement more granular context addition to stay within LLM limits.
* **Custom ranking** – The `ContextAssembler` could be configured or extended to allow different weights for various context pieces (e.g. prioritising semantic-search matches that are highly relevant over less critical information). `kit`’s search results, which include scores, can inform this process.
* **Inline comments** – Parse the LLM’s output to identify suggestions pertaining to specific files and lines, then use GitHub’s *review* API to post comments directly on the diff. `kit`’s symbol mapping (line numbers from `RepoMapper`) is crucial here.
* **Supersonic** – For more advanced automation, tools like Supersonic could leverage `kit`’s understanding to not just *suggest* but also *apply* LLM-recommended changes, potentially opening follow-up PRs.

> With `kit` your LLM sees code the way *humans* do: in the rich context of the entire repository. Better signal in → better reviews out.

# Codebase Summarizer

This tutorial shows how to use `kit` to generate a structured Markdown summary of your codebase, including the file tree and all extracted symbols (functions, classes, etc.). Such a summary can be invaluable for quickly understanding a new project, tracking architectural changes, or for documentation purposes.

## Step 1: Summarize the Codebase Structure

The core of this task lies in using `kit`’s `Repository` object to analyze the codebase and extract its structural information. This process involves two main `kit` operations:

1. **Initializing the `Repository`**: `repo = Repository(repo_path)` creates an instance that points `kit` to the target codebase. `kit` then becomes aware of the repository’s location and is ready to perform various analyses.
2. **Indexing the Repository**: `index = repo.index()` is a key `kit` command. When called, `kit` (typically using its internal `RepoMapper` component) traverses the repository, parses supported source files, identifies structural elements like files, directories, classes, functions, and other symbols. It then compiles this information into a structured `index`.

Use kit’s `Repo` object to index the codebase and gather all relevant information.

```python
from kit import Repository


def summarize_codebase(repo_path: str) -> str:
    repo = Repository(repo_path)
    index = repo.index()
    lines = [f"# Codebase Summary for {repo_path}\n"]
    lines.append("## File Tree\n")
    # The index['file_tree'] contains a list of file and directory paths
    for file_info in index["file_tree"]:
        # Assuming file_info is a dictionary or string representing the path
        # Adjust formatting based on the actual structure of file_info objects from repo.index()
        if isinstance(file_info, dict):
            lines.append(f"- {file_info.get('path', file_info.get('name', 'Unknown file/dir'))}")
        else:
            lines.append(f"- {file_info}") # Fallback if it's just a string path


    lines.append("\n## Symbols\n")
    # The index['symbols'] is typically a dictionary where keys are file paths
    # and values are lists of symbol information dictionaries for that file.
    for file_path, symbols_in_file in index["symbols"].items():
        lines.append(f"### {file_path}")
        for symbol in symbols_in_file:
            # Each symbol dict contains details like 'type' (e.g., 'function', 'class') and 'name'.
            lines.append(f"- **{symbol['type']}** `{symbol['name']}`")
        lines.append("")
    return "\n".join(lines)
```

This function, `summarize_codebase`, first initializes `kit` for the given `repo_path`. Then, `repo.index()` does the heavy lifting of analyzing the code. The resulting `index` object is a dictionary, typically containing at least two keys:

* `'file_tree'`: A list representing the directory structure and files within the repository.
* `'symbols'`: A dictionary where keys are file paths, and each value is a list of symbols found in that file. Each symbol is itself a dictionary containing details like its name and type (e.g., function, class).

The rest of the function iterates through this structured data to format it into a human-readable Markdown string.

***

## Step 2: Command-Line Interface

To make the summarizer easy to use from the terminal, we’ll add a simple command-line interface using Python’s `argparse` module. This allows the user to specify the repository path and an optional output file for the summary.

Provide CLI arguments for repo path and output file:

```python
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Codebase summarizer using kit.")
    parser.add_argument("--repo", required=True, help="Path to the code repository")
    parser.add_argument("--output", help="Output Markdown file (default: stdout)")
    args = parser.parse_args()
    summary = summarize_codebase(args.repo)
    if args.output:
        with open(args.output, "w") as f:
            f.write(summary)
        print(f"Summary written to {args.output}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
```

***

## Step 3: Running the Script

With the script in place, you can execute it from your terminal. You’ll need to provide the path to the repository you want to summarize and, optionally, a path where the Markdown output file should be saved. If no output file is specified, the summary will be printed to the console.

Run the summarizer like this:

```sh
python codebase_summarizer.py --repo /path/to/repo --output summary.md
```

***

## Example Output

The generated Markdown file (or console output) will provide a clear, structured overview of your project, derived directly from `kit`’s analysis. It will list the files and then, for each file, the symbols defined within it.

```plaintext
# Codebase Summary for /path/to/repo


## File Tree
- main.py
- utils.py
- models/
- models/model.py


## Symbols
### main.py
- **function** `main`
- **class** `App`


### utils.py
- **function** `helper`
```

# Codebase Q&A Bot with Summaries

This tutorial demonstrates how to build a simple question-answering bot for your codebase. The bot will:

1. Use `DocstringIndexer` to create semantic summaries of each file.
2. When a user asks a question, use `SummarySearcher` to find relevant file summaries.
3. Fetch the full code of those top files.
4. Use `ContextAssembler` to build a concise prompt for an LLM.
5. Get an answer from the LLM.

This approach is powerful because it combines semantic understanding (from summaries) with the full detail of the source code, allowing an LLM to answer nuanced questions.

## Prerequisites

* You have `kit` installed (`pip install cased-kit`).
* You have an OpenAI API key set (`export OPENAI_API_KEY=...`).
* You have a local Git repository you want to query.

## Steps

1. **Initialize Components**

   First, let’s set up our `Repository`, `DocstringIndexer`, `Summarizer` (for the indexer), `SummarySearcher`, and `ContextAssembler`.

   ```python
   from kit import Repository, DocstringIndexer, Summarizer, SummarySearcher, ContextAssembler
   from kit.summaries import OpenAIConfig


   # --- Configuration ---
   REPO_PATH = "/path/to/your/local/git/repo" #! MODIFY
   # For DocstringIndexer, persist_dir is where the DB is stored.
   # Let's use a directory for ChromaDB as it might create multiple files.
   INDEX_PERSIST_DIR = "./my_code_qa_index_db/"


   # Use a specific summarizer model for indexing, can be different from Q&A LLM
   INDEXER_LLM_CONFIG = OpenAIConfig(model="gpt-4o")


   # LLM for answering the question based on context
   QA_LLM_CONFIG = OpenAIConfig(model="gpt-4o") # Or your preferred model
   # MAX_CONTEXT_CHARS is not directly used by ContextAssembler in this simplified flow
   # TOP_K_SUMMARIES = 3 remains relevant for SummarySearcher
   TOP_K_SUMMARIES = 3
   # --- END Configuration ---


   repo = Repository(REPO_PATH)


   # For DocstringIndexer - requires repo and a summarizer instance
   summarizer_for_indexing = Summarizer(repo=repo, config=INDEXER_LLM_CONFIG)
   indexer = DocstringIndexer(repo, summarizer_for_indexing, persist_dir=INDEX_PERSIST_DIR)


   # For SummarySearcher - get it from the indexer
   searcher = indexer.get_searcher()


   # For assembling context for the Q&A LLM
   assembler = ContextAssembler(repo)


   # We'll need an LLM client to ask the final question
   qa_llm_client = Summarizer(repo=repo, config=QA_LLM_CONFIG)._get_llm_client()
   print("Components initialized.")
   ```

   Make sure to replace `"/path/to/your/local/git/repo"` with the actual path to your repository. Also ensure the directory for `INDEX_PERSIST_DIR` (e.g., `my_code_qa_index_db/`) can be created.

2. **Build or Load the Index**

   The `DocstringIndexer` needs to process your repository to create summaries and embed them. This can take time for large repositories. We’ll check if an index already exists and build it if not.

   ```python
   import os


   # Check based on persist_dir for the indexer
   if not os.path.exists(INDEX_PERSIST_DIR) or not any(os.scandir(INDEX_PERSIST_DIR)):
       print(f"Index not found or empty at {INDEX_PERSIST_DIR}. Building...")
       # Build a symbol-level index for more granular results
       # force=True will rebuild if the directory exists but is perhaps from an old run
       indexer.build(level="symbol", file_extensions=[".py", ".js", ".md"], force=True)
       print("Symbol-level index built successfully.")
   else:
       print(f"Found existing index at {INDEX_PERSIST_DIR}.")
   ```

3. **Define the Question-Answering Function**

   This function will orchestrate the search, context assembly, and LLM query.

   ```python
   def answer_question(user_query: str) -> str:
       print(f"\nSearching for files/symbols relevant to: '{user_query}'")
       # 1. Search for relevant file/symbol summaries
       search_results = searcher.search(user_query, top_k=TOP_K_SUMMARIES)


       if not search_results:
           return "I couldn't find any relevant files or symbols in the codebase to answer your question."


       print(f"Found {len(search_results)} relevant document summaries.")
       # Reset assembler for each new question to start with fresh context
       current_question_assembler = ContextAssembler(repo)


       # 2. Add relevant context to the assembler
       added_content_identifiers = set() # To avoid adding the same file multiple times if symbols from it are retrieved


       for i, res in enumerate(search_results):
           file_path = res.get('file_path')
           identifier_for_log = file_path


           if res.get('level') == 'symbol':
               symbol_name = res.get('symbol_name', 'Unknown Symbol')
               symbol_type = res.get('symbol_type', 'Unknown Type')
               identifier_for_log = f"Symbol: {symbol_name} in {file_path} (Type: {symbol_type})"


           print(f"  {i+1}. {identifier_for_log} (Score: {res.get('score', 0.0):.4f})")


           # For simplicity, add the full file content for any relevant file found,
           # whether the hit was file-level or symbol-level.
           # A more advanced version could add specific symbol code using a custom method.
           if file_path and file_path not in added_content_identifiers:
               try:
                   # Add full file content for context
                   current_question_assembler.add_file(file_path)
                   added_content_identifiers.add(file_path)
                   print(f"    Added content of {file_path} to context.")
               except FileNotFoundError:
                   print(f"    Warning: File {file_path} not found when trying to add to context.")
               except Exception as e:
                   print(f"    Warning: Error adding {file_path} to context: {e}")


       if not added_content_identifiers:
            return "Found relevant file/symbol names, but could not retrieve their content for context."


       # 3. Get the assembled context string
       prompt_context = current_question_assembler.format_context()


       if not prompt_context.strip():
           return "Could not assemble any context for the LLM based on search results."


       # 4. Formulate the prompt and ask the LLM
       system_message = (
           "You are a helpful AI assistant with expertise in the provided codebase. "
           "Answer the user's question based *only* on the following code context. "
           "If the answer is not found in the context, say so. Be concise."
       )
       final_prompt = f"## Code Context:\n\n{prompt_context}\n\n## User Question:\n\n{user_query}\n\n## Answer:"


       print("\nSending request to LLM...")


       # Assuming OpenAI client for this example structure
       # Adapt if using Anthropic or Google
       if isinstance(QA_LLM_CONFIG, OpenAIConfig):
           response = qa_llm_client.chat.completions.create(
               model=QA_LLM_CONFIG.model,
               messages=[
                   {"role": "system", "content": system_message},
                   {"role": "user", "content": final_prompt}
               ]
           )
           answer = response.choices[0].message.content
       # Add elif for AnthropicConfig, GoogleConfig if desired, or abstract further
       else:
           # Simplified fallback or placeholder for other LLMs
           # In a real app, you'd implement the specific API calls here
           raise NotImplementedError(f"LLM client for {type(QA_LLM_CONFIG)} not fully implemented in this example.")


       return answer
   ```

4. **Ask a Question!**

   Now, let’s try it out.

   ```python
   my_question = "How does the authentication middleware handle expired JWTs?"
   # Or try: "What's the main purpose of the UserNotifications class's send_email method?"
   # Or: "Where is the database connection retry logic implemented in the db_utils module?"


   llm_answer = answer_question(my_question)
   print(f"\nLLM's Answer:\n{llm_answer}")
   ```

   Example Output (will vary based on your repo & LLM)

   ```text
   Components initialized.
   Found existing index at ./my_code_qa_index_db/.


   Searching for files/symbols relevant to: 'How does the authentication middleware handle expired JWTs?'
   Found 3 relevant document summaries.
     1. Symbol: authenticate in src/auth/middleware.py (Type: function, Score: 0.8765)
     2. File: src/utils/jwt_helpers.py (Score: 0.7912)
     3. File: tests/auth/test_middleware.py (Score: 0.7500)


   Sending request to LLM...


   LLM's Answer:
   The `authenticate` function in `src/auth/middleware.py` checks for JWT expiration. If an `ExpiredSignatureError` is caught during token decoding (likely using a helper from `src/utils/jwt_helpers.py`), it returns a 401 Unauthorized response, typically with a JSON body like `{"error": "Token expired"}`.
   ```

# Dependency Graph Visualizer in Python

This tutorial demonstrates how to visualize the dependency graph of a codebase using `kit`. `kit`’s `DependencyAnalyzer` supports analyzing dependencies in both Python and Terraform codebases, and can output the graph in DOT format, which you can render with Graphviz to generate visual diagrams.

For Python codebases, the analyzer leverages Abstract Syntax Tree (AST) parsing to track import relationships between modules. For Terraform codebases, it analyzes resource references and dependencies between infrastructure components.

Note

To directly generate an image (e.g., PNG, SVG) without manually handling DOT files, you can use the `analyzer.visualize_dependencies()` method, provided you have both the `graphviz` Python package and the Graphviz system executables installed.

## Step 1: Generate the Dependency Graph in DOT Format

The `kit.Repository` object provides access to a `DependencyAnalyzer` which can build and export the dependency graph.

```python
from kit import Repository


def generate_dot_dependency_graph(repo_path: str) -> str:
    """
    Initializes a Repository, gets its DependencyAnalyzer,
    and exports the dependency graph in DOT format.
    """
    repo = Repository(repo_path)
    # Specify the language ('python' or 'terraform')
    analyzer = repo.get_dependency_analyzer('python')


    # The build_dependency_graph() method is called implicitly by export_dependency_graph
    # if the graph hasn't been built yet.
    dot_output = analyzer.export_dependency_graph(output_format='dot')


    # Ensure dot_output is a string. export_dependency_graph returns the content
    # when output_path is None.
    if not isinstance(dot_output, str):
        # This case should ideally not happen if output_format='dot' and output_path=None
        # based on typical implementations, but good to be defensive.
        raise TypeError(f"Expected DOT output as string, got {type(dot_output)}")


    return dot_output
```

This function `generate_dot_dependency_graph`:

1. Initializes the `Repository` for the given `repo_path`.
2. Gets a `DependencyAnalyzer` instance from the repository.
3. Calls `analyzer.export_dependency_graph(output_format='dot')` to get the graph data as a DOT formatted string.

## Step 2: Command-Line Interface

Add CLI arguments for the repository path and an optional output file for the DOT content.

```python
import argparse


# Assume generate_dot_dependency_graph function from Step 1 is defined above


def main() -> None:
    parser = argparse.ArgumentParser(description="Dependency graph visualizer using kit.")
    parser.add_argument("--repo", required=True, help="Path to the code repository")
    parser.add_argument("--output", help="Output DOT file (default: stdout)")
    args = parser.parse_args()


    try:
        dot_content = generate_dot_dependency_graph(args.repo)
        if args.output:
            with open(args.output, "w") as f:
                f.write(dot_content)
            print(f"Dependency graph (DOT format) written to {args.output}")
        else:
            print(dot_content)
    except Exception as e:
        print(f"An error occurred: {e}")
        # For more detailed debugging, you might want to print the traceback
        # import traceback
        # traceback.print_exc()


if __name__ == "__main__":
    main()
```

## Step 3: Running the Script

Run the visualizer script from your terminal. Provide the path to the repository and optionally an output file name for the DOT data.

```sh
python your_script_name.py --repo /path/to/your/python_project --output project_deps.dot
```

Replace `your_script_name.py` with the actual name of your Python file containing the code from Steps 1 and 2.

## Step 4: Rendering the Graph

To visualize the generated DOT file, you need Graphviz installed on your system. Use the `dot` command-line tool:

```sh
dot -Tpng project_deps.dot -o project_deps.png
```

This will create a PNG image (`project_deps.png`) of your codebase’s import relationships.

## Extending the Visualizer

`kit`’s `DependencyAnalyzer` offers more than just DOT export:

* **Direct Visualization**: Use `analyzer.visualize_dependencies(output_path="graph_image_prefix", format="png")` to directly save an image (requires the `graphviz` Python library).

* **Other Export Formats**: Export to JSON, GraphML, or an adjacency list using `analyzer.export_dependency_graph(output_format=...)`.

* **Cycle Detection**: Use `analyzer.find_cycles()` to identify circular dependencies.

* **Querying the Graph**:

  * For Python: Use `analyzer.get_module_dependencies()` and `analyzer.get_dependents()` to explore module relationships.
  * For Terraform: Use `analyzer.get_resource_dependencies()` and `analyzer.get_resource_by_type()` to explore infrastructure dependencies.

* **Reports and Context**:

  * Generate a comprehensive JSON report with `analyzer.generate_dependency_report()`.
  * Create LLM-friendly context with `analyzer.generate_llm_context()`.

* **File Paths**: The analyzer tracks absolute file paths for each component, making it easy to locate resources in complex projects.

## Using with Terraform

To analyze a Terraform codebase instead of Python, simply specify ‘terraform’ when getting the analyzer:

```python
analyzer = repo.get_dependency_analyzer('terraform')
```

The Terraform analyzer will map dependencies between resources, modules, variables, and other Terraform components. All resources in the graph include their absolute file paths, making it easy to locate them in complex infrastructure projects with files that might have the same name in different directories.

## Conclusion

Visualizing dependencies helps you understand, refactor, and document complex codebases. With `kit`’s `DependencyAnalyzer` and tools like Graphviz, you can gain valuable insights into your project’s structure, whether it’s a Python application or Terraform infrastructure.

# Build a Docstring Search Engine

In this tutorial you’ll build a semantic search tool on top of `kit` using **docstring-based indexing**.

Why docstrings? Summaries distill *intent* rather than syntax. Embedding these short natural-language strings lets the vector DB focus on meaning, giving you relevant hits even when the literal code differs (e.g., `retry()` vs `attempt_again()`). It also keeps the index small (one embedding per file or symbol instead of dozens of raw-code chunks).

***

## 1. Install dependencies

```bash
uv pip install kit sentence-transformers chromadb
```

## 2. Initialise a repo and summarizer

```python
import kit
from kit import Repository, DocstringIndexer, Summarizer, SummarySearcher
from sentence_transformers import SentenceTransformer


REPO_PATH = "/path/to/your/project"
repo = Repository(REPO_PATH)


summarizer = repo.get_summarizer()  # defaults to OpenAIConfig
```

## 3. Build the docstring index

```python
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda txt: embed_model.encode(txt).tolist()


indexer = DocstringIndexer(repo, summarizer, embed_fn)
indexer.build()          # writes REPO_PATH/.kit_cache/docstring_db
```

The first run will take time depending on repo size and LLM latency. Summaries are cached inside the vector DB (and in a meta.json within the persist\_dir), so subsequent runs are cheap if code hasn’t changed.

## 4. Query the index

```python
searcher = indexer.get_searcher()


results = searcher.search("How is the retry back-off implemented?", top_k=3)
for hit in results:
    print(f"→ File: {hit.get('file_path', 'N/A')}\n  Summary: {hit.get('summary', 'N/A')}")
```

You now have a semantic code searcher, using powerful docstring summaries, as easy as that.

# Dump Repo Map

This tutorial explains how to use `kit` to dump a complete map of your repository—including the file tree and all extracted symbols—as a JSON file. This is useful for further analysis, visualization, or integration with other tools. `kit` provides a convenient method on the `Repository` object to achieve this directly.

## Step 1: Create the Script

Create a Python script named `dump_repo_map.py` with the following content. This script uses `argparse` to accept the repository path and the desired output file path.

dump\_repo\_map.py

```python
from kit import Repository # Import the main Repository class
import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Dump a repository's file tree and symbols as JSON using kit.")
    parser.add_argument("repo_path", help="Path to the repository directory.")
    parser.add_argument("output_file", help="Path to the output JSON file.")
    args = parser.parse_args()


    repo_path = args.repo_path
    if not os.path.isdir(repo_path):
        print(f"Error: Repository path not found or not a directory: {repo_path}", file=sys.stderr)
        sys.exit(1)


    try:
        print(f"Initializing repository at: {repo_path}", file=sys.stderr)
        repo = Repository(repo_path)


        print(f"Dumping repository index to: {args.output_file}", file=sys.stderr)
        repo.write_index(args.output_file) # Use the direct method


        print(f"Successfully wrote repository map to {args.output_file}", file=sys.stderr)
    except Exception as e:
        print(f"Error processing repository: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

***

## Step 2: Run the Script

Save the code above as `dump_repo_map.py`. You can then run it from your terminal, providing the path to the repository you want to map and the desired output file name:

```sh
python dump_repo_map.py /path/to/repo repo_map.json
```

This will create a JSON file (e.g., `repo_map.json`) containing the structure and symbols of your codebase.

***

## Example JSON Output

The output JSON file will contain a `file_tree` (also aliased as `files`) and a `symbols` map.

```json
{
  "file_tree": [
    {
      "path": "src",
      "is_dir": true,
      "name": "src",
      "size": 0
    },
    {
      "path": "src/main.py",
      "is_dir": false,
      "name": "main.py",
      "size": 1024
    },
    {
      "path": "README.md",
      "is_dir": false,
      "name": "README.md",
      "size": 2048
    }
    // ... more files and directories
  ],
  "files": [
    // ... same content as file_tree ...
  ],
  "symbols": {
    "src/main.py": [
      {
        "type": "function",
        "name": "main",
        "start_line": 10,
        "end_line": 25,
        "code": "def main():\n  pass"
      },
      {
        "type": "class",
        "name": "App",
        "start_line": 30,
        "end_line": 55
      }
    ],
    "src/utils.py": [
      {
        "type": "function",
        "name": "helper",
        "start_line": 5,
        "end_line": 12
      }
    ]
    // ... more files and their symbols
  }
}
```

Note

The exact content and structure of symbol information (e.g., inclusion of `code` snippets) depends on the `RepoMapper`’s symbol extraction capabilities for the specific languages in your repository.

***

## Integration Ideas

* Use the JSON output to feed custom dashboards or documentation tools.
* Integrate with code search or visualization tools.
* Use for code audits, onboarding, or automated reporting.

***

## Conclusion

With `kit`, you can easily export a structured map of your repository using `repo.write_index()`, making this data readily available for various downstream use cases and custom tooling.

# Exploring Kit Interactively with the Python Shell

> A hands-on guide to trying out Kit's features directly in a Python interpreter.

This guide walks you through interactively exploring the `kit` library’s capabilities using a Python shell. This is a great way to quickly understand how different components work, test out methods, and see the structure of the data they return.

## Prerequisites

Before you begin, ensure you have:

1. Cloned the `kit` repository.

2. Set up your Python environment and installed `kit`’s dependencies. Ideally, you’ve installed `kit` in editable mode if you’re also making changes:

   ```bash
   pip install -e .
   ```

3. (Optional but recommended) Familiarized yourself with the [Core Concepts](/core-concepts/introduction) of `kit`.

## Getting Started: Your First Exploration

Let’s dive in! We’ll start by instantiating the `Repository` class and trying out some of its basic methods.

1. **Launch your Python Interpreter**

   Open your terminal and start Python:

   ```bash
   python
   # or python3
   ```

2. **Import `Repository` and Initialize**

   The `Repository` class is your main entry point for interacting with a codebase.

   ```python
   from kit.repository import Repository
   import os # We'll use this for path joining


   # Replace with the absolute path to your local clone of the 'kit' repository (or any other repo)
   # For example, if you are in the root of the 'kit' repo itself:
   repo_path = os.path.abspath(".")
   # Or provide a full path directly:
   # repo_path = "/path/to/your/repository"


   repo = Repository(repo_path)
   print(repo)
   # This should print something like: <Repository path=/path/to/your/repository, branch=main, files=XX>
   ```

   This confirms your `Repository` object is ready.

## Extracting Symbols from a File

One of the core features of `kit` is its ability to parse source code and extract meaningful symbols like classes, functions, and methods. The `repo.extract_symbols()` method is used for this. After recent updates, this method now provides the full source code for each symbol and the correct line numbers spanning the entire symbol definition.

1. **Choose a File and Extract Symbols**

   Let’s try extracting symbols from the `src/kit/repository.py` file itself.

   ```python
   # Assuming 'repo' is your Repository instance from the previous step
   # and 'os' is imported.


   file_to_test_relative = "src/kit/repository.py"
   full_file_path = os.path.join(repo.repo_path, file_to_test_relative)


   print(f"Extracting symbols from: {full_file_path}")
   symbols_in_repo_py = repo.extract_symbols(full_file_path)


   # You can use pprint for a more readable output of complex objects
   import pprint
   # pprint.pprint(symbols_in_repo_py) # Uncomment to see all symbols
   ```

2. **Inspect a Specific Symbol**

   Let’s look at the first symbol extracted, which should be the `Repository` class itself.

   ```python
   if symbols_in_repo_py:
       repository_class_symbol = None
       for sym in symbols_in_repo_py:
           if sym.get('name') == 'Repository' and sym.get('type') == 'class':
               repository_class_symbol = sym
               break


       if repository_class_symbol:
           print("\n--- Details for 'Repository' class symbol ---")
           print(f"Name: {repository_class_symbol.get('name')}")
           print(f"Type: {repository_class_symbol.get('type')}")
           print(f"Start Line: {repository_class_symbol.get('start_line')}")
           print(f"End Line: {repository_class_symbol.get('end_line')}")
           print(f"File: {repository_class_symbol.get('file')}") # Though we know the file, it's good to see it in the output
           print("\nCode (first ~300 characters):")
           print(repository_class_symbol.get('code', '')[:300] + "...")
           print(f"\n(Full code length: {len(repository_class_symbol.get('code', ''))} characters)")
           print("------")
       else:
           print("Could not find the 'Repository' class symbol.")
   else:
       print(f"No symbols extracted from {file_to_test_relative}")
   ```

   You should see that:

   * The `code` field contains the *entire* source code of the `Repository` class.
   * `start_line` and `end_line` accurately reflect the beginning and end of the class definition.
   * This is a significant improvement, providing much richer data for analysis or use in LLM prompts.

## Listing All Files in the Repository

To get an overview of all files and directories that `kit` recognizes within your repository, you can use the `repo.get_file_tree()` method. This is helpful for understanding the scope of what `kit` will operate on.

1. **Call `get_file_tree()`**

   ```python
   # Assuming 'repo' is your Repository instance


   print("\n--- Getting File Tree ---")
   file_tree = repo.get_file_tree()


   if file_tree:
       print(f"Found {len(file_tree)} files/items in the repository.")
       print("\nFirst 5 items in the file tree:")
       for i, item in enumerate(file_tree[:5]): # Print the first 5 items
           print(f"{i+1}. {item}")
       print("------")


       # Example of what one item might look like:
       # {'path': 'src/kit/repository.py', 'is_dir': False, 'name': 'repository.py', 'size': 14261}
   else:
       print("File tree is empty or could not be retrieved.")
   ```

2. **Understanding the Output**

   The `get_file_tree()` method returns a list of dictionaries. Each dictionary represents a file or directory and typically includes:

   * `'path'`: The relative path from the repository root.
   * `'is_dir'`: `True` if it’s a directory, `False` if it’s a file.
   * `'name'`: The base name of the file or directory.
   * `'size'`: The size in bytes (often 0 for directories in this view).

   This method respects rules defined in `.gitignore` (by default) and gives you a snapshot of the files `kit` is aware of.

## Searching for Text in Files

`kit` allows you to perform text-based searches across your repository, similar to using `grep`. This is handled by the `repo.search_text()` method.

1. **Perform a Search (Default: All Files)**

   Let’s search for the term “app”. By default, `search_text` now looks in all files (`*`).

   ```python
   # Assuming 'repo' is your Repository instance


   print("\n--- Searching for Text ---")
   query_text = "app"
   # The default file_pattern is now "*", so it searches all files
   search_results_all = repo.search_text(query=query_text)


   if search_results_all:
       print(f"Found {len(search_results_all)} occurrences of '{query_text}' in all files.")
       print("\nFirst 3 search results (all files):")
       for i, result in enumerate(search_results_all[:3]):
           print(f"\nResult {i+1}:")
           print(f"  File: {result.get('file')}")
           print(f"  Line Number (0-indexed): {result.get('line_number')}")
           print(f"  Line Content: {result.get('line', '').strip()}")
   else:
       print(f"No occurrences of '{query_text}' found in any files.")
   print("------")
   ```

2. **Search in Specific File Types**

   You can still specify a `file_pattern` to search in specific file types. For example, to search for “Repository” only in Python (`*.py`) files:

   ```python
   query_repo = "Repository"
   pattern_py = "*.py"
   print(f"\nSearching for '{query_repo}' in '{pattern_py}' files...")
   repo_py_results = repo.search_text(query=query_repo, file_pattern=pattern_py)


   if repo_py_results:
       print(f"Found {len(repo_py_results)} occurrences of '{query_repo}' in Python files.")
       print("First result (Python files):")
       first_py_result = repo_py_results[0]
       print(f"  File: {first_py_result.get('file')}")
       print(f"  Line Number (0-indexed): {first_py_result.get('line_number')}")
       print(f"  Line Content: {first_py_result.get('line', '').strip()}")
   else:
       print(f"No occurrences of '{query_repo}' found in '{pattern_py}' files.")
   print("------")
   ```

3. **Understanding the Output**

   `search_text()` returns a list of dictionaries, each representing a match. Key fields include:

   * `'file'`: The path to the file where the match was found.
   * `'line_number'`: The (often 0-indexed) line number of the match.
   * `'line'`: The full content of the line containing the match.
   * `'context_before'` and `'context_after'`: Lists for lines before/after the match (may be empty depending on search configuration).

   Keep in mind this is a literal text search and is case-sensitive by default. It will find the query string as a substring anywhere it appears (e.g., “app” within “mapper” or “happy”).

## Workflow: Get First File’s Content

A common task is to list files, select one, and then retrieve its contents. Here’s a simple workflow to get the content of the first file listed by `get_file_tree()`.

1. **Get File Tree, Pick First File, and Get Content**

   This script finds the first item in the `file_tree` that is a file (not a directory) and prints a snippet of its content.

   ```python
   # Assuming 'repo' is your Repository instance


   print("\n--- Workflow: Get First *File's* Content ---")


   # 1. List all items
   file_tree = repo.get_file_tree()


   first_file_path = None
   if file_tree:
       # 2. Find the path of the first actual file in the tree
       for item in file_tree:
           if not item.get('is_dir', False): # Make sure it's a file
               first_file_path = item['path']
               break # Stop once we've found the first file


   if not first_file_path:
       print("No actual files (non-directories) found in the repository.")
   else:
       print(f"\nPicking the first *file* found in the tree: {first_file_path}")


       # 3. Get its content
       print(f"Attempting to read content from: {first_file_path}")
       try:
           content = repo.get_file_content(first_file_path)
           print(f"\n--- Content of {first_file_path} (first 300 chars) ---")
           print(content[:300] + "..." if len(content) > 300 else content)
           print(f"------ End of {first_file_path} snippet ------")
       except FileNotFoundError:
           print(f"Error: File not found at '{first_file_path}'.")
       except Exception as e:
           print(f"An unexpected error occurred: {e}")
   print("------")
   ```

2. **Example Output**

   If the first file in your repository is `LICENSE`, the output might look like:

   ```text
   --- Workflow: Get First *File's* Content ---


   Picking the first *file* found in the tree: LICENSE
   Attempting to read content from: LICENSE


   --- Content of LICENSE (first 300 chars) ---
   MIT License


   Copyright (c) 2024 Cased


   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, dis...
   ------ End of LICENSE snippet ------
   ------
   ```

   This demonstrates successfully using `get_file_tree()` to discover a file and `get_file_content()` to read it.

### Chunking File Content by Lines (`repo.chunk_file_by_lines()`)

The `Repository` class provides a method to break down a file’s content into smaller string chunks based on a target maximum number of lines. This is useful for preprocessing text for Large Language Models (LLMs) or other tools that have input size limits.

The method signature is: `repo.chunk_file_by_lines(file_path: str, max_lines: int = 50) -> List[str]`

* `file_path`: The relative path to the file within the repository.
* `max_lines`: The desired maximum number of lines for each chunk. The actual number of lines in a chunk might vary slightly as the method attempts to find reasonable break points.
* It returns a list of strings, where each string is a content chunk.

**Example 1: Chunking a small file (e.g., `LICENSE`)**

If the file is smaller than `max_lines`, it will be returned as a single chunk.

```python
license_path = "LICENSE"
license_chunks = repo.chunk_file_by_lines(license_path)


print(f"Number of chunks for {license_path}: {len(license_chunks)}")
if license_chunks:
    print(f"Content of the first chunk (first 50 chars):\n---\n{license_chunks[0][:50]}...\n---")
```

**Expected Output (for `LICENSE`):**

```text
Number of chunks for LICENSE: 1
Content of the first chunk (first 50 chars):
---
MIT License


Copyright (c) 2024 Cased


Permiss...
---
```

**Example 2: Chunking a larger file (e.g., `src/kit/repository.py`)**

For larger files, the content will be split into multiple string chunks.

```python
repo_py_path = "src/kit/repository.py"
repo_py_chunks = repo.chunk_file_by_lines(repo_py_path, max_lines=50)


print(f"\nNumber of chunks for {repo_py_path} (with max_lines=50): {len(repo_py_chunks)}")


for i, chunk_content in enumerate(repo_py_chunks[:2]):
    print(f"\n--- Chunk {i+1} for {repo_py_path} ---")
    print(f"  Approx. line count: {len(chunk_content.splitlines())}")
    print(f"  Content (first 100 chars):\n  \"\"\"\n{chunk_content[:100]}...\n  \"\"\"")
```

**Expected Output (for `src/kit/repository.py`, showing 2 of 7 chunks):**

```text
Number of chunks for src/kit/repository.py (with max_lines=50): 7


--- Chunk 1 for src/kit/repository.py ---
  Approx. line count: 48
  Content (first 100 chars):
  """
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Unio...
  """


--- Chunk 2 for src/kit/repository.py ---
  Approx. line count: 20
  Content (first 100 chars):
  """
                    # If not on a branch (detached HEAD), get commit SHA
                    sha_cmd = ["git", "rev...
  """
```

*(Note: Actual line counts per chunk may vary slightly based on how the chunker splits the content. The second chunk from your output had fewer lines than the first.)*

### Chunking File Content by Symbols (`repo.chunk_file_by_symbols()`)

A more semantically aware way to chunk files is by symbols. This method uses `kit`’s understanding of code structure to break the file into chunks that correspond to whole symbols like functions, classes, or methods. Each chunk represents a meaningful structural unit of the code.

The method signature is: `repo.chunk_file_by_symbols(file_path: str) -> List[Dict[str, Any]]`

* `file_path`: The relative path to the file within the repository.
* It returns a list of dictionaries. Each dictionary represents one symbol-chunk and contains details like the symbol’s `name`, `type`, `code` content, and `start_line`/`end_line` numbers.

**Example: Chunking `src/kit/tree_sitter_symbol_extractor.py` by symbols**

```python
extractor_path = "src/kit/tree_sitter_symbol_extractor.py"
symbol_chunks = repo.chunk_file_by_symbols(file_path=extractor_path)


print(f"Successfully chunked '{extractor_path}' into {len(symbol_chunks)} symbol-based chunks.")


for i, chunk_dict in enumerate(symbol_chunks[:2]): # Show first 2 symbol chunks
    print(f"\n--- Symbol Chunk {i+1} ---")
    symbol_name = chunk_dict.get('name', 'N/A')
    symbol_type = chunk_dict.get('type', 'N/A')
    start_line = chunk_dict.get('start_line', 'N/A')
    end_line = chunk_dict.get('end_line', 'N/A')
    code_content = chunk_dict.get('code', '')


    print(f"  Symbol Name: {symbol_name}")
    print(f"  Symbol Type: {symbol_type}")
    print(f"  Start Line (0-indexed): {start_line}")
    print(f"  End Line (0-indexed): {end_line}")
    print(f"  Line Count of code: {len(code_content.splitlines())}")
    print(f"  Content (first 150 chars of code):\n  \"\"\"\n{code_content[:150]}...\n  \"\"\"")
```

**Expected Output (for `src/kit/tree_sitter_symbol_extractor.py`, showing 2 of 4 chunks):**

```text
Successfully chunked 'src/kit/tree_sitter_symbol_extractor.py' into 4 symbol-based chunks.


--- Symbol Chunk 1 ---
  Symbol Name: TreeSitterSymbolExtractor
  Symbol Type: class
  Start Line (0-indexed): 28
  End Line (0-indexed): 197
  Line Count of code: 170
  Content (first 150 chars of code):
  """
class TreeSitterSymbolExtractor:
    """
    Multi-language symbol extractor using tree-sitter queries (tags.scm).
    Register new languages by addin...
  """


--- Symbol Chunk 2 ---
  Symbol Name: get_parser
  Symbol Type: method
  Start Line (0-indexed): 38
  End Line (0-indexed): 45
  Line Count of code: 8
  Content (first 150 chars of code):
  """
def get_parser(cls, ext: str) -> Optional[Any]:
        if ext not in LANGUAGES:
            return None
        if ext not in cls._parsers:
         ...
  """
```

This provides a more structured way to access and process individual components of a code file.

We’ll add more examples here as we try them out.

*(This document will be updated as we explore more features.)*

# Integrating with Supersonic

> Using kit for code analysis and Supersonic for automated PR creation.

`kit` excels at understanding and analyzing codebases, while [Supersonic](https://github.com/cased/supersonic) provides a high-level Python API specifically designed for programmatically creating GitHub Pull Requests. Combining them allows you to build powerful workflows that analyze code, generate changes, and automatically propose those changes via PRs.

Note

**Use Case** Think of workflows like AI-powered code refactoring, automated dependency updates based on analysis, or generating documentation snippets from code and submitting them for review.

## The Workflow: Analyze with `kit`, Act with `Supersonic`

A typical integration pattern looks like this:

1. **Analyze Code with `kit`**: Use `kit.Repository` methods like `extract_symbols`, `find_symbol_usages`, or `search_semantic` to understand the codebase or identify areas for modification.
2. **Generate Changes**: Based on the analysis (potentially involving an LLM), generate the new code content or identify necessary file modifications.
3. **Create PR with `Supersonic`**: Use `Supersonic`’s simple API (`create_pr_from_content`, `create_pr_from_file`, etc.) to package the generated changes into a new Pull Request on GitHub.

## Example: AI Refactoring Suggestion

Imagine an AI tool that uses `kit` to analyze a Python file, identifies a potential refactoring, generates the improved code, and then uses `Supersonic` to create a PR.

```python
import kit
from supersonic import Supersonic
import os


# Assume kit.Repository is initialized with a local path
LOCAL_REPO_PATH = "/path/to/your/local/repo/clone"
# repo_analyzer = kit.Repository(LOCAL_REPO_PATH)
# Note: kit analysis methods like extract_symbols would still be used here in a real scenario.


# Assume 'ai_generate_refactoring' is your function that uses an LLM
# potentially fed with context from kit (not shown here for brevity)
def ai_generate_refactoring(original_code: str) -> str:
    # ... your AI logic here ...
    improved_code = original_code.replace("old_function", "new_function") # Simplified example
    return improved_code


# --- Configuration ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER_SLASH_NAME = "your-org/your-repo" # For Supersonic PR creation
RELATIVE_FILE_PATH = "src/legacy_module.py" # Relative path within the repo
FULL_FILE_PATH = os.path.join(LOCAL_REPO_PATH, RELATIVE_FILE_PATH)
TARGET_BRANCH = "main" # Or dynamically determine


# --- Main Workflow ---


try:
    # 1. Get original content (assuming local repo)
    if not os.path.exists(FULL_FILE_PATH):
        print(f"Error: File not found at {FULL_FILE_PATH}")
        exit()


    with open(FULL_FILE_PATH, 'r') as f:
        original_content = f.read()


    # 2. Generate Changes (using AI or other logic)
    refactored_content = ai_generate_refactoring(original_content)


    if refactored_content != original_content:
        # 3. Create PR with Supersonic
        supersonic_client = Supersonic(GITHUB_TOKEN)
        pr_title = f"AI Refactor: Improve {RELATIVE_FILE_PATH}"
        pr_body = f"""
        AI analysis suggests refactoring in `{RELATIVE_FILE_PATH}`.


        This PR applies the suggested changes. Please review carefully.
        """


        pr_url = supersonic_client.create_pr_from_content(
            repo=REPO_OWNER_SLASH_NAME,
            content=refactored_content,
            upstream_path=RELATIVE_FILE_PATH, # Path within the target repo
            title=pr_title,
            description=pr_body,
            base_branch=TARGET_BRANCH,
            labels=["ai-refactor", "needs-review"],
            draft=True # Good practice for AI suggestions
        )
        print(f"Successfully created PR: {pr_url}")
    else:
        print("No changes generated.")


except Exception as e:
    print(f"An error occurred: {e}")
```

This example illustrates how `kit`’s analytical strengths can be combined with `Supersonic`’s action-oriented PR capabilities to build powerful code automation.

# Using Ollama with kit review

> Complete guide to using kit with Ollama for free local AI code revews.

# 🦙 Using Ollama with Kit

**Kit** has first-class support for **free local AI models** via [Ollama](https://ollama.ai/). No API keys, no costs, no data leaving your machine.

## Why Ollama?

* **No cost** - unlimited usage
* **Complete privacy** - data never leaves your machine
* **No API keys** - just install and run
* **No rate limits** - only hardware constraints
* **Works offline** - perfect for secure environments
* **Latest models** - access to cutting-edge open source AI

## Quick Setup (2 minutes)

### 1. Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh


# Windows
# Download from https://ollama.ai/download
```

### 2. Pull a Model

Choose based on your use case:

```bash
# Best for code tasks (recommended)
ollama pull qwen2.5-coder:latest


# Best for reasoning
ollama pull deepseek-r1:latest


# Best for coding agents
ollama pull devstral:latest


# Good general purpose
ollama pull llama3.3:latest
```

### 3. Start Using with Kit

```python
from kit import Repository
from kit.summaries import OllamaConfig


# Configure Ollama
config = OllamaConfig(model="qwen2.5-coder:latest")


# Use with any repository
repo = Repository("/path/to/your/project")
summarizer = repo.get_summarizer(config=config)


# Summarize code at no cost
summary = summarizer.summarize_file("main.py")
print(summary)  # Cost: $0.00
```

## 💡 Complete Examples

### Code Summarization

```python
from kit import Repository
from kit.summaries import OllamaConfig


# Setup
repo = Repository("/path/to/project")
config = OllamaConfig(
    model="qwen2.5-coder:latest",
    temperature=0.1,  # Lower for more focused analysis
    max_tokens=1000
)
summarizer = repo.get_summarizer(config=config)


# Summarize a file
summary = summarizer.summarize_file("complex_module.py")
print(f"File Summary: {summary}")


# Summarize a specific function
func_summary = summarizer.summarize_function("utils.py", "parse_config")
print(f"Function Summary: {func_summary}")


# Summarize a class
class_summary = summarizer.summarize_class("models.py", "UserManager")
print(f"Class Summary: {class_summary}")
```

### PR Reviews (No Cost)

```bash
# Setup Ollama for PR reviews
kit review --init-config
# Choose "ollama" as provider
# Choose "qwen2.5-coder:latest" as model


# Review any PR at no cost
kit review https://github.com/owner/repo/pull/123
# Cost: $0.00
```

### Batch Documentation Generation

```python
from kit import Repository
from kit.summaries import OllamaConfig
import os


def generate_docs(project_path, output_file):
    """Generate documentation for an entire project."""
    repo = Repository(project_path)
    config = OllamaConfig(model="qwen2.5-coder:latest")
    summarizer = repo.get_summarizer(config=config)


    with open(output_file, 'w') as f:
        f.write(f"# Documentation for {os.path.basename(project_path)}\n\n")


        # Get all Python files
        files = repo.get_file_tree()
        python_files = [f for f in files if f['path'].endswith('.py') and not f.get('is_dir')]


        for file_info in python_files:
            file_path = file_info['path']
            try:
                summary = summarizer.summarize_file(file_path)
                f.write(f"## {file_path}\n\n{summary}\n\n")
                print(f"✅ Documented {file_path} (Cost: $0.00)")
            except Exception as e:
                print(f"⚠️ Skipped {file_path}: {e}")


# Usage
generate_docs("/path/to/project", "project_docs.md")
```

### Legacy Codebase Analysis

```python
def analyze_legacy_code(repo_path):
    """Analyze and understand legacy code using free AI."""
    repo = Repository(repo_path)
    config = OllamaConfig(model="qwen2.5-coder:latest")
    summarizer = repo.get_summarizer(config=config)


    # Find all symbols
    symbols = repo.extract_symbols()


    # Group by type
    functions = [s for s in symbols if s.get('type') == 'FUNCTION']
    classes = [s for s in symbols if s.get('type') == 'CLASS']


    print(f"Found {len(functions)} functions and {len(classes)} classes")


    # Analyze complex functions (those with many lines)
    complex_functions = [f for f in functions if len(f.get('code', '').split('\n')) > 20]


    for func in complex_functions[:5]:  # Analyze top 5 complex functions
        file_path = func['file']
        func_name = func['name']


        analysis = summarizer.summarize_function(file_path, func_name)
        print(f"\n📝 {func_name} in {file_path}:")
        print(f"   {analysis}")
        print(f"   Cost: $0.00")
```

## ⚙️ Advanced Configuration

### Custom Configuration

```python
config = OllamaConfig(
    model="qwen2.5-coder:32b",        # Use larger model for better results
    base_url="http://localhost:11434", # Default Ollama endpoint
    temperature=0.0,                   # Deterministic output
    max_tokens=2000,                   # Longer responses
)
```

### Remote Ollama Server

```python
config = OllamaConfig(
    model="qwen2.5-coder:latest",
    base_url="http://your-server:11434",  # Remote Ollama instance
    temperature=0.1,
    max_tokens=1000,
)
```

### Multiple Models for Different Tasks

```python
# Code-focused model for functions
code_config = OllamaConfig(model="qwen2.5-coder:latest", temperature=0.1)


# Reasoning model for complex analysis
reasoning_config = OllamaConfig(model="deepseek-r1:latest", temperature=0.2)


# Use different models for different tasks
code_summarizer = repo.get_summarizer(config=code_config)
reasoning_summarizer = repo.get_summarizer(config=reasoning_config)
```

## 🔧 Troubleshooting

### Common Issues

**“Connection refused” error:**

```bash
# Make sure Ollama is running
ollama serve


# Or check if it's already running
ps aux | grep ollama
```

**“Model not found” error:**

```bash
# Pull the model first
ollama pull qwen2.5-coder:latest


# List available models
ollama list
```

**Slow responses:**

* Use smaller models like `qwen2.5-coder:0.5b` for faster responses
* Reduce `max_tokens` for shorter outputs
* Ensure sufficient RAM (8GB+ recommended for 7B models)

### Performance Tips

1. **Choose the right model size:**

   * 0.5B-3B: Fast, good for simple tasks
   * 7B-14B: Balanced speed and quality
   * 32B+: Best quality, requires more resources

2. **Optimize settings:**

   ```python
   # For speed
   config = OllamaConfig(
       model="qwen2.5-coder:0.5b",
       temperature=0.1,
       max_tokens=500
   )


   # For quality
   config = OllamaConfig(
       model="qwen2.5-coder:32b",
       temperature=0.0,
       max_tokens=2000
   )
   ```

3. **Hardware considerations:**

   * RAM: 8GB minimum, 16GB+ recommended for larger models
   * GPU: Optional but significantly speeds up inference
   * Storage: Models range from 500MB to 400GB

## 💰 Cost Comparison

| Provider         | Cost per Review | Setup Time    | Privacy             | Offline             |
| ---------------- | --------------- | ------------- | ------------------- | ------------------- |
| **Ollama**       | **$0.00**       | 2 minutes     | ✅ 100% private      | ✅ Works offline     |
| OpenAI GPT-4o    | \~$0.10         | API key setup | ❌ Sent to OpenAI    | ❌ Requires internet |
| Anthropic Claude | \~$0.08         | API key setup | ❌ Sent to Anthropic | ❌ Requires internet |

## 🤝 Community

* **Discord**: [Join the kit Discord](https://discord.gg/XpqU65pY) for help and discussions
* **GitHub**: [Report issues or contribute](https://github.com/cased/kit)
* **Ollama Community**: [Ollama Discord](https://discord.gg/ollama) for model-specific help

# Practical Recipes

Note

These snippets are *copy-paste-ready* solutions for common developer-productivity tasks with **kit**. Adapt them to scripts, CI jobs, or IDE plugins.

## 1. Rename every function `old_name` → `new_name`

```python
from pathlib import Path
from kit import Repository


repo = Repository("/path/to/project")


# Gather definitions & references (quick heuristic)
usages = repo.find_symbol_usages("old_name", symbol_type="function")


edits: dict[str, str] = {}
for u in usages:
    path, line = u["file"], u.get("line")
    if line is None:
        continue
    lines = repo.get_file_content(path).splitlines()
    lines[line] = lines[line].replace("old_name", "new_name")
    edits[path] = "\n".join(lines) + "\n"


# Apply edits – prompt the user first!
for rel_path, new_src in edits.items():
    Path(repo.repo_path, rel_path).write_text(new_src)


repo.mapper.scan_repo()  # refresh symbols if you'll run more queries
```

***

## 2. Summarize a Git diff for an LLM PR review

```python
from kit import Repository
# Assuming OpenAI for this example, and API key is set in environment
from kit.summaries import OpenAIConfig


repo = Repository(".")
assembler = repo.get_context_assembler()
# diff_text would be a string containing the output of `git diff`
# Example:
# diff_text = subprocess.run(["git", "diff", "HEAD~1"], capture_output=True, text=True).stdout


# Ensure diff_text is populated before this step in a real script
diff_text = """diff --git a/file.py b/file.py
index 0000000..1111111 100644
--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
-old line
+new line
""" # Placeholder diff_text


assembler.add_diff(diff_text)
context_blob = assembler.format_context()


# Get the summarizer and its underlying LLM client to summarize arbitrary text
# This example assumes you want to use the default OpenAI configuration for the summarizer.
# If you have a specific config (OpenAI, Anthropic, Google), pass it to get_summarizer.
summarizer_instance = repo.get_summarizer() # Uses default OpenAIConfig
llm_client = summarizer_instance._get_llm_client() # Access the configured client


summary = "Could not generate summary."
if hasattr(llm_client, 'chat') and hasattr(llm_client.chat, 'completions'): # OpenAI-like client
    try:
        response = llm_client.chat.completions.create(
            model=summarizer_instance.config.model if summarizer_instance.config else "gpt-4o", # Get model from config
            messages=[
                {"role": "system", "content": "You are an expert software engineer. Please summarize the following code changes and context."},
                {"role": "user", "content": context_blob}
            ],
            temperature=0.2,
            max_tokens=500 # Adjust as needed
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        summary = f"Error generating summary: {e}"
elif hasattr(llm_client, 'messages') and hasattr(llm_client.messages, 'create'): # Anthropic-like client
    try:
        response = llm_client.messages.create(
            model=summarizer_instance.config.model if summarizer_instance.config else "claude-3-opus-20240229",
            system="You are an expert software engineer. Please summarize the following code changes and context.",
            messages=[
                {"role": "user", "content": context_blob}
            ],
            max_tokens=500,
            temperature=0.2,
        )
        summary = response.content[0].text.strip()
    except Exception as e:
        summary = f"Error generating summary: {e}"
# Add similar elif for Google GenAI client if needed, or abstract this LLM call further


print(summary)
```

***

## 3. Semantic search for authentication code

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embed = lambda text: model.encode([text])[0].tolist()


repo = Repository(".")
vs = repo.get_vector_searcher(embed_fn=embed)
vs.build_index()


hits = repo.search_semantic("How is user authentication handled?", embed_fn=embed)
for h in hits:
    print(h["file"], h.get("name"))
```

***

## 4. Export full repo index to JSON (file tree + symbols)

```python
repo = Repository("/path/to/project")
repo.write_index("repo_index.json")
```

***

## 5. Find All Callers of a Specific Function (Cross-File)

This recipe helps you understand where a particular function is being used throughout your entire codebase, which is crucial for impact analysis or refactoring.

```python
from kit import Repository


# Initialize the repository
repo = Repository("/path/to/your_project")


# Specify the function name and its type
function_name_to_trace = "my_target_function"


# Find all usages (definitions, calls, imports)
usages = repo.find_symbol_usages(function_name_to_trace, symbol_type="function")


print(f"Usages of function '{function_name_to_trace}':")
for usage in usages:
    file_path = usage.get("file")
    line_number = usage.get("line") # Assuming 'line' is the start line of the usage/symbol
    context_snippet = usage.get("context", "No context available")
    usage_type = usage.get("type", "unknown") # e.g., 'function' for definition, 'call' for a call site


    # We are interested in where it's CALLED, so we might filter out the definition itself if needed,
    # or differentiate based on the 'type' or 'context'.
    # For this example, we'll print all usages.
    if line_number is not None:
        print(f"- Found in: {file_path}:L{line_number + 1}") # (line is 0-indexed, display as 1-indexed)
    else:
        print(f"- Found in: {file_path}")
    print(f"    Type: {usage_type}")
    print(f"    Context: {context_snippet.strip()}\n")


# Example: Filtering for actual call sites (heuristic based on context or type if available)
# print(f"\nCall sites for function '{function_name_to_trace}':")
# for usage in usages:
#     # This condition might need refinement based on what 'find_symbol_usages' returns for 'type' of a call
#     if usage.get("type") != "function" and function_name_to_trace + "(" in usage.get("context", ""):
#         file_path = usage.get("file")
#         line_number = usage.get("line")
#         print(f"- Call in: {file_path}:L{line_number + 1 if line_number is not None else 'N/A'}")
```

***

## 6. Identify Potentially Unused Functions (Heuristic)

This recipe provides a heuristic to find functions that *might* be unused within the analyzed codebase. This can be a starting point for identifying dead code. Note that this is a heuristic because it might not catch dynamically called functions, functions part of a public API but not used internally, or functions used only in parts of the codebase not analyzed (e.g., separate test suites).

```python
from kit import Repository


repo = Repository("/path/to/your_project")


# Get all symbols from the repository index
# The structure of repo.index() might vary; assuming it's a dict like {'symbols': {'file_path': [symbol_dicts]}}
# or a direct way to get all function definitions.
# For this example, let's assume we can iterate through all symbols and filter functions.


# A more robust way might be to iterate files, then symbols within files from repo.index()
# index = repo.index()
# all_symbols_by_file = index.get("symbols", {})


print("Potentially unused functions:")


# First, get a list of all function definitions
defined_functions = []
repo_index = repo.index() # Assuming this fetches file tree and symbols
symbols_map = repo_index.get("symbols", {})


for file_path, symbols_in_file in symbols_map.items():
    for symbol_info in symbols_in_file:
        if symbol_info.get("type") == "function":
            defined_functions.append({
                "name": symbol_info.get("name"),
                "file": file_path,
                "line": symbol_info.get("line_start", 0) # or 'line'
            })


for func_def in defined_functions:
    function_name = func_def["name"]
    definition_file = func_def["file"]
    definition_line = func_def["line"]


    if not function_name: # Skip if name is missing
        continue


    usages = repo.find_symbol_usages(function_name, symbol_type="function")


    # Filter out the definition itself from the usages to count actual calls/references
    # This heuristic assumes a usage is NOT the definition if its file and line differ,
    # or if the usage 'type' (if available and detailed) indicates a call.
    # A simpler heuristic: if only 1 usage, it's likely just the definition.


    actual_references = []
    for u in usages:
        # Check if the usage is different from the definition site
        if not (u.get("file") == definition_file and u.get("line") == definition_line):
            actual_references.append(u)


    # If a function has no other references apart from its own definition site (or very few)
    # It's a candidate for being unused. The threshold (e.g., 0 or 1) can be adjusted.
    if len(actual_references) == 0:
        print(f"- Function '{function_name}' defined in {definition_file}:L{definition_line + 1} has no apparent internal usages.")


:::caution[Limitations of this heuristic:]
**Limitations of this heuristic:**


*   **Dynamic Calls:** Functions called dynamically (e.g., through reflection, or if the function name is constructed from a string at runtime) won't be detected as used.
*   **Public APIs:** Functions intended for external use (e.g., library functions) will appear unused if the analysis is limited to the library's own codebase.
*   **Test Code:** If your test suite is separate and not part of the `Repository` path being analyzed, functions used only by tests might be flagged.
*   **Object Methods:** The `symbol_type="function"` might need adjustment or further logic if you are also looking for unused *methods* within classes, as their usage context is different.
:::
```