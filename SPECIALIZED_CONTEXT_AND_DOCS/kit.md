Command Line Interface
kit provides a comprehensive command-line interface for repository analysis, symbol extraction, and AI-powered development workflows. All commands support both human-readable and machine-readable output formats for seamless integration with other tools.

Installation & Setup
Terminal window
# Install kit
pip install cased-kit

# Verify installation
kit --version

# Get help for any command
kit --help
kit <command> --help

Core Commands
Repository Analysis
kit symbols
Extract symbols (functions, classes, variables) from a repository with intelligent caching for 25-36x performance improvements.

Terminal window
kit symbols <repository-path> [OPTIONS]

Options:

--format, -f <format>: Output format (table, json, names, plain)
--pattern, -p <pattern>: File pattern filter (e.g., *.py, src/**/*.js)
--output, -o <file>: Save output to file
--type, -t <type>: Filter by symbol type (function, class, variable, etc.)
Examples:

Terminal window
# Extract all symbols (uses incremental analysis for speed)
kit symbols /path/to/repo

# Get only Python functions as JSON
kit symbols /path/to/repo --pattern "*.py" --type function --format json

# Export to file for further analysis
kit symbols /path/to/repo --output symbols.json

# Quick symbol names for scripting
kit symbols /path/to/repo --format names | grep "test_"

kit file-tree
Display repository structure with file type indicators and statistics.

Terminal window
kit file-tree <repository-path> [OPTIONS]

Options:

--format, -f <format>: Output format (tree, json, names, plain)
--pattern, -p <pattern>: File pattern filter
--output, -o <file>: Save output to file
--max-depth, -d <depth>: Limit directory depth
Examples:

Terminal window
# Show repository structure
kit file-tree /path/to/repo

# Only Python files, limited depth
kit file-tree /path/to/repo --pattern "*.py" --max-depth 3

# Export structure as JSON
kit file-tree /path/to/repo --format json --output structure.json

kit search
Fast text search across repository files with regex support.

Terminal window
kit search <repository-path> <query> [OPTIONS]

Options:

--pattern, -p <pattern>: File pattern filter
--output, -o <file>: Save output to file
--context, -c <lines>: Show context lines around matches
--case-sensitive, -s: Enable case-sensitive search
Examples:

Terminal window
# Search for function definitions
kit search /path/to/repo "def.*login"

# Search in specific files with context
kit search /path/to/repo "TODO" --pattern "*.py" --context 2

# Case-sensitive search
kit search /path/to/repo "ApiClient" --case-sensitive

kit usages
Find all usages of a specific symbol across the repository.

Terminal window
kit usages <repository-path> <symbol-name> [OPTIONS]

Options:

--symbol-type, -t <type>: Filter by symbol type
--output, -o <file>: Save output to file
--format, -f <format>: Output format
Examples:

Terminal window
# Find all usages of a function
kit usages /path/to/repo "calculate_total"

# Find class usages with JSON output
kit usages /path/to/repo "UserModel" --symbol-type class --format json

Cache Management
kit cache
Manage the incremental analysis cache for optimal performance.

Terminal window
kit cache <action> <repository-path> [OPTIONS]

Actions:

status: Show cache statistics and health
clear: Clear all cached data
cleanup: Remove stale entries for deleted files
stats: Show detailed performance statistics
Examples:

Terminal window
# Check cache status
kit cache status /path/to/repo

# Clear cache if needed
kit cache clear /path/to/repo

# Clean up stale entries
kit cache cleanup /path/to/repo

# View detailed statistics
kit cache stats /path/to/repo

Sample Output:

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

AI-Powered Workflows
kit summarize
Generate AI-powered pull request summaries with optional PR body updates.

Terminal window
kit summarize <pr-url> [OPTIONS]

Options:

--plain, -p: Output raw summary content (no formatting)
--dry-run, -n: Generate summary without posting to GitHub
--model, -m <model>: Override LLM model for this summary
--config, -c <file>: Use custom configuration file
--update-pr-body: Add summary to PR description
Examples:

Terminal window
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

kit commit
Generate intelligent commit messages from staged git changes.

Terminal window
kit commit [repository-path] [OPTIONS]

Options:

--dry-run, -n: Show generated message without committing
--model, -m <model>: Override LLM model
--config, -c <file>: Use custom configuration file
Examples:

Terminal window
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

Sample Output:

Generated commit message:
feat(auth): add JWT token validation middleware

- Implement JWTAuthMiddleware for request authentication
- Add token validation with signature verification
- Include error handling for expired and invalid tokens
- Update middleware registration in app configuration

Commit? [y/N]: y

Content Processing
kit context
Extract contextual code around specific lines for LLM analysis.

Terminal window
kit context <repository-path> <file-path> <line-number> [OPTIONS]

Options:

--lines, -n <count>: Context lines around target (default: 10)
--output, -o <file>: Save output to JSON file
Examples:

Terminal window
# Get context around a specific line
kit context /path/to/repo src/main.py 42

# Export context for analysis
kit context /path/to/repo src/utils.py 15 --output context.json

kit chunk-lines
Split file content into line-based chunks for LLM processing.

Terminal window
kit chunk-lines <repository-path> <file-path> [OPTIONS]

Options:

--max-lines, -n <count>: Maximum lines per chunk (default: 50)
--output, -o <file>: Save output to JSON file
Examples:

Terminal window
# Default chunking (50 lines)
kit chunk-lines /path/to/repo src/large-file.py

# Smaller chunks for detailed analysis
kit chunk-lines /path/to/repo src/main.py --max-lines 20

# Export chunks for LLM processing
kit chunk-lines /path/to/repo src/main.py --output chunks.json

kit chunk-symbols
Split file content by code symbols (functions, classes) for semantic chunking.

Terminal window
kit chunk-symbols <repository-path> <file-path> [OPTIONS]

Options:

--output, -o <file>: Save output to JSON file
Examples:

Terminal window
# Chunk by symbols (functions, classes)
kit chunk-symbols /path/to/repo src/main.py

# Export symbol-based chunks
kit chunk-symbols /path/to/repo src/api.py --output symbol-chunks.json

Export Operations
kit export
Export repository data to structured JSON files for external tools and analysis.

Terminal window
kit export <repository-path> <data-type> <output-file> [OPTIONS]

Data Types:

index: Complete repository index (files + symbols)
symbols: All extracted symbols
file-tree: Repository file structure
symbol-usages: Usages of a specific symbol
Options:

--symbol <name>: Symbol name (required for symbol-usages)
--symbol-type <type>: Symbol type filter (for symbol-usages)
Examples:

Terminal window
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

