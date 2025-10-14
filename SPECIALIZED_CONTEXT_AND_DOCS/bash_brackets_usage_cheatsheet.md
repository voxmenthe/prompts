
## `$( commands )` — Command substitution

**What:** Run a command and substitute its **stdout** into the surrounding command; trailing newlines are trimmed (internal newlines preserved). Prefer this over legacy backticks.

```bash
log_file="/var/log/syslog"
keyword="error"
matches=$(grep -n "$keyword" "$log_file")
printf '%s\n' "$matches"
```

**Compositions / patterns**

```bash
# Nesting:
latest_id=$(curl -s https://api/x | jq -r '.items[0].id')

# Capture a whole file (bash-only fast path):
blob=$(< /etc/hosts)

# Feed into arrays safely (avoid word-splitting with mapfile):
mapfile -t lines < <(grep -n "$keyword" "$log_file")
```

**Tips & pitfalls**

* Use quotes around the substitution when embedding in strings: `echo "-> $(cmd) <-"`.
* Avoid `for x in $(ls)`; it splits on whitespace. Prefer globs or `while IFS= read -r`.
* If you need to **preserve** trailing newlines, read via `read`/`mapfile`.

---

## `{ list; }` — Group in the **current** shell

**What:** Run multiple commands as a block in the same shell (state changes persist). Requires a semicolon (or newline) before `}`.

```bash
{ echo "setup"; cd /tmp; touch a b c; pwd; }  # cd persists afterward
```

**Compositions / patterns**

```bash
# Redirect the whole block:
{ make clean; make -j8; } >build.log 2>&1

# Run with environment vars that persist:
{ set -e; set -o pipefail; cmd1 | cmd2; echo done; }
```

**Tips**

* Use `{ ...; } >file` to capture all output once, instead of per-command redirects.
* Compare with `( ... )`: braces keep state; parentheses isolate (see below).

---

## `( a b c )` — Array literals (bash)

**What:** Parentheses create an array. Iterate with `"${arr[@]}"` to preserve element boundaries.

```bash
files=(log.txt "notes 1.txt" report.txt)
for f in "${files[@]}"; do
  echo "Processing: $f"
done
```

**Compositions / patterns**

```bash
# Append / splice
files+=("extra.txt")

# Indices and length
for i in "${!files[@]}"; do printf '[%d] %s\n' "$i" "${files[i]}"; done
echo "count=${#files[@]}"

# Fill from command output safely
mapfile -t files < <(find . -maxdepth 1 -name '*.txt')
```

**Tips**

* Arrays are **bash/ksh**, not POSIX sh.
* Use `"${arr[@]}"` (not `"${arr[*]}"`) to iterate element-by-element.

---

## `( list )` — Subshell (isolated)

**What:** Run a block in a **child** shell; changes do **not** leak back (great for temporary `cd`, `umask`, `set -e`, etc.).

```bash
( cd /home/user && ls && whoami )
# After this, you’re back where you started.
```

**Compositions / patterns**

```bash
# Isolate flags:
( set -euo pipefail; cmd1 | cmd2 )

# Background the whole block:
( do_long_thing; post_step ) &
```

**Tips**

* Any pipeline segment may run in a subshell—avoid relying on side effects within pipelines.
* Related but different: **process substitution** `<(cmd)` / `>(cmd)` creates named pipes/FIFOs.

---

## `{range}` — Brace expansion

**What:** Compile-time string expansion (not loops). Useful for batch naming.

```bash
echo backup_{1..4}.tar.gz       # backup_1.tar.gz ... backup_4.tar.gz
touch img_{01..05}.png          # zero-padded
echo {a..e} {A..C} {10..0..2}   # step of 2
```

**Compositions / patterns**

```bash
for f in backup_{1..4}.tar.gz; do
  mv "$f" /var/oldbackups/
done
```

**Tips**

* Happens **before** variable expansion and globbing.
* Quotes disable it: `echo "{1..3}"` → literal.
* Bash option `shopt -u braceexpand` can turn it off.

---

## `${variable}` & `${expression}` — Parameter expansion

**What:** Read or transform variables without forking processes.

```bash
username="John"
echo "Hello, ${username}!"
```

**Common transforms**

```bash
# Change extensions / path parts
file="report.txt";  echo "${file%.txt}.bak"          # -> report.bak
path="/var/log/syslog"; echo "${path##*/}"           # -> syslog (basename)
echo "${path%/*}"                                    # -> /var/log (dirname)

# Length / slicing
s="abcdef"; echo "${#s}"        # 6
echo "${s:2:3}"                 # cde

# Defaulting and assignment
: "${PORT:=8080}"               # set PORT if unset or null
echo "${NAME:-anonymous}"       # fallback if unset or null
: "${REQUIRED:?must be set}"    # error if unset or null

# Replacement
s="a-b-c"; echo "${s/-/_}"      # a_b-c (first)
echo "${s//-/_}"                # a_b_c (all)

# Case ops (bash)
v="foo"; echo "${v^}" "${v^^}"  # Foo  FOO
```

**Tips**

* `${var-word}` vs `${var:-word}`: `:-` also triggers on empty.
* Indirection: `${!name}` expands the var whose **name** is in `name`.
* Quote expansions unless you **want** word splitting/globbing: `echo "${arr[@]}"`.

---

## `$(( expression ))` and `(( expression ))` — Arithmetic

**What:** Integer math.

* `$((...))` **expands** to a value.
* `((...))` **evaluates** and returns 0/1 (truthy/falsy), supports in-place assignment.

```bash
a=5 b=3
sum=$((a + b * 2))  # expansion form
echo "$sum"         # 11

(( a += 10, b++ ))  # evaluation form
if (( a > 12 && b == 4 )); then echo "ok"; fi
```

**Compositions / patterns**

```bash
# C-like increments in loops
for ((i=0; i<10; i++)); do printf '%d ' "$i"; done

# Bit ops / powers
((mask = 1<<5)); ((n ^= mask))
((x = y ** 3))  # exponent
```

**Tips**

* Beware leading zeros → octal. Force base-10: `$((10#$num))`.
* Variables inside need **no** `$` sigil: `((a=b+1))`.

---

## `[ expression ]` — Test (POSIX `test`)

**What:** Classic test command; requires spaces: `[`, operands, `]`.

```bash
file="/etc/passwd"
if [ -f "$file" ]; then
  echo "File exists"
fi
```

**Common operators**

```bash
# Strings
[ -z "$s" ]     # empty
[ "$a" = "$b" ] # equal
[ "$a" != "$b" ]

# Integers
[ "$n" -gt 10 ] [ "$n" -eq 0 ]

# Files
[ -r "$p" ] [ -w "$p" ] [ -x "$p" ] [ -d "$p" ] [ -e "$p" ]
```

**Tips**

* Always quote variables to avoid globbing/splitting.
* Combine conditions with `&&`/`||`:
  `[ -f "$f" ] && [ -s "$f" ] || echo "missing or empty"`.
* Prefer `[[ ... ]]` for patterns/regex and fewer quoting hazards (next box).

---

## `[[ expression ]]` — Advanced Bash test

**What:** Bash’s safer, richer conditional. No pathname expansion or word splitting on unquoted vars; supports glob patterns and regex.

```bash
user="$USER"
if [[ $user == root ]]; then echo "You are root"; fi
if [[ $file == *.log ]]; then echo "log file"; fi
```

**Regex & captures**

```bash
s="id=1234"
if [[ $s =~ ^id=([0-9]+)$ ]]; then
  echo "num: ${BASH_REMATCH[1]}"
fi
```

**More examples**

```bash
# String ordering (lexicographic)
[[ $a < $b ]]

# Case-insensitive match
shopt -s nocasematch
[[ $name == *admin* ]]
shopt -u nocasematch
```

**Tips**

* Don’t quote the regex on the RHS of `=~` (quotes make it a literal).
* Extglob patterns (`@(a|b)`, `!(tmp)`) work with `shopt -s extglob`.

---

## Extra “fits together” examples

**1) Capture → transform → test**

```bash
latest=$(ls -1t /var/log/*.log 2>/dev/null | head -n1)
base=${latest##*/}                       # parameter expansion
if [[ $base == *.log ]]; then echo "$base"; fi
```

**2) Group, redirect once, and use arithmetic**

```bash
count=0
{ for f in data_{01..10}.txt; do
    [[ -s $f ]] && ((count++)) && echo "ok: $f"
  done
  echo "total=$count"
} >run.log 2>&1
```

**3) Subshell to isolate state while feeding an array**

```bash
mapfile -t files < <(
  ( cd /var/log && ls -1 *.log )    # cd doesn’t leak
)
printf '%s\n' "${files[@]}"
```

---

### Compatibility notes

* POSIX sh: `[ ]`, `$( )`, `${ }`, brace expansion (often), but **no** arrays, `[[ ]]`, or `(( ))`.
* Bash/ksh/zsh: support varies, but everything above is standard in modern **bash**.

