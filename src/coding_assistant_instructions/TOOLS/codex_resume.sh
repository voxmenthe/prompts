codex_resume() {
  emulate -L zsh
  setopt localoptions extendedglob noshwordsplit
  unsetopt xtrace verbose 2>/dev/null || true
  set +x +v

  local want_last=0 list_any=0 refresh=0 limit=200 target_dir="" sep=$'\t'
  local -a passthru

  while (( $# > 0 )); do
    case "$1" in
      --last) want_last=1; shift ;;
      --any)  list_any=1; shift ;;
      --dir)  target_dir=${2:-}; shift 2 ;;
      --limit) limit=${2:-200}; shift 2 ;;
      --refresh) refresh=1; shift ;;
      --) shift; passthru=("$@"); break ;;
      --*) echo "codex_resume: unknown option: $1" >&2; return 2 ;;
      *) passthru+=("$1"); shift ;;
    esac
  done

  if [[ -z "$target_dir" ]]; then
    target_dir=$(pwd -P)
  else
    if [[ -d "$target_dir" ]]; then
      target_dir=$(cd "$target_dir" 2>/dev/null && pwd -P)
    else
      echo "codex_resume: --dir path does not exist: $target_dir" >&2
      return 2
    fi
  fi

  # Helpers: ripgrep detection and dir extraction
  local __have_rg=0; command -v rg >/dev/null 2>&1 && __have_rg=1
  __codex_extract_dir() {
    local f="$1" line path
    if (( __have_rg )); then
      line=$(rg -m 1 -o 'CurrentDir=[^\n]+' --no-line-number -- "$f" 2>/dev/null)
    else
      line=$(grep -m 1 -ao "CurrentDir=[^\n]*" -- "$f" 2>/dev/null)
    fi
    [[ -z "$line" ]] && return 1
    path=${line#CurrentDir=}
    path=${path%$'\a'*}
    path=${path//$'\e'\]*/}
    path=${path//$'\e'\[[0-9;]#[a-zA-Z]/}
    print -r -- "$path"
    return 0
  }

  __codex_matches_dir_file() {
    local f="$1" curdir
    if curdir=$(__codex_extract_dir "$f" 2>/dev/null); then
      [[ "$curdir" == "$target_dir" ]] && return 0
    fi
    if (( __have_rg )); then
      rg -m 1 -F --quiet -- "$target_dir" "$f" 2>/dev/null && return 0
    else
      grep -m 1 -F -q -- "$target_dir" "$f" 2>/dev/null && return 0
    fi
    return 1
  }

  # Append a single match row to cache TSV (mtime\tfile\tcurdir-or-)
  __codex_append_tsv() {
    local mtime="$1" file="$2" cur="-"
    if cur=$(__codex_extract_dir "$file" 2>/dev/null); then :; else cur="-"; fi
    print -r -- "$mtime$sep$file$sep$cur" >> "$tsv_path"
  }

  # Per-workdir cache under ~/.codex/session_cache
  local cache_root="$HOME/.codex/session_cache"
  [[ -d "$cache_root" ]] || mkdir -p "$cache_root" 2>/dev/null
  local key hash_cmd
  if hash_cmd=$(command -v shasum); then
    key=$(echo -n "$target_dir" | shasum -a 256 | awk '{print $1}')
  else
    key=$(echo -n "$target_dir" | openssl dgst -sha256 2>/dev/null | awk '{print $2}')
  fi
  [[ -n "$key" ]] || key=${${target_dir//\//_}// /_}
  local meta_path="$cache_root/$key.meta"
  local tsv_path="$cache_root/$key.tsv"

  # Read meta (if present)
  local last_date="0000-00-00" last_mtime=0 fmt_ver=1 meta_workdir=""
  if [[ -f "$meta_path" && $refresh -eq 0 ]]; then
    local __cr_meta_line k v
    while IFS='=' read -r k v; do
      case "$k" in
        workdir) meta_workdir="$v" ;;
        format_version) fmt_ver="$v" ;;
        last_scan_date) last_date="$v" ;;
        last_scan_mtime) last_mtime="$v" ;;
      esac
    done < "$meta_path"
  fi
  # If meta workdir mismatches (e.g., moved), reset
  if [[ -n "$meta_workdir" && "$meta_workdir" != "$target_dir" ]]; then
    last_date="0000-00-00"; last_mtime=0
  fi

  # Delta scan date-based sessions to update cache for this workdir
  local -a datedirs
  datedirs=(${~HOME}/.codex/sessions/*/*/*(/N))
  if (( ${#datedirs[@]} == 0 )); then
    echo "No session files found in ~/.codex/sessions" >&2
    return 1
  fi
  IFS=$'\n' datedirs=($(printf '%s\n' ${datedirs[@]} | sort))
  unset IFS

  local max_date="$last_date" max_mtime=$last_mtime
  local dd yyyy mm ddpart date_str __cr_fp __cr_mtime
  exec {__cr_stderr_fd}>&2
  {
    for dd in ${datedirs[@]}; do
      yyyy=${dd:h:h:t}; mm=${dd:h:t}; ddpart=${dd:t}
      date_str="$yyyy-$mm-$ddpart"
      [[ "$date_str" < "$last_date" ]] && continue
      local max_mtime_this_date=0
      local -a files_in_date
      files_in_date=(${dd}/*.jsonl(NOn))
      for __cr_fp in ${files_in_date[@]}; do
        __cr_mtime=$(stat -f %m -- "$__cr_fp" 2>/dev/null)
        (( __cr_mtime > max_mtime_this_date )) && max_mtime_this_date=$__cr_mtime
        if [[ "$date_str" > "$last_date" || ( "$date_str" == "$last_date" && __cr_mtime -gt last_mtime ) ]]; then
          if __codex_matches_dir_file "$__cr_fp"; then
            __codex_append_tsv "$__cr_mtime" "$__cr_fp"
          fi
        fi
      done
      if [[ "$date_str" > "$max_date" ]]; then
        max_date="$date_str"; max_mtime=$max_mtime_this_date
      elif [[ "$date_str" == "$max_date" ]]; then
        (( max_mtime_this_date > max_mtime )) && max_mtime=$max_mtime_this_date
      fi
    done
  } 2>/dev/null
  exec 2>&$__cr_stderr_fd
  exec {__cr_stderr_fd}>&-

  # Write meta atomically
  {
    echo "workdir=$target_dir"
    echo "format_version=1"
    echo "last_scan_date=$max_date"
    echo "last_scan_mtime=$max_mtime"
  } > "$meta_path.tmp.$$" 2>/dev/null && mv -f "$meta_path.tmp.$$" "$meta_path" 2>/dev/null

  # Compact TSV cache: keep newest unique entries, drop missing files
  if [[ -f "$tsv_path" ]]; then
    local -a __tsv
    IFS=$'\n' __tsv=($(LC_ALL=C sort -t"$sep" -k1,1nr "$tsv_path" 2>/dev/null))
    unset IFS
    typeset -A __seen
    local __line __path __keep=0 __max_keep=2000
    : > "$tsv_path.tmp.$$"
    for __line in ${__tsv[@]}; do
      __path=${${__line#*$sep}%%$sep*}
      [[ -f "$__path" ]] || continue
      [[ -n ${__seen[$__path]} ]] && continue
      print -r -- "$__line" >> "$tsv_path.tmp.$$"
      __seen[$__path]=1
      (( ++__keep >= __max_keep )) && break
    done
    mv -f "$tsv_path.tmp.$$" "$tsv_path" 2>/dev/null || rm -f "$tsv_path.tmp.$$" 2>/dev/null
  fi

  local -a matched
  if (( list_any )); then
    # Fallback to live scan across all sessions (recent limited)
    local -a files
    if files=(${~HOME}/.codex/sessions/*/*/*/*.jsonl(Nom[1,$limit])); then
      matched=(${files})
    else
      echo "No session files found in ~/.codex/sessions" >&2
      return 1
    fi
  else
    # Load from cache TSV, dedupe, validate existence, sort by mtime desc
    if [[ -f "$tsv_path" ]]; then
      local -a tsv_lines
      IFS=$'\n' tsv_lines=($(LC_ALL=C sort -t"$sep" -k1,1nr "$tsv_path" 2>/dev/null))
      unset IFS
      typeset -A seen
      local __cr_line __cr_fpath
      for __cr_line in ${tsv_lines[@]}; do
        __cr_fpath=${${__cr_line#*$sep}%%$sep*}
        [[ -f "$__cr_fpath" ]] || continue
        if [[ -z ${seen[$__cr_fpath]} ]]; then
          matched+="$__cr_fpath"
          seen[$__cr_fpath]=1
        fi
        (( ${#matched[@]} >= limit )) && break
      done
    fi
  fi

  local count=${#matched[@]}
  if (( count == 0 )); then
    echo "No sessions found for: $target_dir" >&2
    echo "Tip: use 'codex_resume --any' to browse all recent sessions." >&2
    return 1
  fi

  local picked
  if (( want_last || count == 1 )); then
    picked="$matched[1]"
  else
    local -a menu_lines
    local __cr_pf ts base label
    base=${target_dir:t}
    for __cr_pf in ${matched[@]}; do
      if ts=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" -- "$__cr_pf" 2>/dev/null); then :; else
        local m; m=$(stat -f %m -- "$__cr_pf" 2>/dev/null)
        ts=$(date -r "$m" +"%Y-%m-%d %H:%M" 2>/dev/null)
      fi
      # Remove "rollout-" prefix and ".jsonl" suffix from filename
      clean_filename=${${__cr_pf:t}#rollout-}
      clean_filename=${clean_filename%.jsonl}
      # Add colors: gray timestamp, green base, gray separator, blue filename
      label=$'\033[90m'"$ts"$'\033[0m  \033[32m'"$base"$'\033[0m  \033[90m—\033[0m  \033[94m'"$clean_filename"$'\033[0m'
      menu_lines+="$label$sep$__cr_pf"
    done

    if command -v fzf >/dev/null 2>&1 && [[ -t 1 ]]; then
      local sel
      # Build colorized preview command
      local preview_cmd='
        f={2}
        if [[ ! -f "$f" ]]; then
          echo "File not found: $f"
          exit 1
        fi

        # Extract session metadata for prettier display
        if command -v jq >/dev/null 2>&1; then
          session_id=$(head -1 "$f" 2>/dev/null | jq -r ".id // \"\"" 2>/dev/null)
          branch=$(head -1 "$f" 2>/dev/null | jq -r ".git.branch // \"\"" 2>/dev/null)
          commit=$(head -1 "$f" 2>/dev/null | jq -r ".git.commit_hash // \"\"" 2>/dev/null | cut -c1-8)
          repo_url=$(head -1 "$f" 2>/dev/null | jq -r ".git.repository_url // \"\"" 2>/dev/null)
          timestamp=$(head -1 "$f" 2>/dev/null | jq -r ".timestamp // \"\"" 2>/dev/null | cut -c1-19 | tr T " ")
          instruction=$(head -1 "$f" 2>/dev/null | jq -r ".instructions // \"\"" 2>/dev/null | head -c 120)
        else
          session_id=$(grep -m 1 -o "\"id\":\"[^\"]*\"" "$f" 2>/dev/null | sed "s/\"id\":\"\([^\"]*\)\"/\1/")
          branch=$(grep -m 1 -o "\"branch\":\"[^\"]*\"" "$f" 2>/dev/null | sed "s/\"branch\":\"\([^\"]*\)\"/\1/")
          commit=$(grep -m 1 -o "\"commit_hash\":\"[^\"]*\"" "$f" 2>/dev/null | sed "s/\"commit_hash\":\"\([^\"]*\)\"/\1/" | cut -c1-8)
          repo_url=$(grep -m 1 -o "\"repository_url\":\"[^\"]*\"" "$f" 2>/dev/null | sed "s/\"repository_url\":\"\([^\"]*\)\"/\1/")
          timestamp=$(grep -m 1 -o "\"timestamp\":\"[^\"]*\"" "$f" 2>/dev/null | sed "s/\"timestamp\":\"\([^\"]*\)\"/\1/" | cut -c1-19 | tr T " ")
          instruction=$(grep -m 1 -o "\"instructions\":\"[^\"]*\"" "$f" 2>/dev/null | sed "s/\"instructions\":\"\([^\"]*\)\"/\1/" | head -c 120)
        fi

        # Get repo name from URL
        repo_name=$(echo "$repo_url" | sed "s|.*[/:]||" | sed "s|\.git||")

        # Count messages
        msg_count=$(grep -c "\"type\":\"message\"" "$f" 2>/dev/null || echo "0")

        # Show session header with dynamic width
        header_line1="Session: ${session_id:-unknown}"
        header_line2="${repo_name:-unknown}"
        [[ -n "$branch" ]] && header_line2="$header_line2 ($branch)"
        header_line3="$timestamp"
        [[ "$msg_count" != "0" ]] && header_line3="$header_line3 • $msg_count msgs"

        # Calculate max width needed
        max_width=60
        line1_len=${#header_line1}
        line2_len=${#header_line2}
        line3_len=${#header_line3}

        [[ $line1_len -gt $max_width ]] && max_width=$line1_len
        [[ $line2_len -gt $max_width ]] && max_width=$line2_len
        [[ $line3_len -gt $max_width ]] && max_width=$line3_len

        # Add padding
        ((max_width += 4))

        # Create top border
        printf "\033[90m┌"
        printf "%.0s─" $(seq 1 $max_width)
        printf "┐\033[0m\n"

        # Session ID line
        printf "\033[90m│\033[0m \033[33m%s\033[0m" "$header_line1"
        padding=$((max_width - line1_len - 1))
        printf "%*s" $padding ""
        printf "\033[90m│\033[0m\n"

        # Repo/branch line
        printf "\033[90m│\033[0m \033[32m%s\033[0m" "$repo_name"
        [[ -n "$branch" ]] && printf " \033[90m(\033[36m%s\033[90m)\033[0m" "$branch"
        # Calculate actual printed length (without color codes)
        actual_len=${#repo_name}
        [[ -n "$branch" ]] && actual_len=$((actual_len + ${#branch} + 3))
        padding=$((max_width - actual_len - 1))
        printf "%*s" $padding ""
        printf "\033[90m│\033[0m\n"

        # Timestamp/messages line
        printf "\033[90m│\033[0m \033[37m%s\033[0m" "$timestamp"
        [[ "$msg_count" != "0" ]] && printf " \033[90m•\033[0m \033[37m%s msgs\033[0m" "$msg_count"
        padding=$((max_width - line3_len - 1))
        printf "%*s" $padding ""
        printf "\033[90m│\033[0m\n"

        # Bottom border
        printf "\033[90m└"
        printf "%.0s─" $(seq 1 $max_width)
        printf "┘\033[0m\n\n"

        # Show full conversation history
        if command -v jq >/dev/null 2>&1; then
          sed -E "s/\x1b\\[[0-9;]*[A-Za-z]//g" "$f" | jq -Rr "fromjson? | select(.type==\"message\") | [.role // \"\", .content[0].text // \"\"] | @tsv" 2>/dev/null | while IFS=$'"'"'\t'"'"' read -r role text; do
            # Skip messages with empty/null roles or text
            if [[ -z "$role" || -z "$text" ]]; then
              continue
            fi

            # Skip template blocks but show real content
            if [[ "$text" == "<user_instructions>"* ]] || [[ "$text" == "<environment_context>"* ]]; then
              continue
            fi

            # Format role with color
            if [[ "$role" == "user" ]]; then
              printf "\033[94m▌ User:\033[0m\n"
            elif [[ "$role" == "assistant" ]]; then
              printf "\033[92m▌ Assistant:\033[0m\n"
            else
              printf "\033[90m▌ %s:\033[0m\n" "$role"
            fi

            # Show message content (wrapped and formatted)
            echo "$text" | fold -s -w 80 | sed "s/^/  /"
            printf "\n"
          done
        else
          # Fallback: show raw content with basic formatting
          printf "\033[90mFull conversation:\033[0m\n\n"
          grep "\"type\":\"message\"" "$f" | while IFS= read -r line; do
            # Extract role and basic text without jq
            role=$(echo "$line" | sed -n "s/.*\"role\":\"\([^\"]*\)\".*/\1/p")
            if [[ "$role" == "user" ]]; then
              printf "\033[94m▌ User:\033[0m\n"
            elif [[ "$role" == "assistant" ]]; then
              printf "\033[92m▌ Assistant:\033[0m\n"
            fi
            echo "  [Message content]"
            printf "\n"
          done
        fi
      '
      sel=$(printf '%s\n' ${menu_lines[@]} | fzf --ansi --with-nth=1 --delimiter="$sep" --prompt="codex sessions » " --height=80% --reverse --preview-window=up:60% --preview="$preview_cmd") || return 130
      picked=${sel#*$sep}
    else
      local i=1; for f in ${matched[@]}; do
        echo "[$i] ${f:t}"
        (( i++ ))
      done
      local choice
      printf "Select [1-%d]: " $((i-1))
      read choice || return 130
      if [[ -z "$choice" || ! "$choice" == <-> || $choice -lt 1 || $choice -ge $i ]]; then
        echo "Invalid selection" >&2; return 2
      fi
      picked="$matched[$choice]"
    fi
  fi

  [[ -z "$picked" ]] && { echo "No session selected" >&2; return 1; }
  echo "Resuming from: $picked"
  if (( ${#passthru[@]} )); then
    codex --config experimental_resume="$picked" "${passthru[@]}"
  else
    codex --config experimental_resume="$picked"
  fi
}