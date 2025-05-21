**Tmux Cheatsheet: Persistent Sessions & Multi-Window Management**

**Core Concept:**
`tmux` allows you to manage multiple terminal sessions, windows (tabs), and panes (splits) within a single terminal window. Sessions persist even if you disconnect.

  * **The Prefix Key:** Most `tmux` commands are preceded by a prefix key.
      * Default: `Ctrl+b` (We'll refer to this as `<prefix>`)
  * **Tmux Command Prompt:** Enter `tmux` commands directly.
      * `<prefix> :` (then type your command and press Enter)

-----

**I. Session Management (Persistent Workspaces)**

Manage your `tmux` sessions from your regular terminal (outside `tmux`) or from within an active `tmux` session.

**From your Terminal (Outside `tmux`):**

  * **Start a new unnamed session:**
    ```bash
    tmux
    ```
  * **Start a new named session:**
    ```bash
    tmux new -s <session_name>
    ```
    *Example:* `tmux new -s myproject`
  * **List active sessions:**
    ```bash
    tmux ls
    ```
  * **Attach to the last used session:**
    ```bash
    tmux attach
    ```
    (or `tmux a`)
  * **Attach to a named session:**
    ```bash
    tmux attach -t <session_name>
    ```
    (or `tmux a -t <session_name>`)
    *Example:* `tmux a -t myproject`
  * **Kill a specific session:**
    ```bash
    tmux kill-session -t <session_name>
    ```
  * **Kill all `tmux` sessions:** (Use with caution\!)
    ```bash
    tmux kill-server
    ```

**From Inside `tmux` (Using `<prefix>`):**

  * **Detach from current session (leave it running in background):**
    `<prefix> d`
  * **List sessions (interactive menu to switch or attach):**
    `<prefix> s`
  * **Rename current session:**
    `<prefix> $`
  * **Switch to the next session:**
    `<prefix> )`
  * **Switch to the previous session:**
    `<prefix> (`

-----

**II. Window Management (Like Tabs)**

Within a session, you can have multiple windows.

  * **Create a new window:**
    `<prefix> c`
  * **Go to the next window:**
    `<prefix> n`
  * **Go to the previous window:**
    `<prefix> p`
  * **Select window by its number (0-9):**
    `<prefix> <number>` (e.g., `<prefix> 1` for window 1)
  * **Rename the current window:**
    `<prefix> ,`
  * **List windows (interactive menu to select):**
    `<prefix> w`
  * **Close the current window (will prompt for confirmation):**
    `<prefix> &`
  * **Move current window one position to the left:**
    `<prefix> :swap-window -t -1` (Enter this at the command prompt)
  * **Move current window one position to the right:**
    `<prefix> :swap-window -t +1` (Enter this at the command prompt)

-----

**III. Pane Management (Splitting Windows)**

Divide windows into multiple panes, each running a separate shell.

  * **Split pane vertically (creates a new pane to the right):**
    `<prefix> %`
  * **Split pane horizontally (creates a new pane below):**
    `<prefix> "`
  * **Navigate between panes:**
    `<prefix> <arrow_key>` (e.g., `<prefix> ←` to move to the pane on the left)
  * **Toggle to the last active pane:**
    `<prefix> ;`
  * **Cycle to the next pane:**
    `<prefix> o`
  * **Close the current pane (will prompt for confirmation):**
    `<prefix> x`
  * **Toggle zoom for the current pane (maximize/restore):**
    `<prefix> z`
  * **Convert current pane into a new window:**
    `<prefix> !`
  * **Swap current pane with the previous pane:**
    `<prefix> {`
  * **Swap current pane with the next pane:**
    `<prefix> }`
  * **Resize current pane (after `<prefix>`):**
      * Hold `Ctrl` then press `<arrow_key>` (e.g., `<prefix>` then `Ctrl+↑` to make taller)
      * *Alternatively, enter command mode (`<prefix> :`)*
          * `resize-pane -D <number_of_cells>` (Down)
          * `resize-pane -U <number_of_cells>` (Up)
          * `resize-pane -L <number_of_cells>` (Left)
          * `resize-pane -R <number_of_cells>` (Right)
            *Example:* `<prefix> :resize-pane -D 5`
  * **Cycle through predefined pane layouts:**
    `<prefix> <spacebar>`

-----

**IV. Useful Extras**

  * **Copy Mode (for scrolling and copying text):**
      * Enter copy mode: `<prefix> [`
      * Navigate: Use `arrow keys`, `PgUp`, `PgDn`. `g` for top, `G` for bottom.
      * Start selection (vi mode): `Spacebar` then move cursor, `Enter` to copy.
      * Start selection (emacs mode): `Ctrl+Spacebar` then move cursor, `Alt+w` to copy.
      * Exit copy mode: `q`
  * **Paste the most recently copied text:**
    `<prefix> ]`
  * **Configuration File:**
      * Located at `~/.tmux.conf`
      * To apply changes after editing: `<prefix> :source-file ~/.tmux.conf`
  * **List all key bindings:**
    `<prefix> ?`
  * **Display time in current pane:**
    `<prefix> t`

-----

Remember to replace `<session_name>` and `<number_of_cells>` with your desired values.