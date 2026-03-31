from agents.agent import Agent


class HumanAgent(Agent):
    """Interactive human player for Dots and Boxes.

    Renders the board with coordinate labels and prompts the player
    to enter moves as (row, col) on the visual grid.
    """

    def __init__(self, env):
        super().__init__(env)

    def act(self):
        """Display board, list available moves, and prompt for input."""
        available = self.env._get_available_actions()
        self._render_board()
        self._show_available(available)

        while True:
            try:
                text = input(
                    f"\n  Player {self.env.current_player} ▸ Enter move (row,col): "
                ).strip()

                if text.lower() in ("q", "quit", "exit"):
                    raise KeyboardInterrupt("Player quit")
                if text.lower() == "help":
                    self._print_help()
                    continue

                # Parse as  row,col  or  row col
                parts = (
                    text.replace(",", " ")
                    .replace("(", "")
                    .replace(")", "")
                    .split()
                )
                if len(parts) != 2:
                    print("  ✗ Enter two numbers: row col  (e.g. 0 1 or 0,1)")
                    continue

                row, col = int(parts[0]), int(parts[1])
                action = self.action_from_rc(row, col)

                if action not in available:
                    print(f"  ✗ ({row},{col}) is already placed or out of range.")
                    continue

                return action

            except ValueError as exc:
                print(f"  ✗ {exc}")
            except (EOFError, KeyboardInterrupt):
                print("\n  Game aborted.")
                raise

    # ── board rendering ────────────────────────────────────────────

    def _render_board(self):
        env = self.env
        rows, cols = env.rows, env.cols

        print(f"\n  ╔{'═' * 42}╗")
        print(
            f"  ║  Player {env.current_player}'s turn"
            f"{'':>{16}}"
            f"P1: {env.score[0]}  P2: {env.score[1]}  ║"
        )
        print(f"  ╚{'═' * 42}╝\n")

        for i in range(rows + 1):
            # ── horizontal edges row ──
            line = "  "
            for j in range(cols):
                line += "•"
                if env.horizontal_edges[i][j]:
                    line += "═══"
                else:
                    line += " · "
            line += "•"
            print(line)

            # ── vertical edges + boxes row ──
            if i < rows:
                line = "  "
                for j in range(cols + 1):
                    if env.vertical_edges[i][j]:
                        line += "║"
                    else:
                        line += "·"
                    if j < cols:
                        if env.boxes[i][j] == 1:
                            line += " 1 "
                        elif env.boxes[i][j] == 2:
                            line += " 2 "
                        else:
                            line += "   "
                print(line)

        if env.done:
            if env.score[0] > env.score[1]:
                print("  ★ Player 1 wins!")
            elif env.score[1] > env.score[0]:
                print("  ★ Player 2 wins!")
            else:
                print("  ★ It's a draw!")

    # ── available-move listing ─────────────────────────────────────

    def _show_available(self, actions):
        horiz = sorted(
            [self.action_to_rc(a) for a in actions if a[0] == 0]
        )
        vert = sorted(
            [self.action_to_rc(a) for a in actions if a[0] == 1]
        )

        def _fmt(lst, per_line=8):
            parts = [f"({r},{c})" for r, c in lst]
            lines = []
            for i in range(0, len(parts), per_line):
                lines.append("  ".join(parts[i : i + per_line]))
            return lines

        print(f"\n  Available moves [{len(actions)}]:")
        if horiz:
            print("    Horizontal ─ :", end="")
            for ln in _fmt(horiz):
                print(f"  {ln}")
        if vert:
            print("    Vertical   │ :", end="")
            for ln in _fmt(vert):
                print(f"  {ln}")

    # ── help ───────────────────────────────────────────────────────

    @staticmethod
    def _print_help():
        print(
            """
  ╔════════════════════════════════════════════════════════════╗
  ║  HOW TO PLAY                                              ║
  ╠════════════════════════════════════════════════════════════╣
  ║  Enter moves as:  row,col   (e.g. 0,1  or  0 1)          ║
  ║                                                            ║
  ║  The board lives on a visual grid where:                   ║
  ║    • Dots  are at even row, even col                       ║
  ║    • Horiz edges at even row, odd  col   (e.g. 0,1)       ║
  ║    • Vert  edges at odd  row, even col   (e.g. 1,0)       ║
  ║                                                            ║
  ║  Complete a box → you get another turn!                    ║
  ║  Type 'quit' to exit.                                      ║
  ╚════════════════════════════════════════════════════════════╝
            """
        )
