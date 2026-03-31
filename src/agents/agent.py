class Agent:
    def __init__(self, env):
        self.env = env

    def act(self):
        pass

    def act_rc(self):
        """Return the chosen action in visual-grid (row, col) format."""
        action = self.act()
        if action is None:
            return None
        return self.action_to_rc(action)

    # ── row,col  ↔  (type, i, j) conversion ──────────────────────

    @staticmethod
    def action_to_rc(action):
        """Convert internal action (type, i, j) → visual-grid (row, col).

        The visual grid has size (2*rows+1) × (2*cols+1):
          - Horizontal edge (0, i, j) → (2*i,   2*j+1)
          - Vertical   edge (1, i, j) → (2*i+1, 2*j)
        """
        if action[0] == 0:
            return (2 * action[1], 2 * action[2] + 1)
        else:
            return (2 * action[1] + 1, 2 * action[2])

    @staticmethod
    def action_from_rc(row, col):
        """Convert visual-grid (row, col) → internal action (type, i, j).

        - Even row, odd col  → horizontal edge (0, row//2, col//2)
        - Odd  row, even col → vertical   edge (1, row//2, col//2)
        """
        if row % 2 == 0 and col % 2 == 1:
            return (0, row // 2, col // 2)
        elif row % 2 == 1 and col % 2 == 0:
            return (1, row // 2, col // 2)
        else:
            raise ValueError(
                f"Invalid edge position ({row}, {col}). "
                "Horizontal edges: (even_row, odd_col), "
                "Vertical edges: (odd_row, even_col)."
            )

    def get_available_actions_rc(self):
        """Return available actions as (row, col) on the visual grid."""
        return [self.action_to_rc(a) for a in self.env._get_available_actions()]

    def train(self):
        pass

    def save(self):
        pass

    def load(self):
        pass