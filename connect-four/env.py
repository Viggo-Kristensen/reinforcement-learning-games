import random
import collections
import numpy as np
import pygame

SCREEN_HEIGHT = 708
SCREEN_WIDTH = 822

class ConnectFourEnv:
    def __init__(self, render=False):
        self.render_mode=render
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.player = 1
        self.grid = [[0 for r in range(6)] for c in range(7)]
        self.valid_columns = [0, 1, 2, 3, 4, 5, 6]
        self.invalid_columns = []
        self.column_heights = [0] * 7
        self.done = False
        return self._get_state()

    def step(self, action):
        if self.render_mode:
            pygame.event.pump()  # keep window responsive

        # Actions: 0 1 2 3 4 5 6 Drop in i'th column agent 1
        self.grid[action][self.column_heights[action]] = self.player

        if self.column_heights[action] == 5:
            self.valid_columns.remove(action)
            self.invalid_columns.append(action)
        c_placed, r_placed = action, self.column_heights[action]
        self.column_heights[action] += 1

        won = False

        # Horizontal
        for r in range(6):
            for c in range(0, 4):
                if self.grid[c][r] == self.grid[c + 1][r] == self.grid[c + 2][r] == self.grid[c + 3][r] == self.player:
                    won = True
                    self.done = True

        # Vertical
        for r in range(0, 3):
            for c in range(7):
                if self.grid[c][r] == self.grid[c][r + 1] == self.grid[c][r + 2] == self.grid[c][r + 3] == self.player:
                    won = True
                    self.done = True

        # Diagonal up
        for r in range(0, 3):
            for c in range(0, 4):
                if self.grid[c][r] == self.grid[c + 1][r + 1] == self.grid[c + 2][r + 2] == self.grid[c + 3][r + 3] == self.player:
                    won = True
                    self.done = True

        # Diagonal down
        for r in range(3, 6):
            for c in range(0, 4):
                if self.grid[c][r] == self.grid[c + 1][r - 1] == self.grid[c + 2][r - 2] == self.grid[c + 3][r - 3] == self.player:
                    won = True
                    self.done = True


        # reward if action blocked four
        blocked = False
        if not won:
            self.grid[c_placed][r_placed] *= -1

            # Horizontal
            for r in range(6):
                for c in range(0, 4):
                    if self.grid[c][r] == self.grid[c + 1][r] == self.grid[c + 2][r] == self.grid[c + 3][r] != 0:
                        blocked = True

            # Vertical
            for r in range(0, 3):
                for c in range(7):
                    if self.grid[c][r] == self.grid[c][r + 1] == self.grid[c][r + 2] == self.grid[c][r + 3] != 0:
                        blocked = True

            # Diagonal up
            for r in range(0, 3):
                for c in range(0, 4):
                    if self.grid[c][r] == self.grid[c + 1][r + 1] == self.grid[c + 2][r + 2] == self.grid[c + 3][r + 3] != 0:
                        blocked = True

            # Diagonal down
            for r in range(3, 6):
                for c in range(0, 4):
                    if self.grid[c][r] == self.grid[c + 1][r - 1] == self.grid[c + 2][r - 2] == self.grid[c + 3][r - 3] != 0:
                        blocked = True

            self.grid[c_placed][r_placed] *= -1


        # Reward based on number of 3s
        threes = 0

        # Horizontal
        for r in range(6):
            for c in range(0, 4):
                count = {0: 0, 1: 0, -1: 0}
                count[self.grid[c][r]] += 1
                count[self.grid[c + 1][r]] += 1
                count[self.grid[c + 2][r]] += 1
                count[self.grid[c + 3][r]] += 1

                if count[0] == 1 and count[self.player] == 3:
                    threes += 1

        # Vertical
        for r in range(0, 3):
            for c in range(7):
                count = {0: 0, 1: 0, -1: 0}
                count[self.grid[c][r]] += 1
                count[self.grid[c][r + 1]] += 1
                count[self.grid[c][r + 2]] += 1
                count[self.grid[c][r + 3]] += 1

                if count[0] == 1 and count[self.player] == 3:
                    threes += 1

        # Diagonal up
        for r in range(0, 3):
            for c in range(0, 4):
                count = {0: 0, 1: 0, -1: 0}
                count[self.grid[c][r]] += 1
                count[self.grid[c + 1][r + 1]] += 1
                count[self.grid[c + 2][r + 2]] += 1
                count[self.grid[c + 3][r + 3]] += 1

                if count[0] == 1 and count[self.player] == 3:
                    threes += 1

        # Diagonal down
        for r in range(3, 6):
            for c in range(0, 4):
                count = {0: 0, 1: 0, -1: 0}
                count[self.grid[c][r]] += 1
                count[self.grid[c + 1][r - 1]] += 1
                count[self.grid[c + 2][r - 2]] += 1
                count[self.grid[c + 3][r - 3]] += 1

                if count[0] == 1 and count[self.player] == 3:
                    threes += 1


        # give negative reward if four is possible after move Not done yet
        # Horizontal
        opp_win = False
        for r in range(6):
            for c in range(0, 4):
                count = {0: 0, 1: 0, -1: 0}

                line = [self.grid[c][r],
                        self.grid[c + 1][r],
                        self.grid[c + 2][r],
                        self.grid[c + 3][r]
                        ]
                line_chords = [[c, r], [c + 1, r], [c + 2, r], [c + 3, r]]

                for v in line:
                    count[v] += 1

                if count[0] == 1 and count[-1 * self.player] == 3:
                    for i, v in enumerate(line):
                        if v == 0:
                            r0, c0 = line_chords[i]
                            if self.column_heights[c0] == r0:
                                opp_win = True


        # Vertical
        for r in range(0, 3):
            for c in range(7):
                count = {0: 0, 1: 0, -1: 0}

                line = [self.grid[c][r],
                        self.grid[c][r + 1],
                        self.grid[c][r + 2],
                        self.grid[c][r + 3]
                        ]
                line_chords = [[c, r], [c, r + 1], [c, r + 2], [c, r + 3]]

                for v in line:
                    count[v] += 1

                if count[0] == 1 and count[-1 * self.player] == 3:
                    for i, v in enumerate(line):
                        if v == 0:
                            r0, c0 = line_chords[i]
                            if self.column_heights[c0] == r0:
                                opp_win = True

        # Diagonal up
        for r in range(0, 3):
            for c in range(0, 4):
                count = {0: 0, 1: 0, -1: 0}

                line = [self.grid[c][r],
                        self.grid[c + 1][r + 1],
                        self.grid[c + 2][r + 2],
                        self.grid[c + 3][r + 3]
                        ]
                line_chords = [[c, r], [c + 1, r + 1], [c + 2, r + 2], [c + 3, r + 3]]

                for v in line:
                    count[v] += 1

                if count[0] == 1 and count[-1 * self.player] == 3:
                    for i, v in enumerate(line):
                        if v == 0:
                            r0, c0 = line_chords[i]
                            if self.column_heights[c0] == r0:
                                opp_win = True

        # Diagonal down
        for r in range(3, 6):
            for c in range(0, 4):
                count = {0: 0, 1: 0, -1: 0}

                line = [self.grid[c][r],
                        self.grid[c + 1][r - 1],
                        self.grid[c + 2][r - 2],
                        self.grid[c + 3][r - 3]
                        ]
                line_chords = [[c, r], [c + 1, r - 1], [c + 2, r - 2], [c + 3, r - 3]]

                for v in line:
                    count[v] += 1

                if count[0] == 1 and count[-1 * self.player] == 3:
                    for i, v in enumerate(line):
                        if v == 0:
                            r0, c0 = line_chords[i]
                            if self.column_heights[c0] == r0:
                                opp_win = True

        # Punish for stacking up
        # stacking = False
        # if r_placed != 0 and self.grid[c_placed][r_placed-1] == 1:
        #     stacking = True


        # Reward
        reward = 0.0
        reward += threes * 0.06

        # if stacking:
        #    reward -= 0.35

        if won:
            reward += 1
        elif blocked:
            reward += 0.2
        elif opp_win:
            reward -= 0.7

        self.player *= -1

        return self._get_state(), reward, self.done, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        self.screen.fill((0, 0, 255))
        x = 69
        for i in range(7):
            y = SCREEN_HEIGHT - 69
            for j in range(6):
                if self.grid[i][j] == 0:
                    pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 45)
                elif self.grid[i][j] == 1:
                    pygame.draw.circle(self.screen, (255, 255, 0), (x, y), 45)
                else:
                    pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 45)
                y -= 114
            x += 114
        pygame.display.update()
        self.clock.tick(4)


    def _get_state(self):
        grid = np.array(self.grid, dtype=np.float32)
        if self.player == -1:
            grid *= -1

        feature_1 = np.where(grid == -1, grid, 0)
        feature_2 = np.where(grid == 1, grid, 0)

        return np.stack((feature_1, feature_2))

    def close(self):
        if self.render_mode:
            try:
                pygame.display.quit()
            except Exception:
                pass
            self.render_mode = False


# Replay Buffer
Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return (Transition(*zip(*batch)))
    def __len__(self):
        return len(self.buffer)