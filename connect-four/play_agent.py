import torch
import numpy as np
import pygame
import random
from model import QNetwork

SCREEN_HEIGHT = 708
SCREEN_WIDTH = 822

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.font = pygame.font.SysFont(None, 56)
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.grid = [[0 for r in range(6)] for c in range(7)]
        self.valid_columns = [0, 1, 2, 3, 4, 5, 6]
        self.invalid_columns = []
        self.turn = random.randint(0, 1)
        self.done = False


    def place(self, column, player):
        if column in self.valid_columns:
            for idx, felt in enumerate(self.grid[column]):
                if self.grid[column][idx] == 0:
                    self.grid[column][idx] = player
                    self.turn += 1
                    if idx == 5:
                        self.valid_columns.remove(column)
                        self.invalid_columns.append(column)
                    break

    def check(self, x, player):
        c_mid = 69
        for i in range(7):
            if c_mid - 22.5 < x < c_mid + 22.5:
                c_placed = i
                break
            else:
                c_mid += 114
        try:
            c_placed
        except NameError:
            pass
        else:
            self.place(c_placed, player)

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
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

        if game.check_win():
            if game.turn % 2 == 0:
                text = game.font.render("AI Won", True, (255, 255, 255))
            else:
                text = game.font.render("You Won", True, (255, 255, 255))
            game.screen.blit(text, (324, 321))

        pygame.display.update()
        self.clock.tick(10)

    def _get_state(self):
        grid = np.array(self.grid, dtype=np.float32)
        feature_1 = np.where(grid == -1, grid, 0)
        feature_2 = np.where(grid == 1, grid, 0)
        return np.stack((feature_1, feature_2))

    def check_win(self):
        # Horizontal
        for r in range(6):
            for c in range(0, 4):
                if self.grid[c][r] == self.grid[c + 1][r] == self.grid[c + 2][r] == self.grid[c + 3][r] != 0:
                    return True

        # Vertical
        for r in range(0, 3):
            for c in range(7):
                if self.grid[c][r] == self.grid[c][r + 1] == self.grid[c][r + 2] == self.grid[c][r + 3] != 0:
                    return True

        # Diagonal up
        for r in range(0, 3):
            for c in range(0, 4):
                if self.grid[c][r] == self.grid[c + 1][r + 1] == self.grid[c + 2][r + 2] == self.grid[c + 3][r + 3] != 0:
                    return True

        # Diagonal down
        for r in range(3, 6):
            for c in range(0, 4):
                if self.grid[c][r] == self.grid[c + 1][r - 1] == self.grid[c + 2][r - 2] == self.grid[c + 3][r - 3] != 0:
                    return True

        return False


hidden_dim = 128
output_dim = 7

game = Game()
policy_net = QNetwork(hidden_dim, output_dim)
policy_net.load_state_dict(torch.load("connect_four_agent_2"))

while not game.done:
    if game.turn % 2 == 0:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                game.check(x, -1)
    else:
        state = game._get_state()  # shape (2, 7, 6), numpy array
        state_tensor = torch.from_numpy(state).unsqueeze(0).float()  # shape (1, 2, 7, 6)
        policy_net.eval()
        with torch.no_grad():
            qvals = policy_net(state_tensor)
            for c in game.invalid_columns:
                qvals[0, c] = float('-inf')
        action = int(torch.argmax(qvals, dim=1).item())
        game.place(action, 1)
        pygame.time.delay(400)

    game.render()


    if game.check_win():
        pygame.time.delay(3000)
        pygame.display.quit()








