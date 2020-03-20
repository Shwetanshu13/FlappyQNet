import pygame
from pygame import Rect
from random import randint
from collections import deque
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

SCREEN_HEIGHT = 620
SCREEN_WIDTH = 960
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Bird:
    def __init__(self):
        self.rect = Rect(200, 310, 20, 20)
        self.dy = 0
        self.ddy = 20
        self.state = "alive"

    def update(self):
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT
            self.dy = 0
            self.state = "dead"
        if self.rect.top < 0:
            self.rect.top = 0
        self.dy += self.ddy / 30
        self.rect.y += self.dy

    def flap(self):
        self.dy = -10

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)


class Pipes:
    def __init__(self):
        self.width = 40
        self.height1 = randint(0, 540)
        self.height2 = 620 - (self.height1 + 140)
        self.rect1 = Rect(SCREEN_WIDTH, SCREEN_HEIGHT - self.height1, self.width, self.height1)
        self.rect2 = Rect(SCREEN_WIDTH, 0, self.width, self.height2)
        self.dx = -5

    def update(self):
        self.rect1.x += self.dx
        self.rect2.x += self.dx

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect1)
        pygame.draw.rect(screen, WHITE, self.rect2)


class Game:
    def __init__(self):
        self.score = 0
        self.collision = False
        self.bird = Bird()
        self.buffer = deque(maxlen=5)
        self.frame = 0
        self.fps = 60
        self.clock = pygame.time.Clock()
        #self.font = pygame.font.Font("/Users/Snehal/Downloads/kenvector_future_thin.ttf", 26)
        #self.text = self.font.render(str(self.score), True, (255, 0, 0))
        #self.textRect = self.text.get_rect()
        #self.textRect.center = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

    def reset(self):
        self.score = 0
        self.buffer.clear()
        self.bird.rect.y = 310
        self.bird.dy = 0
        self.frame = 0
        self.buffer.appendleft(Pipes())
        self.collision = False
        self.bird.state = "alive"

    def append_pipe(self):
        if self.frame % 40 == 0:
            self.buffer.appendleft(Pipes())

    def event_processor(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.exit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.bird.flap()

    def draw(self, screen):
        #screen.fill(BLACK)
        #screen.blit(self.text, self.textRect)
        self.bird.draw(screen)
        for i in self.buffer:
            i.draw(screen)
        pygame.display.flip()

    def update(self):
        self.bird.update()
        for i in self.buffer:
            i.update()

    def time(self):
        self.clock.tick(self.fps)
        self.frame += 1
        self.score += 1

    def collision_detection(self):
        try:
            if self.buffer[-2].rect1.x <= self.bird.rect.x <= self.buffer[-2].rect1.x + self.buffer[-2].width:
                if self.bird.rect.top <= self.buffer[-2].rect2.bottom or self.bird.rect.bottom >= self.buffer[-2].rect1.top:
                    self.collision = True
                    print(2)
                    print(len(self.buffer))
            elif self.buffer[-1].rect1.x <= self.bird.rect.x <= self.buffer[-1].rect1.x + self.buffer[-1].width:
                if self.bird.rect.top <= self.buffer[-1].rect2.bottom or self.bird.rect.bottom >= self.buffer[-1].rect1.top:
                    self.collision = True
                    print(1)
                    print(len(self.buffer))
        except IndexError:
            pass

        if self.collision or self.bird.state == "dead":
            self.score -= 30
            self.reset()


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]


class DeepQNetwork:
    def __init__(self):
        self.model = self.build_net()
        self.epsilon = 1
        self.min_eps = 0.01
        self.gamma = 0.9

    def build_net(self):
        model = tf.keras.Sequential()
        model.add(Dense(10, activation='relu', input_shape=(5, )))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(2, activation='softmax')) # index 0 is no jump and index 1 is jump
        model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])
        return model

    def act(self, state, bird):
        random = np.random.random()
        if random > self.epsilon:
            q_values = self.model.predict(state)
            action = np.argmax(q_values)
            if action[1]:
                bird.flap()
            else:
                pass
        else:
            random_num = np.random.randint(0, 99)
            if random_num > 49:
                bird.flap()
            else:
                pass

    def train(self, memory, target_network):
        temp_mem = memory.sample(40)
        for i in temp_mem:
            if i[3] == "terminal":
                targets = i[2]
            else:
                targets = np.amax(target_network.model.predict(i[3])) + i[2]
            x = i[0]
            y = self.model.predict(x)
            y[i[1]] = targets
            self.model.fit(x, y, epochs=1, verbose=0)
        if self.epsilon < self.min_eps:
            self.epsilon *= .995



def main():
    # initialize pygame
    pygame.init()

    # setup
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    game = Game()
    memory = Memory(100000)
    # initialize networks
    '''prediction_net = DeepQNetwork()
    target_net = DeepQNetwork()
    target_net.model.set_weights(prediction_net.model.get_weights)'''

    while True:
        # define the state
        height = game.bird.rect.y
        velocity = game.bird.dy
        if len(game.buffer) < 5:
            try:
                x_distance = game.buffer[-1].rect1.x - game.bird.rect.right
                y_top_distance = game.bird.y - game.buffer[-1].rect2.bottom
                y_bottom_distance = game.buffer[-1].rect1.top - game.bird.bottom
                state = (height, x_distance, y_bottom_distance, y_top_distance, velocity)
            except:
                pass
        elif len(game.buffer) == 5:
            x_distance = game.buffer[-2].rect1.x - game.bird.rect.right
            y_top_distance = game.bird.y - game.buffer[-2].rect2.bottom
            y_bottom_distance = game.buffer[-2].rect1.top - game.bird.bottom
            state = (height, x_distance, y_bottom_distance, y_top_distance, velocity)

        # initialize a list (state_t, action_t, reward_t, state_t+1)
        replay_list = []
        try:
            replay_list.append(state)
        except:
            pass

        # process events
        game.event_processor()

        # every 120px add a pipe
        game.append_pipe()

        # collision detection
        game.collision_detection()

        # drawing cheese
        game.update()
        game.draw(screen)

        # add action_t, reward_t, state_t=1
        #ac

        # finish
        game.time()


if __name__ == "__main__":
    main()