import pygame
from pygame import Rect
from random import randint
from collections import deque
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from time import time
from tensorflow.python.keras.callbacks import TensorBoard


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
            self.state = "dead"
        self.dy += self.ddy / 30
        self.rect.y += self.dy

    def flap(self):
        self.dy = -10

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)


class Pipes:
    def __init__(self):
        self.width = 40
        self.height1 = randint(0, 200)
        self.height2 = 620 - (self.height1 + 330)
        self.rect1 = Rect(SCREEN_WIDTH, SCREEN_HEIGHT - self.height1, self.width, self.height1)
        self.rect2 = Rect(SCREEN_WIDTH, 0, self.width, self.height2)
        self.dx = -10

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
        self.buffer.appendleft(Pipes())
        self.frame = 0
        self.fps = 60
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("/Users/Snehal/Downloads/kenvector_future_thin.ttf", 26)
        self.text = self.font.render(str(self.score), True, (255, 0, 0))
        self.textRect = self.text.get_rect()
        self.textRect.center = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

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
        if self.frame % 40 == 0 and self.frame != 0:
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
        screen.fill(BLACK)
        screen.blit(self.text, self.textRect)
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
            elif self.buffer[-1].rect1.x <= self.bird.rect.x <= self.buffer[-1].rect1.x + self.buffer[-1].width:
                if self.bird.rect.top <= self.buffer[-1].rect2.bottom or self.bird.rect.bottom >= self.buffer[-1].rect1.top:
                    self.collision = True
        except IndexError:
            pass

        if self.collision or self.bird.state == "dead":
            self.score -= 30
            self.reset()
            return 1 # yes there was a collision
        else:
            return 0 # no there wasn't a collision

    def rewards(self, done):
        if done:
            reward = -1
        else:
            reward = 0.004

        if self.frame > 140 and (self.frame - 140) % 40 == 0:
            reward += 1

        return reward


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

        #tensorboard = TensorBoard(log_dir="/Users/Snehal/PycharmProjects/untitled1/logs/{}".format(time()))
        model.compile(optimizer=Adam(lr=0.005), loss="mse")

        return model

    def act(self, state, bird):
        random = np.random.rand()
        if random > self.epsilon:
            print("predicted")
            q_values = self.model.predict(state)
            blah = np.max(q_values)
            action = (q_values == blah)
            if action[0][1]:
                bird.flap()
                return 1
            else:
                return 0
        else:
            random_num = np.random.randint(0, 99)
            print("random")
            if random_num > 92:
                bird.flap()
                return 1
            else:
                return 0

    def train(self, memory, target_network):
        temp_mem = memory.sample(50)
        for i in temp_mem:
            if i[3] == []:
                targets = i[2]
            else:
                targets = (0.9 * np.amax(target_network.model.predict(i[3]))) + i[2]
            x = i[0]
            y = self.model.predict(x)
            y[0][i[1]] = targets
            self.model.fit(x, y, epochs=1, verbose=0)
        if self.epsilon > self.min_eps:
            self.epsilon *= .995


def main():
    # initialize pygame
    pygame.init()

    # setup
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    game = Game()
    memory = Memory(100000)
    score = 0

    # initialize networks
    model = DeepQNetwork()
    model.model.summary()
    target_model = DeepQNetwork()
    target_model.model.summary()
    target_model.model.set_weights(model.model.get_weights())
    q = False
    while not q:
        # define the state
        height = game.bird.rect.y
        velocity = game.bird.dy
        if len(game.buffer) < 5:
            try:
                x_distance = game.buffer[-1].rect1.x - game.bird.rect.right
                y_top_distance = game.bird.rect.y - game.buffer[-1].rect2.bottom
                y_bottom_distance = game.buffer[-1].rect1.top - game.bird.rect.bottom
                state = [height, x_distance, y_bottom_distance, y_top_distance, velocity]
                state = np.array([state])
            except IndexError:
                pass
        elif len(game.buffer) == 5:
            x_distance = game.buffer[-2].rect1.x - game.bird.rect.right
            y_top_distance = game.bird.rect.y - game.buffer[-2].rect2.bottom
            y_bottom_distance = game.buffer[-2].rect1.top - game.bird.rect.bottom
            state = [height, x_distance, y_bottom_distance, y_top_distance, velocity]
            state = np.array([state])

        # initialize a list (state_t, action_t, reward_t, state_t+1)
        replay_list = []
        app_state = True
        try:
            replay_list.append(state)
        except:
            app_state = False

        # process events
        game.event_processor()

        # take an action and record it
        action = model.act(state, game.bird)
        if app_state:
            replay_list.append(action)

        # every 120px add a pipe
        game.append_pipe()

        # collision and reward
        done = game.collision_detection()
        reward = game.rewards(done)
        if app_state:
            replay_list.append(reward)
            score += reward
        if app_state and done:
            replay_list.append([])

        # drawing cheese
        game.update()
        game.draw(screen)

        # add state_t+1
        if len(game.buffer) < 5 and app_state:
            try:
                x_distance = game.buffer[-1].rect1.x - game.bird.rect.right
                y_top_distance = game.bird.rect.y - game.buffer[-1].rect2.bottom
                y_bottom_distance = game.buffer[-1].rect1.top - game.bird.rect.bottom
                state_p = [height, x_distance, y_bottom_distance, y_top_distance, velocity]
                state_p = np.array([state_p])
                replay_list.append(state_p)
            except IndexError:
                pass
        elif len(game.buffer) == 5 and app_state:
            x_distance = game.buffer[-2].rect1.x - game.bird.rect.right
            y_top_distance = game.bird.rect.y - game.buffer[-2].rect2.bottom
            y_bottom_distance = game.buffer[-2].rect1.top - game.bird.rect.bottom
            state_p = [height, x_distance, y_bottom_distance, y_top_distance, velocity]
            state_p = np.array([state_p])
            replay_list.append(state_p)

        # finish
        if app_state:
            memory.add(replay_list)

        # update weights
        if game.frame % 200 == 0 and game.frame > 0:
            model.train(memory, target_model)

        if game.frame % 3000 == 0 and game.frame > 0:
            target_model.model.set_weights(model.model.get_weights())

        # escape route
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    q = True

        # finish
        if done:
            print(score)
            print(model.epsilon)
            score = 0
        game.time()

if __name__ == "__main__":
    main()