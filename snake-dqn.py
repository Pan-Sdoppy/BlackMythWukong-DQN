import math
import random
import time

import pygame
import torch

from common import constants
from common.nn import DQN
from common.utils import ReplayBuffer, get_game_screenshot, img_to_state, save_model, load_model

# 初始化pygame
pygame.init()

# 定义颜色
black = (0, 0, 0)
red = (213, 50, 80)
green = [0, 255, 0]
blue = (0, 0, 255)

# 定义显示设置
dis_width = 500
dis_height = 500
dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('贪吃蛇-DQN')

clock = pygame.time.Clock()

snake_block = 50
snake_speed = 2


def draw_snake(snake_block, snake_list):
    global green
    for i, x in enumerate(reversed(snake_list)):
        if i == 0:
            pygame.draw.rect(dis, red, [x[0], x[1], snake_block, snake_block])
        else:
            pygame.draw.rect(dis, green, [x[0], x[1], snake_block, snake_block])
            if i % 5 == 0:
                green[1] = green[1] - 5
    green = [0, 255, 0]


def calculate_distance(x1, x2, y1, y2):
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 实例化经验池
replay_buffer = ReplayBuffer(constants.BUFFER_CAPACITY)
# 实例化DQN
agent = DQN(n_hidden=constants.N_HIDDEN,
            n_actions=4,
            learning_rate=constants.LR,
            gamma=constants.GAMMA,
            epsilon=constants.EPSILON,
            target_update_frequency=constants.TARGET_UPDATE_FREQUENCY,
            device=device,
            )
load_model(agent.q_net, r"./model/snake_dqn_0_epoch.pth")
load_model(agent.target_q_net, r"./model/snake_dqn_0_epoch.pth")


def game_loop():
    j = 0
    ccc = 0
    while True:
        game_on = True
        ccc += 1
        count = 0
        x = dis_width / 2
        y = dis_height / 2


        x_change = 0
        y_change = 0

        snake_list = []
        snake_head = [x, y]
        length_of_snake = 1
        snake_list.append(snake_head)

        food_x = round(random.randrange(0, dis_width - snake_block) / snake_block) * snake_block
        food_y = round(random.randrange(0, dis_height - snake_block) / snake_block) * snake_block
        while [food_x, food_y] in snake_list:
            food_x = round(random.randrange(0, dis_width - snake_block) / snake_block) * snake_block
            food_y = round(random.randrange(0, dis_height - snake_block) / snake_block) * snake_block

        dis.fill(black)
        pygame.draw.rect(dis, blue, [food_x, food_y, snake_block, snake_block])
        draw_snake(snake_block, snake_list)
        pygame.display.update()

        img = get_game_screenshot()
        state = img_to_state(img).to(device)
        total_reward = 0

        while game_on:
            eat = False
            # 获取当前状态下需要采取的动作
            # 50步后仍吃不到食物
            reward = 0
            if count == -50:
                game_on = False
                reward -= 200
            action = agent.choose_action(state, train_mode=True)
            if action == 0:
                x_change = -snake_block
                y_change = 0
            elif action == 1:
                x_change = snake_block
                y_change = 0
            elif action == 2:
                y_change = -snake_block
                x_change = 0
            elif action == 3:
                y_change = snake_block
                x_change = 0

            # for event in pygame.event.get():
            #     if event.type == pygame.KEYDOWN:
            #         if event.key == pygame.K_LEFT:
            #             x_change = -snake_block
            #             y_change = 0
            #         elif event.key == pygame.K_RIGHT:
            #             x_change = snake_block
            #             y_change = 0
            #         elif event.key == pygame.K_UP:
            #             y_change = -snake_block
            #             x_change = 0
            #         elif event.key == pygame.K_DOWN:
            #             y_change = snake_block
            #             x_change = 0

            if x >= dis_width or x < 0 or y >= dis_height or y < 0:
                game_on = False
                reward -= 50

            last_dis = calculate_distance(x, food_x, y, food_y)
            x += x_change
            y += y_change
            now_dis = calculate_distance(x, food_x, y, food_y)
            snake_list.append([x, y])
            if last_dis > now_dis:
                reward += 20
            else:
                reward -= 50

            if x == food_x and y == food_y:
                food_x = round(random.randrange(0, dis_width - snake_block) / snake_block) * snake_block
                food_y = round(random.randrange(0, dis_height - snake_block) / snake_block) * snake_block
                while [food_x, food_y] in snake_list:
                    food_x = round(random.randrange(0, dis_width - snake_block) / snake_block) * snake_block
                    food_y = round(random.randrange(0, dis_height - snake_block) / snake_block) * snake_block
                length_of_snake += 1
                reward += (500*length_of_snake)
                count = 0
                pygame.draw.rect(dis, blue, [food_x, food_y, snake_block, snake_block])
                eat = True
            else:
                count -= 1

            if len(snake_list) > length_of_snake:
                pygame.draw.rect(dis, black, [snake_list[0][0], snake_list[0][1], snake_block, snake_block])
                del snake_list[0]

            snake_head = snake_list[-1]

            for temp in snake_list[:-1]:
                if temp == snake_head:
                    game_on = False
                    reward -= 50

            draw_snake(snake_block, snake_list)
            pygame.display.update()

            next_img = get_game_screenshot()
            next_state = img_to_state(next_img).to(device)
            # 此次行为添加到经验池
            replay_buffer.add(state, action, reward, next_state, not game_on)
            state = next_state
            if replay_buffer.size() > constants.BUFFER_MIN_SIZE:
                # 当经验池超过一定数量后，开始训练网络
                states, actions, rewards, next_states, dones = replay_buffer.sample(constants.BATCH_SIZE)
                # 构造训练集
                transition_dict = {
                    'states': states,
                    'actions': actions,
                    'rewards': rewards,
                    'next_states': next_states,
                    'dones': dones,
                }
                # DQN训练
                agent.train(transition_dict)
            clock.tick(snake_speed)
            total_reward += reward
        if ccc > 4000:
            save_model(agent.q_net, f'./model/snake_dqn_{str(j)}_epoch.pth')
            j += 1
            ccc = 0
            print(total_reward)


game_loop()
