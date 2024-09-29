import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from common import constants
from common.nn import DQN
from common.utils import ReplayBuffer, load_model, get_game_screenshot, img_to_state, get_blood_img, get_blood_value, \
    take_action, calculate_reward, save_model, Action

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 记录第几次轻棍连击
    n_light_attack = 0
    return_list = []
    # 实例化经验池
    replay_buffer = ReplayBuffer(constants.BUFFER_CAPACITY)
    # 实例化DQN
    agent = DQN(n_hidden=constants.N_HIDDEN,
                n_actions=constants.N_ACTIONS,
                learning_rate=constants.LR,
                gamma=constants.GAMMA,
                epsilon=constants.EPSILON,
                target_update=constants.TARGET_UPDATE_FREQUENCY,
                device=device,
                )
    # 导入权重
    load_model(agent.q_net, r"./model/wukong_dqn_9_epoch.pth")
    # 训练30回合
    for i in range(30):
        # 截屏并获取玩家和敌人血量
        img = get_game_screenshot()
        state = img_to_state(img)
        self_blood_img = get_blood_img(img, self_mode=True)
        self_blood = get_blood_value(self_blood_img, constants.SELF_BLOOD_LEN)
        boss_blood_img = get_blood_img(img, self_mode=False)
        boss_blood = get_blood_value(boss_blood_img, constants.BOSS_BLOOD_LEN)
        # 记录每个回合的回报
        epoch_return = 0
        # 记录当前回合是否完成
        done = 0
        # 打印训练进度，一共10回合
        with tqdm(total=1, desc='Iteration %d' % i) as pbar:
            while True:
                # 获取当前状态下需要采取的动作
                resolute_strike = False
                action = agent.choose_action(state)
                if action == Action.LIGHT_ATTACK:
                    # 轻棍连击数加一
                    n_light_attack += 1
                else:
                    if n_light_attack > 0:
                        # 轻棍第五段无法打出切手技
                        if n_light_attack % 5 != 0:
                            resolute_strike = True
                    n_light_attack = 0
                # 模拟行动输入
                take_action(action, n_light_attack, time_delay_flag, resolute_strike)
                # 获取下一个状态
                next_img = get_game_screenshot()
                next_state = img_to_state(next_img)
                next_self_blood_img = get_blood_img(next_img, self_mode=True)
                next_self_blood = get_blood_value(next_self_blood_img, constants.SELF_BLOOD_LEN)
                next_boss_blood_img = get_blood_img(next_img, self_mode=False)
                next_boss_blood = get_blood_value(next_boss_blood_img, constants.BOSS_BLOOD_LEN)
                # 计算此次行动的奖励，以及当前回合是否结束
                reward, done = calculate_reward(self_blood, next_self_blood, boss_blood, next_boss_blood)
                # 此次行动添加到经验池
                replay_buffer.add(state, action, reward, next_state, done)
                # 更新当前状态
                state = next_state
                # 更新回合回报
                epoch_return += reward
                # 当经验池超过一定数量后，开始训练网络
                if replay_buffer.size() > constants.BUFFER_MIN_SIZE:
                    # 开始出现操作延迟
                    time_delay_flag = True
                    # 从经验池中随机抽样作为训练集
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
                # 回合结束，进行下一轮训练
                if done:
                    pbar.update(1)
                    break
                self_blood = next_self_blood
                boss_blood = next_boss_blood
            # 记录每个回合的回报
            return_list.append(epoch_return)
            # 更新进度条信息
            pbar.set_postfix({'得分': return_list[-1]})
        save_model(agent.q_net, f'./model/wukong_dqn_{str(i)}_epoch.pth')
    # 绘图
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('回合数')
    plt.ylabel('得分')
    plt.title('DQN回合得分')
    plt.show()
