import keyboard
import matplotlib.pyplot as plt
import torch
from loguru import logger
from tqdm import tqdm

from common import constants
from common.nn import DQN
from common.utils import ReplayBuffer, load_model, take_action, calculate_reward, save_model, get_state_and_blood, \
    train_dqn, get_action_condition

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 记录第几次轻棍连击
    n_light_attack = 0
    time_delay_flag = False
    return_list = []
    # 实例化经验池
    replay_buffer = ReplayBuffer(constants.BUFFER_CAPACITY)
    interrupt_flag = False
    # 实例化DQN
    agent = DQN(n_hidden=constants.N_HIDDEN,
                n_actions=constants.N_ACTIONS,
                learning_rate=constants.LR,
                gamma=constants.GAMMA,
                epsilon=constants.EPSILON,
                target_update_frequency=constants.TARGET_UPDATE_FREQUENCY,
                device=device,
                )
    # 导入权重
    load_model(agent.q_net, r"./model/wukong_dqn_9_epoch.pth")
    # 等待开始按键被按下
    logger.info("按下'n'键开始")
    while True:
        if keyboard.is_pressed('n'):
            break
    logger.info("开始执行")
    try:
        for i in range(constants.EPOCH):
            if interrupt_flag:
                break
            # 截屏并获取玩家和敌人血量
            state, self_blood, boss_blood = get_state_and_blood()
            # 记录每个回合的回报
            epoch_return = 0
            # 记录当前回合是否完成
            done = 0
            # 打印训练进度，一共10回合
            with tqdm(total=1, desc='Iteration %d' % i) as pbar:
                while not done:
                    # 获取当前状态下需要采取的动作
                    action = agent.choose_action(state, train_mode=True)
                    n_light_attack, resolute_strike = get_action_condition(action, n_light_attack)
                    # 模拟行为输入
                    take_action(action, n_light_attack, time_delay_flag, resolute_strike)
                    # 获取下一个状态
                    next_state, next_self_blood, next_boss_blood = get_state_and_blood()
                    # 计算此次行为的奖励，以及当前回合是否结束
                    reward, done = calculate_reward(self_blood, next_self_blood, boss_blood, next_boss_blood)
                    # 此次行为添加到经验池
                    replay_buffer.add(state, action, reward, next_state, done)
                    # 更新当前状态
                    state = next_state
                    self_blood = next_self_blood
                    boss_blood = next_boss_blood
                    # 更新回合回报
                    epoch_return += reward
                    # 当经验池超过一定数量后，开始训练网络
                    train_dqn(agent, replay_buffer)
                    # 回合结束，进行下一轮训练
                    if done:
                        pbar.update(1)
                    if keyboard.is_pressed('m'):
                        interrupt_flag = True
                        break  # 退出循环
                # 记录每个回合的回报
                return_list.append(epoch_return)
                # 更新进度条信息
                pbar.set_postfix({'得分': return_list[-1]})
            save_model(agent.q_net, f'./model/wukong_dqn_{str(i)}_epoch.pth')
        # 绘图
        plt.plot(range(len(return_list)), return_list)
        plt.xlabel('回合数')
        plt.ylabel('得分')
        plt.title('DQN回合得分')
        plt.show()
    except KeyboardInterrupt:
        logger.warning("用户中断，退出程序")
