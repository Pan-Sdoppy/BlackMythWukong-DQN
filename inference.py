import keyboard
import torch
from loguru import logger

from common import constants
from common.nn import DQN
from common.utils import ReplayBuffer, load_model, get_game_screenshot, img_to_state, get_action_condition, take_action

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    n_light_attack = 0
    time_delay_flag = False
    # 实例化经验池
    replay_buffer = ReplayBuffer(constants.BUFFER_CAPACITY)
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
        # 测试模型
        while True:
            # 获取当前状态下需要采取的动作
            img = get_game_screenshot()
            state = img_to_state(img)
            action = agent.choose_action(state, train_mode=False)
            n_light_attack, resolute_strike = get_action_condition(action, n_light_attack)
            # 模拟行为输入
            take_action(action, n_light_attack, time_delay_flag, resolute_strike)
            # 检查结束按键是否被按下
            if keyboard.is_pressed('m'):
                break
    except KeyboardInterrupt:
        logger.warning("用户中断，退出程序")
