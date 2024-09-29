import torch

from common import constants
from common.nn import DQN
from common.utils import ReplayBuffer, load_model, get_game_screenshot, img_to_state, take_action

if __name__ == '__main__':
    # GPU运算
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
    # 测试模型
    while True:
        # 获取当前状态下需要采取的动作
        resolute_strike = False
        img = get_game_screenshot()
        state = img_to_state(img)
        action = agent.choose_action(state, train_mode=False)
        if action == 1:
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
