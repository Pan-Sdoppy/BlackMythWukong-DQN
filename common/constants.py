# --------------------------- 截图参数 ---------------------------
# 屏幕截图区域
MONITOR_ROI = {"top": 0, "left": 2126, "width": 1204, "height": 999}
# 血量二值化阈值
THRESHOLD = 50
# 玩家血量框
SELF_BLOOD_X = 2
SELF_BLOOD_Y = 980
SELF_BLOOD_W = 336
SELF_BLOOD_H = 18
# 敌人血量框
BOSS_BLOOD_X = 477
BOSS_BLOOD_Y = 912
BOSS_BLOOD_W = 578
BOSS_BLOOD_H = 8
# --------------------------- 训练参数 ---------------------------
# 经验缓冲池容量
BUFFER_CAPACITY = 100
# 学习率
LR = 2e-3
# 折扣因子
GAMMA = 0.9
# 贪心系数
EPSILON = 0.9
# 目标网络更新频率
TARGET_UPDATE_FREQUENCY = 30
# 批大小
BATCH_SIZE = 16
# 经验池最小训练大小
BUFFER_MIN_SIZE = 30
# 行动数
N_HIDDEN = 64 * 35 * 35
N_ACTIONS = 5
LIGHT_ATTACK_TIME_SLEEP_DICT = {
    1: [0.40, 0.00],
    2: [0.80, 0.25],
    3: [0.80, 0.25],
    4: [1.30, 0.75],
    0: [1.30, 0.75],
}
HEAVY_ATTACK_TIME_SLEEP_DICT = {
    0: [4.40, 3.85],
    1: [1.50, 0.95],
}
DODGE_TIME_SLEEP_DICT = {
    0: 0.70,
    1: 0.15,
}
JUMP_LIGHT_ATTACK_SLEEP_DICT = {
    0: 1.50,
    1: 0.95,
}
JUMP_HEAVY_ATTACK_SLEEP_DICT = {
    0: 3.00,
    1: 2.45,
}  # 角色跳至最高点时间
JUMP_TIME = 0.3
