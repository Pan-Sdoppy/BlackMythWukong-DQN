import collections
import random
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Tuple

import cv2
import mss
import numpy as np
import pyautogui
import torch
from PIL import Image
from torchvision import transforms

import common.nn
from common import constants


class Action(Enum):
    """行为枚举类"""
    DODGE = 0
    LIGHT_ATTACK = 1
    HEAVY_ATTACK = 2
    JUMP_LIGHT_ATTACK = 3
    JUMP_HEAVY_ATTACK = 4


class ReplayBuffer:
    """经验缓冲池"""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state: torch.Tensor, action: int, reward: int, next_state: torch.Tensor, done: int):
        """
        往经验池缓冲池中添加历史经验
        :param state: 模型输入
        :param action: 模型输出
        :param reward: 奖励得分
        :param next_state: 下一个状态的模型输入
        :param done: 回合是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, int, int, torch.Tensor, int]:
        """
        在经验池缓冲池中随机采样
        :param batch_size: 一次采样批大小
        :return: 随机采样集合
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return torch.cat(state, dim=0), action, reward, torch.cat(next_state, dim=0), done

    def size(self) -> int:
        """
        经验池大小
        :return: 经验池大小
        """
        return len(self.buffer)


def save_model(model: common.nn.CNN, path: str):
    """
    保存模型
    :param model: 模型
    :param path: 保存路径
    """
    path_obj = Path(path)
    parent_dir = path_obj.parent
    # 如果父目录不存在，则创建它
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: common.nn.CNN, path: str):
    """
    导入模型权重
    :param model: 模型
    :param path: 权重文件路径
    """
    model.load_state_dict(torch.load(path))


def save_img(image: np.ndarray, img_path: str):
    """
    保存图片
    :param image: 图片数据
    :param img_path: 保存路径
    """
    suffix = Path(img_path).suffix
    cv2.imencode(suffix, image)[1].tofile(img_path)


def read_img(img_path: str) -> np.ndarray:
    """
    读灰度图
    :param img_path: 图片路径
    :return: 图片
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return img


def get_blood_img(img: np.ndarray, self_mode: bool = True) -> np.ndarray:
    """
    获取血量截图
    :param img: 图片数据
    :param self_mode: 是否寻找自己血量
    :return: 血量区域截图
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if self_mode:
        roi_img = gray_img[constants.SELF_BLOOD_Y1:constants.SELF_BLOOD_Y2,
                  constants.SELF_BLOOD_X1:constants.SELF_BLOOD_X2].copy()
    else:
        roi_img = gray_img[constants.BOSS_BLOOD_Y1:constants.BOSS_BLOOD_Y2,
                  constants.BOSS_BLOOD_X1:constants.BOSS_BLOOD_X2].copy()
    return roi_img


def get_game_screenshot() -> np.ndarray:
    """
    获取游戏截图
    :return: 图片数据
    """
    with mss.mss() as sct:
        # 获取屏幕截图
        screenshot = sct.grab(constants.MONITOR_ROI)
        # 将截图转换为OpenCV格式
        img = np.array(screenshot)
        # OpenCV默认使用BGR颜色空间，mss返回的是BGRA，
        # 因此需要去掉alpha通道n
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


def get_blood_value(roi_img: np.ndarray, denominator: int) -> int:
    """
    获取血量占比
    :param roi_img: 血量区域图片
    :param denominator: 血量总宽像素
    :return: 血量占比
    """
    ret, threshold_image = cv2.threshold(roi_img, 150, 255, cv2.THRESH_BINARY)
    # 寻找白色区域
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_x = 0
    for cnt in contours:
        # cv2.boundingRect获取轮廓坐标
        x, y, w, h = cv2.boundingRect(cnt)
        if x + w > max_x and x < denominator // 10:
            max_x = x + w
            if max_x > denominator:
                max_x = denominator
    return int(round((max_x / denominator), 3) * 1000)


def calculate_reward(self_blood: int, next_self_blood: int, boss_blood: int, next_boss_blood: int) -> Tuple[int, int]:
    """
    计算奖励分数
    :param self_blood: 行为前自己血量
    :param next_self_blood: 行为后自己血量
    :param boss_blood: 行为前敌人血量
    :param next_boss_blood: 行为后敌人血量
    :return: 奖励分数
    """
    reward = 0
    done = 0
    # 突然大量回血代表回合结束
    if next_self_blood - self_blood > 800:
        # 回血前玩家血量少于14%判负
        if self_blood <= 170:
            reward -= 400
        else:
            reward += 400
        done = 1
    # 防止有时候玩家回血事件没被捕捉到
    elif next_boss_blood - boss_blood > 800:
        if boss_blood <= 170:
            reward -= 400
        else:
            reward += 400
        done = 1
    else:
        if (self_blood - next_self_blood) > 0:
            reward -= (self_blood - next_self_blood)
        if (boss_blood - next_boss_blood) > 0:
            reward += (boss_blood - next_boss_blood)
    return reward, done


def light_attack(time_delay_flag: bool, n_light_attack: int):
    """
    模拟轻棍攻击
    :param time_delay_flag: 模型训练前后会有一定的操作延迟
    :param n_light_attack: 第几次轻棍连击
    """
    pyautogui.click(button='left')
    n_light_attack %= 5
    # 不同次数的轻棍攻击的后摇时间不同
    time.sleep(constants.LIGHT_ATTACK_TIME_SLEEP_DICT[n_light_attack][int(time_delay_flag)])


def heavy_attack(time_delay_flag: bool, resolute_strike: bool):
    """
    模拟重棍攻击
    :param time_delay_flag: 模型训练前后会有一定的操作延迟
    :param resolute_strike: 是否触发切手技
    """
    pyautogui.click(button='right')
    time.sleep(constants.HEAVY_ATTACK_TIME_SLEEP_DICT[int(resolute_strike)][int(time_delay_flag)])


def dodge(time_delay_flag: bool):
    """
    模拟闪避
    :param time_delay_flag: 模型训练前后会有一定的操作延迟
    """
    pyautogui.keyDown('space')
    pyautogui.keyUp('space')
    time.sleep(constants.DODGE_TIME_SLEEP_DICT[int(time_delay_flag)])


def jump_light_attack(time_delay_flag: bool):
    """
    模拟跳跃轻击
    :param time_delay_flag: 模型训练前后会有一定的操作延迟
    """
    pyautogui.keyDown('ctrl')
    # 跳至最高点再按攻击键
    time.sleep(constants.JUMP_TIME)
    pyautogui.click(button='left')
    pyautogui.keyUp('ctrl')
    time.sleep(constants.JUMP_LIGHT_ATTACK_SLEEP_DICT[int(time_delay_flag)])


def jump_heavy_attack(time_delay_flag: bool):
    """
    模拟跳跃重击
    :param time_delay_flag: 模型训练前后会有一定的操作延迟
    """
    pyautogui.keyDown('ctrl')
    # 跳至最高点再按攻击键
    time.sleep(constants.JUMP_TIME)
    pyautogui.click(button='right')
    pyautogui.keyUp('ctrl')
    time.sleep(constants.JUMP_HEAVY_ATTACK_SLEEP_DICT[int(time_delay_flag)])


def take_action(action: int, n_light_attack: int, time_delay_flag: bool, resolute_strike: bool):
    """
    模拟相应的行为
    :param action: 行为枚举值
    :param n_light_attack: 第几次轻棍连击
    :param time_delay_flag: 模型训练前后会有一定的操作延迟
    :param resolute_strike: 是否触发切手技
    """
    if action == Action.LIGHT_ATTACK.value:
        light_attack(time_delay_flag, n_light_attack)
    elif action == Action.HEAVY_ATTACK.value:
        heavy_attack(time_delay_flag, resolute_strike)
    elif action == Action.JUMP_LIGHT_ATTACK.value:
        jump_light_attack(time_delay_flag)
    elif action == Action.JUMP_HEAVY_ATTACK.value:
        jump_heavy_attack(time_delay_flag)
    elif action == Action.DODGE.value:
        dodge(time_delay_flag)
    else:
        raise ValueError("检查模型输出尺寸")


def img_to_state(img: np.ndarray) -> torch.Tensor:
    """
    把图像转成模型可接受的状态输入类型
    :param img: 截屏图像
    :return: 转化后的tensor输入
    """
    image_pil = Image.fromarray(img.astype('uint8'), 'RGB')
    transform = transforms.Compose([
        transforms.Resize((308, 308)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 应用转换
    image = transform(image_pil)
    state = image.unsqueeze(0)
    return state


def get_state_and_blood() -> Tuple[torch.Tensor, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    获取模型输入和血量信息
    :return: 模型输入和血量信息
    """
    img = get_game_screenshot()
    img_copy = img[constants.STATE_Y1:constants.STATE_Y2, constants.STATE_X1:constants.STATE_X2].copy()
    state = img_to_state(img_copy)
    self_blood_img = get_blood_img(img, self_mode=True)
    self_blood = get_blood_value(self_blood_img, constants.SELF_BLOOD_LEN)
    boss_blood_img = get_blood_img(img, self_mode=False)
    boss_blood = get_blood_value(boss_blood_img, constants.BOSS_BLOOD_LEN)
    return state, self_blood, boss_blood, img, self_blood_img, boss_blood_img


def train_dqn(agent: common.nn.DQN, replay_buffer: ReplayBuffer):
    """
    训练 DQN
    :param agent: DQN 模型
    :param replay_buffer: 经验缓冲池
    """
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


def get_action_condition(action: int, n_light_attack: int) -> Tuple[int, bool]:
    """
    获取轻棍连击次数和是否使出切手技
    :param action: 行为枚举值
    :param n_light_attack: 轻棍连击次数
    :return: 轻棍连击次数和是否使出切手技
    """
    resolute_strike = False
    if action == Action.LIGHT_ATTACK.value:
        # 轻棍连击数加一
        n_light_attack += 1
    else:
        if n_light_attack > 0:
            # 轻棍第五段无法打出切手技
            if n_light_attack % 5 != 0:
                resolute_strike = True
    return n_light_attack, resolute_strike


def get_time() -> str:
    """
    获取当前时间
    :return: 格式化时间字符串
    """
    now = datetime.now()
    formatted_time = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
    return formatted_time


def save_debug_img(img: np.ndarray, self_blood_img: np.ndarray, boss_blood_img: np.ndarray):
    path_obj = Path(r"./debug")
    parent_dir = path_obj
    # 如果父目录不存在，则创建它
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
    now = get_time()
    img_path = f"./debug/{now}_img.jpg"
    self_blood_img_path = f"./debug/{now}_self_blood_img.jpg"
    boss_blood_img_path = f"./debug/{now}_boss_blood_img.jpg"
    save_img(img, img_path)
    save_img(self_blood_img, self_blood_img_path)
    save_img(boss_blood_img, boss_blood_img_path)
