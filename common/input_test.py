import time

from pynput import mouse

# 全局变量记录鼠标左键按下的时间
left_click_pressed_time = time.time()


def on_click(_, __, button, pressed):
    global left_click_pressed_time
    # 如果是鼠标左键
    if button in [mouse.Button.left, mouse.Button.right]:
        if pressed:
            temp_time = time.time()
            duration = temp_time - left_click_pressed_time
            print(f"鼠标左键单击间隙时长: {round(duration, 3)}")
            left_click_pressed_time = temp_time


if __name__ == '__main__':
    # 设置鼠标监听器监听点击事件
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
