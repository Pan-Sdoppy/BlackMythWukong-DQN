import time

from pynput import mouse

# 全局变量记录鼠标左键按下的时间
left_click_pressed_time = None


def on_click(_, __, button, pressed):
    global left_click_pressed_time
    # 如果是鼠标左键
    if button == mouse.Button.left:
        if pressed:
            # 记录按下的时间
            left_click_pressed_time = time.time()
        else:
            # 如果之前记录过按下时间，则计算按下持续的时间
            if left_click_pressed_time:
                duration = time.time() - left_click_pressed_time
                print(f"鼠标左键按压时长: {round(duration, 3)}")
            # 重置按下时间
            left_click_pressed_time = None


# 设置鼠标监听器监听点击事件
with mouse.Listener(on_click=on_click) as listener:
    listener.join()
