import time
import threading

import cv2
import mss
import numpy as np
import pyautogui
from pynput import keyboard
from ultralytics import YOLO


model = YOLO("runs/detect/cookie_detector-4/weights/best.onnx") 
CONFIDENCE = 0.7

sct = mss.mss()
monitor = sct.monitors[1]

# Установлено — программа работает; снято — пауза.
active = threading.Event()
active.set()

space_is_down = False


def on_press(key):
    global space_is_down

    if key == keyboard.Key.space and not space_is_down:
        space_is_down = True

        if active.is_set():
            active.clear()
            print("Пауза. Нажмите пробел, чтобы продолжить.")
        else:
            active.set()
            print("Продолжаю работу.")


def on_release(key):
    global space_is_down

    if key == keyboard.Key.space:
        space_is_down = False


def find_and_click(class_name):
    # Не делать снимок, пока программа на паузе.
    active.wait()

    screenshot = np.array(sct.grab(monitor))[:, :, :3]
    results = model(screenshot, verbose=False)

    for result in results:
        for box in result.boxes:
            # Если паузу включили во время распознавания — не кликать.
            if not active.is_set():
                return False

            detected_class = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            if detected_class == class_name and confidence >= CONFIDENCE:
                x1, y1, x2, y2 = box.xyxy[0]
                x = int((x1 + x2) / 2) + monitor["left"]
                y = int((y1 + y2) / 2) + monitor["top"]

                pyautogui.click(x, y)
                print(f"Клик по {class_name}: ({x}, {y})")
                return True

    return False


def click_with_timeout(class_name, timeout=5):
    deadline = time.time() + timeout

    while time.time() < deadline:
        active.wait()  # Если пауза включена, ждать повторного нажатия пробела.

        if find_and_click(class_name):
            time.sleep(0.5)
            return True

        time.sleep(0.2)

    print(f"{class_name} не найден за {timeout} секунд — перехожу дальше.")
    return False


if __name__ == "__main__":
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    listener.start()

    print("Программа запущена. Пробел — пауза/продолжение.")

    try:
        while True:
            for _ in range(4):
                if not click_with_timeout("game", timeout=5):
                    break

            click_with_timeout("taper", timeout=5)
            click_with_timeout("reklama", timeout=5)

    except KeyboardInterrupt:
        print("\nПрограмма завершена.")
    finally:
        listener.stop()
