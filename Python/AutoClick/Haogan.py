import pyautogui
import pydirectinput
import time
pyautogui.PAUSE = 0.5
pyautogui.FAILSAFE = True

time.sleep(5)

for loop in range(13):
    # Initialize Position
    pydirectinput.press('m')
    time.sleep(1)
    pyautogui.click(1731, 701)
    pyautogui.click(2502, 1077)
    pyautogui.click(3048, 1351)
    time.sleep(3)
    for i in range(5):
        pydirectinput.press('w')
        time.sleep(1)

    # Go to Position of the goal
    pydirectinput.keyDown('shiftleft')
    pydirectinput.keyDown('w')
    time.sleep(3.2)
    pydirectinput.write(' ')
    pydirectinput.keyUp('w')
    pydirectinput.keyUp('shiftleft')

    pydirectinput.keyDown('w')
    time.sleep(1.4)
    pydirectinput.keyUp('w')

    # Do the task
    time.sleep(2)
    pydirectinput.press('f')
    pyautogui.click(2012, 1013)
    time.sleep(2)

    # Back To initial Position
    pydirectinput.press('m')
    time.sleep(1)
    pyautogui.click(1731, 701)
    pyautogui.click(2502, 1077)
    pyautogui.click(3048, 1351)
    time.sleep(10)

    # Re-login
    pydirectinput.press('esc')
    time.sleep(12)
    pyautogui.click(157, 1367)
    pyautogui.click(2012, 1013)

    time.sleep(20)
    pyautogui.click(1717, 1356)

    time.sleep(25)

#print(pyautogui.position())