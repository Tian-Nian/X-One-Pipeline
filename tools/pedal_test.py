from robot.utils.base.footpedal import FootPedal
import time

footpedal = FootPedal("/dev/pedal")

while not footpedal.was_pressed():
    print("等待踏板按下...")
    time.sleep(0.1)
print("踏板已按下！")