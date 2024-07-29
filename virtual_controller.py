import vgamepad as vg
import time

class VirtualController:
    def __init__(self):
        self.controller = vg.VX360Gamepad()

        #Map buttons
        self.A = vg.XUSB_BUTTON.XUSB_GAMEPAD_A
        self.B = vg.XUSB_BUTTON.XUSB_GAMEPAD_B
        self.X = vg.XUSB_BUTTON.XUSB_GAMEPAD_X
        self.Y = vg.XUSB_BUTTON.XUSB_GAMEPAD_Y
        self.LB = vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER
        self.RB = vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
        self.LT = vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB
        self.RT = vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB
        self.START = vg.XUSB_BUTTON.XUSB_GAMEPAD_START
        self.GUIDE = vg.XUSB_BUTTON.XUSB_GAMEPAD_GUIDE
        self.BACK = vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK
        self.DPAD_UP = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP
        self.DPAD_DOWN = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN
        self.DPAD_LEFT = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT
        self.DPAD_RIGHT = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT
        self.RIGHT_SHOULDER = vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
        self.LEFT_SHOULDER = vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER

    def press_button(self, button):
        self.controller.press_button(button)
        self.controller.update()
        time.sleep(0.1)
        self.controller.release_button(button)
        self.controller.update()

if __name__ == "__main__":
    xbox = VirtualController()
    while True:
        xbox.press_button(xbox.DPAD_RIGHT)