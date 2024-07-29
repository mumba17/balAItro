import virtual_controller as vc

class Act:
    def __init__(self, vc):
        self.vc = vc
        self.directions = {
            'up': vc.DPAD_UP,
            'down': vc.DPAD_DOWN,
            'left': vc.DPAD_LEFT,
            'right': vc.DPAD_RIGHT
        }

    def reroll_shop(self):
        # Presses the X button to reroll shop
        vc.press_button(self.vc.X)
        
    def next_blind(self):
        # Presses the Y button to move to next blind
        vc.press_button(self.vc.Y)
        
    def sell_joker(self):
        # Presses the LB button to sell the selected joker
        vc.press_button(self.vc.LB)
        
    def move_pad(self, direction):
        # Moves the pad in the specified direction
        vc.press_button(vc.directions[direction])
        
    
        
    
    