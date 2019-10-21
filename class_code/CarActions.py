import time


class CarActions:
    """
    This class is used to have the car perform specific actions
    """
    def __init__(self, cc):
        self.cc = cc

    def drive_straight(self):
        self.cc.steer(0)
        time.sleep(5)

    def turn_left_while_moving(self):
        time.sleep(2)
        self.cc.steer(-17)
        time.sleep(2.5)
        # self.cc.steer(0)

    def turn_right_while_moving(self):
        time.sleep(1)
        self.cc.steer(30)
        time.sleep(2)
        # self.cc.steer(0)

    def turn_right_out_of_parking_spot(self):
        self.cc.steer(30)
        self.cc.drive(0.6)
        time.sleep(2.2)
        self.cc.drive(0)
        self.cc.steer(0)
