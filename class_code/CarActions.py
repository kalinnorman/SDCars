import time


class CarActions:
    """
    This class is used to have the car perform specific actions
    """
    def __init__(self, cc):
        self.cc = cc

    def drive_straight(self):
        self.cc.steer(0)
        # self.cc.drive(0.5)
        time.sleep(3.5)
        # self.cc.drive(0)

    def turn_left_from_stop_at_corner(self):
        self.cc.drive(0.6)
        time.sleep(1.6)
        self.cc.steer(-20)
        time.sleep(1.5)
        self.cc.drive(0) # FIXME must be deleted for actual implementation
        self.cc.steer(-3)

    def turn_left_from_stop_at_intersection(self):
        self.cc.drive(0.6)
        time.sleep(1.7)
        self.cc.steer(-20)
        time.sleep(1.5)
        self.cc.drive(0) # FIXME must be deleted for actual implementation
        self.cc.steer(-3)

    def turn_left_while_moving(self):
        #Simulate Driving up to the turn
        # self.cc.drive(0.6) # FIXME must be deleted for actual implementation
        # time.sleep(3) # FIXME must be deleted for actual implementation
        #Done Simulating Driving up to the turn
        time.sleep(0.4)
        self.cc.steer(-13)
        time.sleep(1.6)
        # self.cc.drive(0) # FIXME must be deleted for actual implementation
        self.cc.steer(0)

    def turn_right_from_stop_at_corner(self):
        self.cc.drive(0.6)
        time.sleep(1.5)
        self.cc.steer(30)
        time.sleep(1.2)
        self.cc.drive(0) # FIXME must be deleted for implementation
        self.cc.steer(-3)

    def turn_right_from_stop_at_intersection(self):
        self.cc.drive(0.6)
        time.sleep(1.8)
        self.cc.steer(30)
        time.sleep(1)
        self.cc.drive(0) # FIXME must be deleted for implementation
        self.cc.steer(-3)

    def turn_right_while_moving(self):
        #Simulate driving up to the turn
        #self.cc.drive(0.6) # FIXME must be deleted for actual implementation
        #time.sleep(2.3) # FIXME must be deleted for actual implementation
        #End of driving up
        time.sleep(1)
        self.cc.steer(33)
        time.sleep(2)
        # self.cc.drive(0) # FIXME must be deleted for implementation
        self.cc.steer(-3)

    def turn_right_out_of_parking_spot(self):
        self.cc.steer(30)
        self.cc.drive(0.6)
        time.sleep(2.2)
        self.cc.drive(0)
        self.cc.steer(-3)
