from car_control import carControl
import time

class carActions():
    """
    This class is used to have the car perform specific actions
    """
    def init(self):
        self.c = carControl()

    def driveStraight(self):
        self.steer(-3)
        self.drive(0.6)
        time.sleep(3.5)
        self.drive(0)

    def turnLeftFromStopAtCorner(self):
        self.drive(0.6)
        time.sleep(1.6)
        self.steer(-20)
        time.sleep(1.5)
        self.drive(0) # FIXME must be deleted for actual implementation
        self.steer(-3)

    def turnLeftFromStopAtIntersection(self):
        self.drive(0.6)
        time.sleep(1.7)
        self.steer(-20)
        time.sleep(1.5)
        self.drive(0) # FIXME must be deleted for actual implementation
        self.steer(-3)

    def turnLeftWhileMoving(self):
        #Simulate Driving up to the turn
        self.drive(0.6) # FIXME must be deleted for actual implementation
        time.sleep(3) # FIXME must be deleted for actual implementation
        #Done Simulating Driving up to the turn
        time.sleep(0.4)
        self.steer(-16)
        time.sleep(1.6)
        self.drive(0) # FIXME must be deleted for actual implementation
        self.steer(-3)

    def turnRightFromStopAtCorner(self):
        self.drive(0.6)
        time.sleep(1.5)
        self.steer(30)
        time.sleep(1.2)
        self.drive(0) # FIXME must be deleted for implementation
        self.steer(-3)

    def turnRightFromStopAtIntersection(self):
        self.drive(0.6)
        time.sleep(1.8)
        self.steer(30)
        time.sleep(1)
        self.drive(0) # FIXME must be deleted for implementation
        self.steer(-3)

    def turnRightWhileMoving(self):
        #Simulate driving up to the turn
        self.drive(0.6) # FIXME must be deleted for actual implementation
        time.sleep(2.3) # FIXME must be deleted for actual implementation
        #End of driving up
        time.sleep(0.8)
        self.steer(30)
        time.sleep(1)
        self.drive(0) # FIXME must be deleted for implementation
        self.steer(-3)

    def turnRightOutOfParkingSpot(self):
        self.steer(30)
        self.drive(0.6)
        time.sleep(2.2)
        self.drive(0)
        self.steer(-3)
