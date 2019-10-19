"""
CarControl.py
Author: redd
"""

import serial
import time

class CarTune:
    """
    This class will be used to control the car.
    """

    def __init__(self):
        self.ser = self._initialize_serial_communication()  # establish serial communication
        self._initialize_car()  # initialize the car

    def drive(self, speed):
        """
        Commands the car to drive.
        :param speed: -2.0 to 2.0
        :return: nothing
        """
        # drive_command = "!speed" + str(speed) + "\n"
        # self.ser.write(drive_command.encode())
        self._send_command("!speed" + str(speed) + "\n")

    def steer(self, degree):
        """
        Commands the car to turn.
        :param degree: -30.0 to 30.0
        :return: nothing
        """
        # degree = degree - 3
        self._send_command("!steering" + str(degree) + "\n")

    def _initialize_serial_communication(self):
        """
        Initializes the serial communication.

        :return: Object required for communication.
        """
        print("Initializing Serial Communications")
        ser = serial.Serial("/dev/ttyUSB0", 115200)
        time.sleep(2)  # must sleep for a bit while initializing
        print("Flushing Input")
        ser.flushInput()
        time.sleep(1)  # must sleep for a bit while initializing
        return ser

    def _send_command(self, command, addnewline=False):
        """
        Sends a command to the car. Remember that every command must end with a new line.

        Author: redd
        """
        if addnewline:
            command = command + "\n"
        self.ser.write(command.encode())

    def _initialize_car(self, pid_flag=True):
        """
        Initializes the car. This must be run before we can control the car.

        Author: redd
        """

        print("Initializing Car")
        print("Sending Start Command")
        self._send_command("!start1590\n")
        # time.sleep(1)

        self._send_command("!kp0.01\n")
        self._send_command("!kd0.01\n")
        self._send_command("!straight1450\n")
        if pid_flag:
            self._send_command("!pid1\n")
        else:
            self._send_command("!pid0\n")

        print("Stopping Car")
        self.drive(0.0)

if __name__ == "__main__":
    cc = CarTune()
    cc.steer(20)
    cc.steer(0)
    cc.drive(0.5)
    time.sleep(0.3)
    cc.drive(0.3)
    time.sleep(5)
    cc.drive(0)
    print('done')

