# car-control.py

import serial
import time


class carControl():

    def __init__(self):
        self.ser = self._initialize_serial_communication()  # establish serial communication
        self._initialize_car()  # initialize the car

    def _initialize_serial_communication(self):
        print("Initializing Serial Communications")
        ser = serial.Serial("/dev/ttyUSB0", 115200)
        ser.flushInput()
        time.sleep(1)
        return ser

    def drive(self, speed):
        command = "!speed" + str(speed) + "\n"
        self.ser.write(command.encode())

    def steer(self, degree):
        steer_command = "!steering" + str(degree) + "\n"
        self.ser.write(steer_command.encode())

    # def send_command(ser, command, addnewline=False):
    #     """
    #     Sends a command to the car. Remember that every command must end with a new line.
    #
    #     Author: redd
    #
    #     :param ser: the serial port to send the string to
    #     :param command: the command to send
    #     :return: no return
    #     """
    #     if addnewline:
    #         command = command + "\n"
    #     ser.write(command.encode())

    def _initialize_car(self, pid_flag=True):
        """
        Initializes the car. This must be run before we can control the car.

        Author: norman
        """

        print("Initializing Car")
        start = "!start1590\n"
        self.ser.write(start.encode())

        # inits = "!inits0.5\n"
        # kp = "!kp0.01\n"
        # kd = "!kd0.01\n"
        # straight = "!straight1500\n"
        # if pid_flag:
        #     pid = "!pid1\n"
        # else:
        #     pid = "!pid0\n"
        #
        # self.ser.write(inits.encode())
        # self.ser.write(kp.encode())
        # self.ser.write(kd.encode())
        # self.ser.write(straight.encode())
        # self.ser.write(pid.encode())

        #self.drive(0.0)
        #self.steer(0.0)

