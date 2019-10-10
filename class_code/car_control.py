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
        drive_command = "!speed" + str(speed) + "\n"
        self.ser.write(drive_command.encode())

    def steer(self, degree):
        steer_command = "!steering" + str(degree) + "\n"
        self.ser.write(steer_command.encode())

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
        start = "!start1590\n"
        self.ser.write(start.encode())

        # inits = "!inits0.5\n"
        # kp = "!kp0.01\n"
        # kd = "!kd0.01\n"
        straight = "!straight1500\n"
        # if pid_flag:
        #     pid = "!pid1\n"
        # else:
        #     pid = "!pid0\n"
        #
        # self.ser.write(inits.encode())
        # self.ser.write(kp.encode())
        # self.ser.write(kd.encode())
        self.ser.write(straight.encode())
        # self.ser.write(pid.encode())

        '''
        self._send_command("!start1590\n")
        self._send_command("!kp0.01\n")
        self._send_command("!kd0.01\n")
        self._send_command("!straight1500\n")
        if pid_flag:
            self._send_command("!pid1\n")
        else:
            self._send_command("!pid0\n")
        '''

        self.ser.write("!speed0.2\n".encode())
        time.sleep(2)
        self.ser.write("!speed0\n".encode())

        #self.drive(0.0)
        #self.steer(0.0)

