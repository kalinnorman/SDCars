import serial
import time
from sensors import Sensors

class CarControl:
    def __init__(self):
        self.ser = self._initialize_serial_communication()  # establish serial communication
        self._initialize_car()  # initialize the car
        self.sensor = Sensors()  # initialize sensors
        # first few frames of camera feed are low quality
        for i in range(0, 100):
            self.update_sensors()

    def update_sensors(self):
        self.sensor.update_sensors()
    
    def drive(self, speed):
        self._send_command("!speed" + str(speed) + "\n")

    def steer(self, degree):
        self._send_command("!steering" + str(degree) + "\n")

    def _initialize_serial_communication(self):
        print("Initializing Serial Communications")
        ser = serial.Serial("/dev/ttyUSB0", 115200)
        time.sleep(2)  # must sleep for a bit while initializing
        ser.flushInput()
        time.sleep(1)  # must sleep for a bit while initializing
        print("Serial Communications Initialized")
        return ser

    def _send_command(self, command, addnewline=False):
        if addnewline:
            command = command + "\n"
        self.ser.write(command.encode())

    def _initialize_car(self, pid_flag=True):
        print("Initializing Car")
        self._send_command("!straight1430\n")
        self._send_command("!kp0.01\n")
        self._send_command("!kd0.01\n")
        if pid_flag:
            self._send_command("!pid1\n")
        else:
            self._send_command("!pid0\n")
        self._send_command("!start1590\n")
        self.drive(0.0)
        print("Initialization Completed")