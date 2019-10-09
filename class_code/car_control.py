# car-control.py

import serial
import time

ser = serial.Serial("/dev/ttyUSB0", 115200)
ser.flushInput()
time.sleep(1)


def drive(speed):
    forward_command = "!drive" + str(speed) + "\n"
    ser.write(forward_command.encode())


def steer(degree):
    steer_command = "!turn" + str(degree) + "\n"
    ser.write(steer_command.encode())


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


def initialize_car(pid=True):
    """
    Initializes the car. This must be run before we can control the car.

    Author: norman
    """

    start = "!start1590\n"
    inits = "!inits0.5\n"
    kp = "!kp0.01\n"
    kd = "!kd0.01\n"
    straight = "!straight1500\n"
    if pid:
        pid = "!pid1\n"
    else:
        pid = "!pid0\n"

    ser.write(start.encode())
    ser.write(inits.encode())
    ser.write(kp.encode())
    ser.write(kd.encode())
    ser.write(straight.encode())
    ser.write(pid.encode())

    drive(0.0)
    steer(0.0)
