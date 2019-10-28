"""
A series of helper functions for our car.
Author: redd
"""

import serial


# initialize communication with the arduino
ser = serial.Serial("/dev/ttyUSB0", 115200)
ser.flushInput()


def send_command(ser, command, addnewline=False):
    """
    Sends a command to the car. Remember that every command must end with a new line.
    :param ser:
    :param command:
    :return:
    """
    if addnewline:
        command = command + "\n"
    ser.write(command.encode())


def initialize_car(pid=True):
    """
    Initializes the car. This must be run before we can control the car.
    :return:
    """

    # initialize controller values
    send_command(ser, "!start1590\n")
    send_command(ser, "!inits0.5\n")
    send_command(ser, "!kp0.01\n")
    send_command(ser, "!kd0.01\n")
    send_command(ser, "!straight1500\n")

    if pid:
        send_command(ser, "!pid1")
    else:
        send_command(ser, "!pid0")

    return ser
