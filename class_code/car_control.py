# car-control.py

import serial
import time

ser = serial.Serial("/dev/ttyUSB0", 115200)
ser.flushInput()
time.sleep(0.5)


def drive(speed):
    forward_command = "!drive" + str(speed) + "\n"
    ser.write(forward_command.encode())


def steer(degree):
    steer_command = "!turn" + str(degree) + "\n"
    ser.write(steer_command.encode())


def send_command(ser, command, addnewline=False):
    """
    Sends a command to the car. Remember that every command must end with a new line.

    Author: redd

    :param ser: the serial port to send the string to
    :param command: the command to send
    :return: no return
    """
    if addnewline:
        command = command + "\n"
    ser.write(command.encode())


def initialize_car(pid=True):
    """
    Initializes the car. This must be run before we can control the car.

    Author: redd
    """

    # initialize controller values
    send_command(ser, "!start1590\n")
    send_command(ser, "!inits00.50\n")
    send_command(ser, "!kp00.01\n")
    send_command(ser, "!kd00.01\n")
    send_command(ser, "!straight1500\n")

    if pid:
        send_command(ser, "!pid1")
    else:
        send_command(ser, "!pid0")

