import serial
import time

ser = serial.Serial("/dev/ttyUSB0", 115200)
ser.flushInput()
time.sleep(1)

start = "!start1590\n"
inits = "!inits0.5\n"
kp = "!kp0.01\n"
kd = "!kd0.01\n"
straight = "!straight1500\n"
ser.write(start.encode())
ser.write(inits.encode())
ser.write(kp.encode())
ser.write(kd.encode())
ser.write(straight.encode())

def drive(value):
    command = "!speed" + str(value) + "\n"
    ser.write(command.encode())

def steer(value):
    command = "!steering" + str(value) + "\n"
    ser.write(command.encode())

print("right")
steer(15.0)
time.sleep(2)
print("left")
steer(-15.0)
time.sleep(2)
