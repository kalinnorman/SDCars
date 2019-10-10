import serial
import time

ser = serial.Serial("/dev/ttyUSB0", 115200)
ser.flushInput()

ser.write("!start1600\n".encode())
time.sleep(1)

try:
    print("Starting Loop")
    while True:
        ser.write("!speed0\n".encode())
        time.sleep(1)
        ser.write("!speed1\n".encode())
        time.sleep(1)
except KeyboardInterrupt:
    ser.write("!speed0\n".encode())
    print("Done!")