import serial
import time

ser = serial.Serial("/dev/ttyUSB0", 115200)
time.sleep(3)
ser.flushInput()
time.sleep(3)
ser.write("!start1600\n".encode())
time.sleep(3)

try:
    print("Starting Loop")
    while True:
        print("go")
        ser.write("!speed0\n".encode())
        time.sleep(1)
        print("stop")
        ser.write("!speed1\n".encode())
        time.sleep(1)
except KeyboardInterrupt:
    ser.write("!speed0\n".encode())
    print("Done!")