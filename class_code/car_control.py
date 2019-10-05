# car-control.py
'''
******* Controlling The Car **********
	steer(int degree)	1000 = Full left turn	
				2000 = Full right turn	
				1500 = (nearly) Straight
	drive(int speed)  	1000 = Fast reverse 	
				2000 = Fast forward	
				1500 = (nearly) Stopped
				EXTREMELY IMPORTANT: Be very careful whe controlling the car. 
				NEVER tell it to go full speed. Safely test the car to find 
				a safe range for your particular car, and don't go beyond that 
				speed. These cars can go very fast, and there is expensive hardware 
				on them, so don't risk losing control of the car and breaking anything.
**************************************
'''

import serial
import time

ser = serial.Serial("/dev/ttyUSB0", 115200)
ser.flushInput()
time.sleep(2)

def drive(speed):
    forward_command = "!drive" + str(speed) + "\n"
    ser.write(forward_command.encode())
   
def steer(degree):
    steer_command = "!turn" + str(degree) + "\n"
    ser.write(steer_command.encode())

'''
#ser.write(("!turn1800\n").encode())
'''

