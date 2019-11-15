"""
    14 Nov 2019

    'Drive' class is the main one used. Lots of black magic
"""

class Drive:
    def __init__(self):
        """
            Drive class constructor 
        """
        # Variables
        self.speed = 0.25
        self.startup_speed = 0.4
        self.angle_multiplier = 0.7

        # Object instantiation 
        self.cc = CarControl()       
        self.predict = PredictiveFollower(self.lane_follow_img, search_radius=50)

        # Other variables
        self.gray_desired = 220 # 210 # The gray value that we want the car to follow
        self.cur_angle = 0
        self.cur_gps = (0, 0)
        self.cur_gray_val = 0
        self.cur_region = 0
        self.log_filename = datetime.now().strftime("%b-%d-%Y_%H:%M:%S") + ".txt" # Creates file named by the date and time to log items for debugging
        self.out_file = open("LogFiles/"+self.log_filename,"w") # Opens (creates the file)
        self.waypoints = []
        self.lane_follow_img = cv2.imread('Maps/binary_drivable_rounded_outside_expanded_inside_2_blurred.bmp') # Reads in the RGB image
        self.lane_follow_img = cv2.cvtColor(self.lane_follow_img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        self.regions_img = cv2.imread('Maps/regions.bmp') # RGB
        self.regions_img = cv2.cvtColor(self.regions_img, cv2.COLOR_BGR2GRAY) # Grayscale

        # Not sure...
        self.one_left = cv2.imread('Maps/3to2_and_1to4.bmp') # RGB
        self.one_left = self.three_right = cv2.cvtColor(self.one_left, cv2.COLOR_BGR2GRAY) # Grayscale
        self.one_right = cv2.imread('Maps/1to1_and_2to2.bmp') # RGB
        self.one_right = self.two_left = cv2.cvtColor(self.one_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.one_straight = cv2.imread('Maps/intersection_1_straight.bmp') # RGB
        self.one_straight = cv2.cvtColor(self.one_straight, cv2.COLOR_BGR2GRAY) # Grayscale
        self.two_right = cv2.imread('Maps/2to3_and_4to1.bmp') # RGB
        self.two_right = self.four_left = cv2.cvtColor(self.two_right, cv2.COLOR_BGR2GRAY) # Grayscale
        self.two_straight = cv2.imread('Maps/intersection_2_straight.bmp') # RGB
        self.two_straight = cv2.cvtColor(self.two_straight, cv2.COLOR_BGR2GRAY) # Grayscale
        self.three_left = cv2.imread('Maps/3to3_and_4to4.bmp') # RGB
        self.three_left = self.four_right = cv2.cvtColor(self.three_left, cv2.COLOR_BGR2GRAY) # Grayscale
        self.three_straight = cv2.imread('Maps/intersection_3_straight.bmp') # RGB
        self.three_straight = cv2.cvtColor(self.three_straight, cv2.COLOR_BGR2GRAY) # Grayscale
        self.four_straight = cv2.imread('Maps/intersection_4_straight.bmp') # RGB
        self.four_straight = cv2.cvtColor(self.four_straight, cv2.COLOR_BGR2GRAY) # Grayscale

    def get_angle(self,cur_gps,prev_gps):
        angle_rads = self.predict.find_angle(1600-cur_gps[1],cur_gps[0],1600-prev_gps[1],prev_gps[0])
        angle_deg = angle_rads*180.0/np.pi  # Convert angle from radians to degrees
        angle_mod = round(self.angle_multiplier*angle_deg)

        if abs(angle_mod) > 30: # Cap the angle at -30 and 30
            angle_mod = np.sign(angle_mod)*30
        self.cur_angle = angle_mod # update the class value tracking the current angle

        return angle_mod
    
    def get_gray_value(self, coordinates, img): 
        """
            Converts from cv2 coordinates to Dr. Lee's image coordinates

            @param coordinates: (Type) (Description...?)

            @param img: (Image) Some type of image...

            @return: (int) Desired gray value from the given x and y coordinate

        """
        imgWidth = img.shape[1] # Get width
        x = round(coordinates[0]) # x translates directly
        y = imgWidth - round(coordinates[1]) # y is inverted
        self.cur_gps = (x,y)
        self.cur_gray_val = img[x,y] # Obtains the desired gray val from the x and y coordinate
        return self.cur_gray_val

    def get_intersection_map(self, cur_region, desired_region):
        """
            Determines what action to take base on the current region and region of desired GPS coordinates
            Returns the appropriate map for the action to take, as well as the next region the car will be in.

            @param cur_region: (int?) The current region that the car is in (?)

            @param desired_region: (int?) The region that the desired GPS coordinates are in

            @return: (map) The map dictating the appropriate action for the car to take 

                     (int) The next region that the car will be in 

        """

        if cur_region == 1:
            if desired_region == 1:
                return self.one_right, 1
            elif desired_region == 2:
                return self.one_straight, 3
            elif desired_region == 3:
                return self.one_straight, 3
            elif desired_region == 4:
                return self.one_left, 4
            else:
                return self.lane_follow_img, 0
        elif cur_region == 2:
            if desired_region == 1:
                return self.two_straight, 4
            elif desired_region == 2:
                return self.two_left, 2
            elif desired_region == 3:
                return self.two_right, 3
            elif desired_region == 4:
                return self.two_straight, 4
            else:
                return self.lane_follow_img, 0
        elif cur_region == 3:
            if desired_region == 1:
                return self.three_straight, 1
            elif desired_region == 2:
                return self.three_right, 2
            elif desired_region == 3:
                return self.three_left, 3
            elif desired_region == 4:
                return self.three_straight, 1
            else:
                return self.lane_follow_img, 0
        elif cur_region == 4:
            if desired_region == 1:
                return self.four_left, 1
            elif desired_region == 2:
                return self.four_straight, 2
            elif desired_region == 3:
                return self.four_straight, 2
            elif desired_region == 4:
                return self.four_right, 4
            else:
                return self.lane_follow_img, 0
        else:
            return self.lane_follow_img, 0

    def get_region(self, coordinates):
        current_gray_val = self.get_gray_value(coordinates, self.regions_img)
        self.cur_region = gp.region_dict[gp.region_values[current_gray_val]]
        return self.cur_region

    def update_log_file(self):
        if self.cur_region == 0:
            out_string = "OUTSIDE OF GPS BOUNDS"
        else:
            out_string = "Region:"+str(self.cur_region)+" | GPS:"+str(self.cur_gps)+" | Gray:"+str(self.cur_gray_val)+" | Angle:"+str(self.cur_angle)
        self.out_file.write(out_string + "\n")

    def get_waypoints(self):
        try:
            waypoints_file = open(self.waypoints_filename,"r") # open the text file with waypoints
            lines = waypoints_file.readlines() # Reads in each line of the file into a list
            for i in lines:
                if not i.__contains__('#'): # If the line is not a comment
                    try:
                        x,y = i.split(',') # split the line at the comma
                        temp_tuple = (int(x),int(y)) # Create a tuple of the gps coordinates
                        self.waypoints.append(temp_tuple) # Add the tuple to the list of waypoints
                    except:
                        pass  # don't do anything
            if len(self.waypoints) == 0:
                print("ERROR: No valid waypoints in file,", self.waypoints_filename)
                print("Terminating program")
                sys.exit()
        except:
            print("ERROR: waypoints file:", self.waypoints_filename, "does not exist.")
            print("Terminating program")
            sys.exit()

    def get_next_coordinates(self):
        desired_coordinates = self.waypoints[0]
        des_x = desired_coordinates[0]
        des_y = desired_coordinates[1]
        desired_region = car.get_region(desired_coordinates)  # pass in tuple: (x,y)
        if desired_region == 0 or desired_region == 5:
            print("Desired coordinates", desired_coordinates, "are not located in a valid location but in region ", desired_region)
            car.out_file.close()
            sys.exit()

        return desired_coordinates, des_x, des_y, desired_region

    def update_desired_gray_val(self, region):
        if region == gp.region_dict['Region 1'] or region == gp.region_dict['Region 4']:
            self.predict.set_gray_val(215)
        else:
            self.predict.set_gray_val(215)

    def update_waypoints(self):
        self.waypoints_filename = "waypoints.txt"
        self.get_waypoints()

    def start_car(self):
        """
            Gets the car moving. 
                Speed to get car moving is often higher than desired speed.
                Starts at a higher speed to get the car in motion, then slows down to the desired driving speed

            @return: (None) None

        """
        self.cc.drive(self.startup_speed)  # get the car moving
        time.sleep(0.1)  # ...but only briefly
        self.cc.drive(self.speed)  # get the car moving again

    def stop_car(self):
        """
            Stops the car
                No brakes, but the 'gas' is turned off 

            @return: (None) None

        """
        self.cc.steer(0)
        self.cc.drive(0.0)
        time.sleep(0.1)