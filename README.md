# SDCars

Repository for Team 2, Self-Driving Cars class

## GitHub empty object file error (corrupted repository)

Run the command "find .git/objects/ -size 0 -delete" while located in the Desktop directory (the home directory of the GitHub repository)

## Jetson Notes

Note that both your computer and the Jetson must be on the NETGEAR24 network in order to SSH into it correctly (password is greattrail272).

If you do not see the `SDCars` or `class_code` folders on the Desktop when you power on the Jetson, it did not boot to the SD card. You must restart the Jetson if this happens. Unplug it, reinsert the SD card, and try again. If you use the `reboot` command in the terminal, you will likely not boot to the SD card. 

## Repo Notes

redd cloned this repo onto the Jeston's `~/Desktop` directory for easy access. This way we will easily be able to pull and push code. He included all of the code from `class_code` in the initial commit. This includes the modified script for `record_video.py` that actually records video.

## `record_video.py`

This script will record video from the car's camera until the user ends the script using `Ctrl+C`.

This script requires an `.avi` output in a directory that already exists. We ran into issues when trying to save to `video_output/` before that directory existed. After we created it, though, the file saved with no issues.

