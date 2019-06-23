import cv2 as cv2
from pygame import mixer # Load the required library

def capture_image(camId, image_nums, save_path):
    cam = cv2.VideoCapture(camId)
    count = 0
    mixer.init()
    mixer.music.load('resources/beep-07.mp3')
    while count < image_nums + 1:
        valid, image = cam.read()
        if valid:
            if count > 1:
                fname = save_path + "/cam_"+str(camId)+"_"+str(count)+".jpeg"
                cv2.imwrite(fname, image)
                cv2.imshow("cam",image)
                cv2.waitKey(2000)
                mixer.music.play(8)
                cv2.waitKey(2000)
            count += 1


if __name__ == "__main__":
    capture_image(camId = 1, 
                image_nums = 2, 
                save_path = "DataSet/webcam")