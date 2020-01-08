import cv2
import os
import numpy as np

from face_train import Model

def catch_video(window_name, cam_idx, catch_pic_num, path_name, user):
	cv2.namedWindow(window_name)

	cap = cv2.VideoCapture(cam_idx)

	# face detect classifier
	# use haar
	classifier = cv2.CascadeClassifier("face_detect_model/haarcascade_frontalface_alt2.xml")

	# frame color
	color = (0, 255, 0)

	num = 0

	while cap.isOpened():
		ok, frame = cap.read()
		if not ok:
			break

		# Converting to gray can reduce calculation
		grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faceRects = classifier.detectMultiScale(grey,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
		
		if len(faceRects) > 0:
			for faceRect in faceRects:
				x, y, w, h = faceRect

				img_name = "%s/%s-%d.jpg"%(path_name, user, num)
				image = frame[y-10:y+h+10, x-10:x+w+10]
				cv2.imwrite(img_name, image)

				num += 1

				if num > catch_pic_num:
					break

				cv2.rectangle(frame, (x-10,y-10), (x+w+10,y+h+10), color, 2)
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(frame, "num:%d" % (num), (x+30, y+30), font, 1, (255,0,255), 4)

		if num > catch_pic_num:
			break

		cv2.imshow(window_name, frame)
		c = cv2.waitKey(10) & 0xFF
		if c == ord("q"):
			break

	cap.release()
	cv2.destroyAllWindows()

def get_user_name():
	user = str(input("Enter your name: "))
	return user

def main():
	user = get_user_name()
	if not os.path.exists("face_samples/"+user):
		os.makedirs("face_samples/"+user)

	catch_video("hello", 1, 500, "face_samples/"+user, user)

if __name__=="__main__":
	main()