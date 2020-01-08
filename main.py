import numpy as np
import tkinter
import cv2
# PIL (Python Imaging Library)
import PIL.Image, PIL.ImageTk
import dlib
import os
import sys
from samples import coco
from mrcnn import utils
from mrcnn import model as modellib
import sys
import colorsys
import random
from datetime import datetime
from keras import backend as K
from keras.models import load_model

# ------------for GUI------------
# 0 => back camera
# 1 => front camera
VIDEO_SOURCE_FROM = 1
# Windows width and height
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
# Image size
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
# APP title
APP_TITLE = "test"
BACKGROUND_DICT = {}
GOTO_LIST = [
    "beach",
    "baseball_field",
    "hawai",
    "KFC",
    "korea"
]
BACKGROUND_IMAGE_NAME_LIST = [
    "goto_image/beach.png",
    "goto_image/baseball_field.png",
    "goto_image/hawai.png",
    "goto_image/KFC.png",
    "goto_image/korea.png",
]

# ------------for rcnn------------
# Root dir
ROOT_DIR = os.getcwd()
# Log & Model dir
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Weights file path
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "rcnn_weight/mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# COCO dataset object names
rcnn_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
print("#### SETTING MODEL DONE ####")
# Load weight file
rcnn_model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Labels 
class_names = [
    'BG','person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush', 'swimming pool',
    'towel','FitnessBike', 'Treadmill'
]

class RCNNF:
    def filter_r(self, r):
        # filter useful result
        filter_idex = []
        for i,clss_id in enumerate(r['class_ids']):
            if class_names[clss_id] in ['person']:
                filter_idex.append(i)

        r['rois'] = np.array([_ for i,_ in enumerate(r['rois']) if i in filter_idex])
        r['masks'] = np.concatenate([r['masks'][...,i:i+1] for i,_ in enumerate(r['masks']) if i in filter_idex],axis = -1)
        r['class_ids'] = np.array([_ for i,_ in enumerate(r['class_ids']) if i in filter_idex])
        r['scores'] = np.array([_ for i,_ in enumerate(r['scores']) if i in filter_idex])
        
        return r

    # display_instances(image, rr['rois'], rr['masks'], rr['class_ids'], class_names, rr['scores'], figsize=(8, 8), show_bbox = True)
    def display_instances(self, image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
        """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        title: (optional) Figure title
        show_mask, show_bbox: To show masks and bounding boxes or not
        figsize: (optional) the size of the image
        colors: (optional) An array or colors to use with each object
        captions: (optional) A list of strings to use as captions for each object
        """
        # Number of instances
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        # Generate random colors
        colors = colors or self.random_colors(N)

        # Show area outside image boundaries.
        height, width = image.shape[:2]

        black = np.zeros(image.shape)
        for i in range(N):
            color = colors[i]

            # Label
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]

            # Mask
            mask = masks[:, :, i]
            color = [0,0,1]
            
            # The mask
            black = self.apply_mask_cover(black, mask, color)

        img = PIL.Image.fromarray(black.astype('uint8'), 'RGB')
        
        return img

    def random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def apply_mask_cover(self, image, mask, color, alpha=0.5):
        """
        Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(
                mask == 1,
                color[c]*255,
                image[:, :, c]
            )
        return image


class APP:
    def __init__(self, window, window_title, video_source=VIDEO_SOURCE_FROM):
        self.window = window
        self.window.title(window_title)
        # Unresizable
        self.window.resizable(False, False)
        self.video_source = video_source
        self.mode = "normal"
        self.goto = "nowhere"

        # OPen the detector
        self.imageInterpreter = ImageInterpreter()
        
        # Open the video source
        self.videoObj = VideoCapture(self.video_source)

        # Create a canvas 
        self.canvas = tkinter.Canvas(self.window, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        self.canvas.pack(side="top", fill="both", expand=True)

        # Create a button frame
        self.button_frame = tkinter.Frame(self.window)
        self.button_frame.pack(side="bottom", fill="x", expand=False)

        # Create a button element
        self.btn_snapshot = tkinter.Button(self.button_frame, text="Snapshot", command=self.snapshot)
        self.btn_snapshot.pack(side="left")

        render_list = [
            "normal",
            "gray",
            "negative",
            "erode",
            "dilate",
            "smooth",
            "gaussian_blur",
            "face_detect",
            "skin_color_detect",
            "mosaic",
            "face_recognize"
        ]
        for item in render_list:
            bt = tkinter.Button(self.button_frame, text=item, bg="yellow", command=lambda x=item: self.set_mode(x))
            bt.pack(side="left")

        for item in GOTO_LIST:
            bt = tkinter.Button(self.button_frame, text="goto_"+item, bg="pink", command=lambda x=item: self.set_goto(x))
            bt.pack(side="left")

        # fps
        self.delay = 15
        self.update()

        # If read 'q', then end
        self.window.bind("q", lambda x: self.window.destroy())
        self.window.mainloop()

    def snapshot(self, frame=None):
        now = datetime.now().strftime("%H-%M-%S")
        if frame is None:
        # Get a frame from video source
            frame = self.videoObj.get_frame()
            frame = self.render(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_name = "snapshot/" + now + ".png"
        
        cv2.imwrite(image_name, frame)

    def set_mode(self, mode):
        self.mode = mode

    def set_goto(self, goto):
        self.goto = goto

    def render(self, frame):
        # Mode
        if self.mode == "normal":
            img = frame
        elif self.mode == "gray":
            img = self.imageInterpreter.gray(frame)
        elif self.mode == "negative":
            img = self.imageInterpreter.negative(frame)
        elif self.mode == "erode":
            img = self.imageInterpreter.erode(frame, 2)
        elif self.mode == "dilate":
            img = self.imageInterpreter.dilate(frame, 2)
        elif self.mode == "smooth":
            img = self.imageInterpreter.smooth(frame, 3)
        elif self.mode == "gaussian_blur":
            img = self.imageInterpreter.gaussian_blur(frame)
        elif self.mode == "face_detect":
            img = self.imageInterpreter.face_detect(frame)
        elif self.mode == "skin_color_detect":
            img = self.imageInterpreter.skin_color_detect(frame, 2)
        elif self.mode == "mosaic":
            img = self.imageInterpreter.mosaic(frame, 5)
        elif self.mode == "face_recognize":
            img = self.imageInterpreter.face_recognize(frame)
        else:
            img = frame

        # Goto
        if self.goto != "nowhere":
            pic = self.imageInterpreter.goto_somewhere(frame, BACKGROUND_DICT.get(self.goto))
            self.snapshot(pic)
            # Come back
            self.goto = "nowhere"
        
        return img

    def update(self):
        # Get a frame from video source
        frame = self.videoObj.get_frame()

        # Render
        frame = self.render(frame)

        # Create a photo image which is Tkinter-compatible
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))

        # NW is north-west
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        # Get the frame, and wait 'delay' ms to update
        self.window.after(self.delay, self.update)
    


class VideoCapture:
    def __init__(self, video_source=VIDEO_SOURCE_FROM):
        # Open the video source
        self.videoObj = cv2.VideoCapture(video_source)
        if not self.videoObj.isOpened():
            raise ValueError("Failed to open camera", video_source)

        # Set the video size
        self.videoObj.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
        self.videoObj.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

        # Get the video source width and height
        self.width = self.videoObj.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.videoObj.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.videoObj.isOpened():
            # Read the frame from video source
            ret, frame = self.videoObj.read()
            frame = np.fliplr(frame)
            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                print("Get frame failed.")
                return None
        else:
            print("Cannot get frame. camera doesn't open!")
            return None

    # Release the video source when object is destroyed
    def __del__(self):
        if self.videoObj.isOpened():
            self.videoObj.release()

class ImageInterpreter:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_recognizer = FaceRecognizer()
        self.rcnn = RCNNF()
        self.erode_mask_size = 3
        self.dilate_mask_size = 3
        self.gaussian_blur_mask_size = 9
        self.contours_thickness = 2
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.blue = (255, 0, 0)
        
    def gray(self, img):
        # Convert image from BGR to gray
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def negative(self, img):
        # Convert image from BGR to negative
        return (255 - img).astype(np.uint8)

    def erode(self, img, level):
        # Erode with struct element 'kernel'
        kernel = np.ones((self.erode_mask_size, self.erode_mask_size), np.uint8)
        return cv2.erode(img, kernel, iterations=level)

    def dilate(self, img, level):
        # Dilate with struct element 'kernel'
        kernel = np.ones((self.dilate_mask_size, self.dilate_mask_size), np.uint8)
        return cv2.dilate(img, kernel, iterations=level)

    def smooth(self, img, level):
        img = self.erode(img, level)
        img = self.dilate(img, level)
        return img
    
    def contours_detect(self, img):
        # RETR_EXTERNAL => Retrieves only the extreme outer contours
        # CV_CHAIN_APPROX_SIMPLE => Simple compression
        return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def gaussian_blur(self, img):
        # 0 => Auto compute color deviation
        return cv2.GaussianBlur(img, (self.gaussian_blur_mask_size, self.gaussian_blur_mask_size), 0)

    def face_detect(self, img):
        face_rects = self.face_detector(img, 0)
        for i, d in enumerate(face_rects):
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            # Draw the face region rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
        return img

    def draw_contours(self, img, contours, color):
        cv2.drawContours(img, contours, -1, color, self.contours_thickness)

    def skin_color_detect(self, img, level):
        # Convert image from RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Blur the image
        img = self.gaussian_blur(img)

        # Convert image from BGR to YCcCb
        mask_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        yellow_skin_mask = (((133 <= mask_img[:,:,1]) & (mask_img[:,:,1] <= 177)) & ((98 <= mask_img[:,:,2]) & (mask_img[:,:,2] <= 122))).astype(np.uint8)
        
        # Smooth the image
        yellow_skin_mask = self.smooth(yellow_skin_mask, 1)

        # Get the image contours
        contours, hierarchy = self.contours_detect(yellow_skin_mask)

        # Draw contours
        self.draw_contours(img, contours, self.green)

        # Convert image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img
    
    def get_rid_of_background(self, img):
        # Detect human
        results = rcnn_model.detect([img], verbose=0)
        print("#### detect done ####")

        # Filter the result
        r = results[0]
        r = self.rcnn.filter_r(r)

        # Get mask
        mask = self.rcnn.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        
        # Convert PIL.Image to np.array
        mask = np.array(mask)
        
        # Convert RGB to gray
        gray_img = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        # Convert gray to binary
        ret, bin_mask = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Mask
        img = cv2.bitwise_and(img, img, mask=bin_mask)

        return img

    def goto_somewhere(self, img, somewhere):
        # Copy somewhere
        where = somewhere.copy()

        # Get rid of background
        human = self.get_rid_of_background(img)

        # Image width and height
        height, width, channel = human.shape

        # Create an image BGRA
        r_channel, g_channel, b_channel = cv2.split(human)
        alpha_channel = np.where(human==0, 0, 255).astype('uint8')
        
        human = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        human = np.delete(human, np.s_[4:6], 2)

        # Alpha mix
        alpha_human = human[:, :, 3] / 255.0
        alpha_where = 1.0 - alpha_human

        for c in range(0, 3):
            where[0:height, 0:width, c] = (alpha_human*human[:, :, c] + alpha_where*where[0:height, 0:width, c])

        where = cv2.cvtColor(where, cv2.COLOR_BGRA2RGB)

        return where

    def mosaic(self, img, level):
        height, width, channel = img.shape
        img = cv2.resize(img, (int(width/level**2), int(height/level**2)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        
        return img

    def face_recognize(self, img):
        return self.face_recognizer.face_recognize(img)

class FaceCatcher:
    def __init__(self):
    	# Face classifier (use haar)
        self.classifier = cv2.CascadeClassifier("face_detect_model/haarcascade_frontalface_alt2.xml")
        self.path_name = "./face_sample/"
        self.img_number = 0

    def catch_sample(self, frame):
        # Converting to gray image can reduce calculation overhead
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = self.classifier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32,32))

        if len(face_rects) > 0:
            for face_rect in face_rects:
                x, y, w, h = face_rect

                # Save image
                img_name = "%s/%d.jpg" % (self.path_name, self.img_number)
                image = frame[y-10:y+h+10, x-10:x+w+10]
                cv2.imwrite(img_name, image)
                
                self.img_number += 1

class FaceRecognizer:
    def __init__(self):
        self.classifier = cv2.CascadeClassifier("face_detect_model/haarcascade_frontalface_alt2.xml")
        self.detect_model = load_model("face_detect_model/jerry.face.model.h5")
        self.detect_image_size = 64

    def face_recognize(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_rects = self.classifier.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=3, minSize=(32,32))

        if len(face_rects) > 0:
            for face_rect in face_rects:
                x, y, w, h = face_rect
                faceID = self.face_predict(img)

                cv2.rectangle(img, (x-10,y-10), (x+w+10,y+h+10), (255, 0, 0), 2)
                if faceID == 0:
                    cv2.putText(img, "jerry", (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                else:
                    cv2.putText(img, "unknown", (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    def face_predict(self, img):
		# th => (channels, rows, cols)
        if K.image_dim_ordering() == "th" and img.shape != (1, 3, self.detect_image_size, self.detect_image_size):
            img = self.resize_image(img)
            img = img.reshape((1, 3, self.detect_image_size, self.detect_image_size))
		# tf => (rows, cols, channels)
        elif K.image_dim_ordering() == "tf" and img.shape != (1, self.detect_image_size, self.detect_image_size, 3):
            img = self.resize_image(img)
            img = img.reshape((1, self.detect_image_size, self.detect_image_size, 3))

        img = img.astype("float32")
        img /= 255

        result = self.detect_model.predict_classes(img)

        return result[0]
    
    def resize_image(self, img, height=64, width=64):
        top, bottom, left, right = (0, 0, 0, 0)

        h, w, _ = img.shape

        longest_edge = max(h,w)

        # Calculate border size
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass

        BLACK = [0, 0, 0]

        constant = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

        return cv2.resize(constant, (height, width))

def init_background():
    for i in range(len(GOTO_LIST)):
        img = cv2.imread( BACKGROUND_IMAGE_NAME_LIST[i] )
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        b_channel, g_channel, r_channel = cv2.split(img)

        # 0 -> transparent, 255 => full
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype)*255

        img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        BACKGROUND_DICT.update({GOTO_LIST[i] : img})

def init_snapshot():
    if not os.path.exists("snapshot"):
        os.makedirs("snapshot")

def main():
    init_background()
    init_snapshot()
    APP(tkinter.Tk(), APP_TITLE)

if __name__ == '__main__':
    main()
