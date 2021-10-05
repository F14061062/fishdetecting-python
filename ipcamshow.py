from tkinter.constants import BOTH
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import cv2
from numba import jit


class cnn_yolo_detect():
    def __init__(self):
        self.weights_path =  "yolov4-tiny-obj_final.weights"
        self.cfg_path = "yolov4-tiny-obj.cfg"
        self.classes = ["tuna-head","tuna-tail","marlin-head","marlin-tail","shark-head","shark-tail"]
        self.net = cv2.dnn.readNet(self.weights_path,self.cfg_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def integrate(self,img):
        outs , height , width =self.start_detect(img)
        outs = np.array(outs)
        boxes,confidences,class_ids = self.detect_output_arrange(outs,height,width)
        #img = self.output_arrange_convert_frame(boxes,confidences,class_ids,img)
        return img


    def start_detect(self,img):
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        return outs,height,width

    #@jit(nopython=True)
    def detect_output_arrange(self,outs,height,width):
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        return boxes,confidences,class_ids

    #@jit(nopython=True)
    def output_arrange_convert_frame(self,boxes,confidences,class_ids,img):
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])+str("  ")+str(int(float(confidences[i])*100))+str("%")
                color = self.colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
        return img



class ipcam_detect():
    
    def __init__(self):
        self.cap = cv2.VideoCapture("http://192.168.0.162/video1.mjpg")
        self.cnn =cnn_yolo_detect()
        

    def get_image(self):
        ret, img = self.cap.read()
        img= self.cnn.integrate(img)
        img = self.imageprocess(img)
        return img

    def imageprocess(self,img):
        blue,green,red = cv2.split(img)
        img = cv2.merge((red,green,blue))
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        return imgtk



class mainapp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.snapshot = ipcam_detect()
        #self.cnn =cnn_yolo_detect()

    def set_pic(self):
        
        img=self.snapshot.get_image()
        #img= self.cnn.integrate(img)
        self.label= tk.Label(app.frame1,image=img)
        self.label.imgtk = img
        self.label.pack()
        self.refresh_pic()
    def refresh_pic(self):
        img = self.snapshot.get_image()
        #img= self.cnn.integrate(img)
        self.label.configure(image=img)
        self.label.imgtk = img
        self.after(1,lambda:self.refresh_pic())
        pass

class Application(tk.Frame):
    def __init__(self):
        super().__init__()
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)
        fileMenu = tk.Menu(menu,tearoff=0)
        fileMenu.add_command(label="Seting IP")
        fileMenu.add_command(label="Exit", command=self.exitProgram)
        menu.add_cascade(label="base   ", menu=fileMenu)

        editMenu = tk.Menu(menu,tearoff=0)
        editMenu.add_command(label="Undo")
        editMenu.add_command(label="Redo")
        menu.add_cascade(label="none", menu=editMenu,font=30)
        self.intui()
    def exitProgram(self):
        exit()

    def intui(self):
        
        print("hello")
        self.master.title("t265-distance")
        self.frame1 = tk.Frame(self.master, relief=tk.RAISED, borderwidth=2)

        self.frame1.pack(fill=tk.BOTH, expand=True)


roost = mainapp()
roost.geometry('900x600')
roost.resizable(width=False, height=False)
app=Application()
roost.set_pic()
app.mainloop()