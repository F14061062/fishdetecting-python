from tkinter.constants import BOTH
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from tkinter import ttk
import time
import datetime


class cnn_yolo_detect():
    def __init__(self):
        
        self.weights_path =  "1020.weights"
        self.cfg_path = "123.cfg"
        self.classes = ["tuna-head","fish-tail","marlin-head","shark-head"]
        self.net = cv2.dnn.readNet(self.weights_path,self.cfg_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def integrate(self,img):
        outs , height , width =self.start_detect(img)
        outs = np.array(outs)
        boxes,confidences,class_ids = self.detect_output_arrange(outs,height,width)
        #img=cv2.Canny(img,0,255)
        #img = np.hstack((v1,v2))
        img = self.output_arrange_convert_frame(boxes,confidences,class_ids,img)
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
        heads=[]
        tail=[]
        temp=0
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
                    if class_id == 1:
                        tail.append(temp)
                    else:
                        heads.append(temp)
                    temp=temp+1

        tail = np.array(tail)
        heads = np.array(heads)
        return boxes,confidences,class_ids

    #@jit(nopython=True)
    def output_arrange_convert_frame(self,boxes,confidences,class_ids,img):
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        #print(type(boxes))
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(w)+str(self.classes[class_ids[i]])+str("  ")+str(int(float(confidences[i])*100))+str("%")
                color = self.colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y), font, 1.1, color, 1)
        return img

    def Caculate_length(self,boxes,tail,heads,img):
        def areas(x1,y1,x2,y2):
            height = abs(y2-y1)
            length = abs (x2-x1)
            return height*length
        head_box_array = np.ones((heads.shape[0],4))
        for i in heads:
            x1, y1, w1, h1 = boxes[i]
            area_array=np.ones(tail.shape[0])
            box_array=np.ones((tail.shape[0],4))
            for ii in tail:
                x2, y2, w2, h2 = boxes[ii]
                _1xarray=np.array(0,w1,0,w1)
                _1yarray=np.array(0,0,h1,h1)
                _2xarray=np.array(0,w2,0,w2)
                _2yarray=np.array(0,0,h2,h2)
                area =np.ones(16)
                for j in range(4):
                    xx1= x1 + _1xarray[j]
                    yy1 = y1 + _1yarray[j]
                    for k in range(4):
                        xx2= x2 + _2xarray[k]
                        yy2 = y2 + _2yarray[k]
                        area[i*4+ii]=areas(xx1,yy1,xx2,yy2)
                target = np.argmax(area)
                _max = np.max(area)
                xx1 = x1+_1xarray[int(abs(target/4))]
                yy1 = y1+_1yarray[int(abs(target/4))]
                xx2 = boxes[ii]+_1xarray[int(abs(target%4))]
                yy2 = boxes[ii]+_1yarray[int(abs(target%4))]
                area_array[ii]=_max
                box_array[ii]=xx1,yy1,xx2,yy2
            target = np.argmax(area_array)
            head_box_array[i]=box_array[target]

    def Caculate_length2(self,boxes,tail,heads,img):
        def areas(x1,y1,x2,y2):
            height = abs(y2-y1)
            length = abs (x2-x1)
            return height*length
        head_box_array = np.ones((heads.shape[0],4))
        for i in heads:
            x1, y1, w1, h1 = boxes[i]
            _x1_square_array=np.ones(4)*x1
            _y1_square_array=np.ones(4)*y1
            _1xarray=np.array(0,w1,0,w1)
            _1yarray=np.array(0,0,h1,h1)
            _x1_square_array=_x1_square_array-_1xarray
            _y1_square_array=_y1_square_array-_1yarray
            for ii in tail:
                x2, y2, w2, h2 = boxes[ii]
                _x2_square_array=np.ones(4)*x2
                _y2_square_array=np.ones(4)*y2
                _2xarray=np.array(0,w2,0,w2)
                _2yarray=np.array(0,0,h2,h2)
                _x2_square_array=_x2_square_array-_2xarray
                _y2_square_array=_y2_square_array-_2yarray





class ipcam_detect():
    
    def __init__(self):
        self.cap = cv2.VideoCapture("http://192.168.0.20/video1.mjpg")#162
        self.cnn =cnn_yolo_detect()
        temp=cv2.imread('python1s_grey.png')
        self.temp =cv2.Canny(temp,100,200)
        

    def get_image(self):
        ret, img = self.cap.read()
        img1= cv2.Canny(img,200,210)
        img= self.cnn.integrate(img)
        img = self.imageprocess(img,img1)
        return img

    def imageprocess(self,img,img1):
        blue,green,red = cv2.split(img)
        img = cv2.merge((red,green,blue))
        #img1= cv2.Canny(img,150,210)
        img1=img1-self.temp
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(np.sum(img1)/255)
        img = np.hstack((img,img1))
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        return imgtk



class mainapp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.snapshot = ipcam_detect()

    def set_pic(self,signal):
        
        img=self.snapshot.get_image()
        self.label= tk.Label(app.frame1,image=img)
        self.label.imgtk = img
        self.label.pack()
        self.Sent_detect_information(signal)
        self.refresh_pic()

    def refresh_pic(self):
        img = self.snapshot.get_image()
        self.label.configure(image=img)
        self.label.imgtk = img
        
        self.after(10,lambda:self.refresh_pic())

    def Sent_detect_information(self,signal):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        signal.insert('', 'end', text="1", values=(st, "lab611", "dolphin fish", "4.6", "22.632558", "119.168472"))
        self.after(10000000,lambda:self.Sent_detect_information(signal))

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
        self.master.title("IP-Camara")
        self.frame1 = tk.Frame(self.master, relief=tk.RAISED, borderwidth=2)

        self.frame1.pack(side=tk.TOP,fill='x')
        self.frame2 = tk.Frame(self.frame1, relief=tk.RAISED, borderwidth=5,height=500)

        self.frame2.pack(side=tk.BOTTOM,fill='x')



class maintable():
    def __init__(self):
        style = ttk.Style()
        style.theme_use('clam')
        self.tree =ttk.Treeview(app.frame2, column=("Date_Time", "ID", "Fish_Type","Fish_Length","Latitude","Longitude"), show='headings', height=20)
        self.createtable()

    def createtable(self):
        self.tree.column("# 1", anchor=tk.CENTER,width=150)
        self.tree.heading("# 1", text="Date_Time")
        self.tree.column("# 2", anchor=tk.CENTER,width=150)
        self.tree.heading("# 2", text="ID")
        self.tree.column("# 3", anchor=tk.CENTER,width=150)
        self.tree.heading("# 3", text="Fish_Type")
        self.tree.column("# 4", anchor=tk.CENTER,width=150)
        self.tree.heading("# 4", text="Fish_Length")
        self.tree.column("# 5", anchor=tk.CENTER,width=150)
        self.tree.heading("# 5", text="Latitude")
        self.tree.column("# 6", anchor=tk.CENTER,width=150)
        self.tree.heading("# 6", text="Longitude")
        self.tree.pack()

roost = mainapp()
roost.geometry('1000x1000')
roost.resizable(width=False, height=False)
app=Application()

test = maintable()
#test.tree.insert('', 'end', text="1", values=("2021-08-03 15:08:16", "lab611", "dolphin fish", "4.6", "22.632558", "119.168472"))
roost.set_pic(test.tree)
#test.tree.insert('', 'end', text="1", values=("2021-08-03 11:08:16", "lab611", "shark", "6.2", "22.641158", "119.187472"))
#roost.Sent_detect_information(test.tree)
app.mainloop()

