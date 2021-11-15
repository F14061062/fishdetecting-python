from tkinter.constants import BOTH
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from tkinter import ttk
import time
import datetime
import math
import pyodbc
from datetime import datetime
from numpy.lib.function_base import delete, insert

class cnn_yolo_detect():
    def __init__(self):
        
        self.weights_path =  "1104.weights"
        self.cfg_path = "123.cfg"
        self.classes = ["tail","tuna-head","marlin-head","shark-head"]
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

    def integrate2(self,img,img1):
        outs , height , width =self.start_detect(img)
        outs = np.array(outs)
        boxes,confidences,class_ids = self.detect_output_arrange(outs,height,width)
        #img=cv2.Canny(img,0,255)
        #img = np.hstack((v1,v2))
        img ,img1,name, center = self.output_arrange_convert_frame1(boxes,confidences,class_ids,img,img1)
        return img , img1,name ,center

    def start_detect(self,img):
        height=360
        width=640
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

    def output_arrange_convert_frame1(self,boxes,confidences,class_ids,img,img1):
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        center =[]
        name = []
        #print(type(boxes))
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(w)+str(self.classes[class_ids[i]])+str("  ")+str(int(float(confidences[i])*100))+str("%")
                color = self.colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y), font, 1.1, color, 1)
                #cv2.circle(img,(int(x+w/2), int(y+h/2)), 10, (255, 255, 0), -1)
                #cv2.circle(img1,(int(x+w/2), int(y+h/2)), 10, (255, 255, 0), -1)
                name.append(class_ids[i])
                center.append(boxes[i])
        return img,img1,name ,center    


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

class connect_to_database():
    def __init__(self):
        self.conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=140.116.85.227,57964;'
                      'Database=NCKUfish2;'
                      'Uid=root;'
                      'Pwd=123456;'
                      'Trusted_Connection=yes;')

    def write_sql(self,lines):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM fishdata2')
        cursor.execute("INSERT INTO fishdata2 VALUES('"+lines[0]+" "+lines[1]+"','"+lines[2]+"','"+lines[3]+"','"+lines[4]+"','"+lines[7]+"','"+lines[5]+"','"+lines[6]+"')")
        self.conn.commit()


class ipcam_detect():
    
    def __init__(self):
        self.cap = cv2.VideoCapture("http://192.168.0.20/video1.mjpg")#162
        self.cnn =cnn_yolo_detect()
        temp=cv2.imread('python1s_grey.png')
        self.temp =cv2.Canny(temp,100,200)
        

    def get_image(self):
        if (self.cap.isOpened()):
            ret, img = self.cap.read()
        else:
            self.cap.release()
            self.cap = cv2.VideoCapture("http://192.168.0.20/video1.mjpg")
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
        self.snapshot = ipcam_detect2()
        self.set_time=0
        self.rf_id = "rf_id.txt"

    def set_pic(self,signal):
        print("helllooooo")
        img,infor=self.snapshot.get_image()
        self.label= tk.Label(app.frame1,image=img)
        self.label.imgtk = img
        self.label.pack()
        #self.Sent_detect_information(signal)
        self.refresh_pic(signal)
        

    def refresh_pic(self,signal):
        result = time.localtime(time.time())
        img,infor = self.snapshot.get_image()
        self.label.configure(image=img)
        self.label.imgtk = img
        #print("sssssssss",time.time()-self.set_time)
        if (len(infor)!=0):
            if (time.time()-self.set_time) >300:
                self.set_time = time.time()
                self.Sent_detect_information1(signal,infor)

        self.after(10,lambda:self.refresh_pic(signal))

    def get_rf_id(self):
        a_file =open(self.rf_id,"r")
        lines = a_file.readlines()
        a_file.close()
        return lines

    def write_rf_id(self ,lines):
        del lines[0]
        a_file = open(self.rf_id,"w+")
        for i in lines:
            a_file.write(i)
        a_file.close()

    def Sent_detect_information(self,signal):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        signal.insert('', 'end', text="1", values=(st, "lab611", "dolphin fish", "4.6", "22.632558", "119.168472"))
        #self.after(10000000,lambda:self.Sent_detect_information(signal))
    def Sent_detect_information1(self,signal,infor):
        for i in infor:
            lines = self.get_rf_id()
            if len(lines) >0:
                signal.insert('', 'end', text="1", values=(i[0], i[1], i[2], i[3]+" cm",i[4],i[5],lines[0]))
                temp=[i[0], i[1], i[2], i[3],i[4],i[5],lines[0]]
                his.append(temp)
                self.write_rf_id(lines)
            else:
                signal.insert('', 'end', text="1", values=(i[0], i[1], i[2], i[3]+" cm",i[4],i[5],"exceed"))
                temp=[i[0], i[1], i[2], i[3],i[4],i[5],"exceed"]
                his.append(temp)

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
        self.store_path="test.txt"
        self.connect_to_database =connect_to_database()

    def exitProgram(self):
        exit()

    def save_to_database(self):
        self.save_his()
        his.clear()
        lines=self.read_file_his()
        print(lines)
        for i in lines:
            self.connect_to_database.write_sql(i)

    def save_his(self):
        a_file =open("test.txt","r")
        lines = a_file.readlines()
        a_file.close()

        a_file = open("test.txt","w")
        for i in lines:
            a_file.write(i)
        for i in his:
            for ii in i:
                a_file.write(ii)
                a_file.write(" ")
            a_file.write("\n")
        a_file.close()

    def read_file_his(self):
        a_file =open("test.txt","r+")
        lines = a_file.readlines()
        a_file.close()
        for i in range(len(lines)):
            lines[i]=lines[i].split(" ")
            lines[i].pop(len(lines[i])-1)
        return lines
    def intui(self):
        
        print("hello")
        self.master.title("IP-Camara")
        self.frame1 = tk.Frame(self.master, relief=tk.RAISED, borderwidth=2)

        self.frame1.pack(side=tk.TOP,fill='x')
        self.frame2 = tk.Frame(self.frame1, relief=tk.RAISED, borderwidth=5,height=500)
        
        
        self.frame3 = tk.Frame(self.frame1, relief=tk.RAISED, borderwidth=5,height=200)
        self.frame3.pack(side=tk.BOTTOM,fill='x')
        self.frame2.pack(side=tk.BOTTOM,fill='x')
        mybutton = tk.Button(self.frame3, text='Save_data_to_data_base', command=self.save_to_database,width='20',height='5',borderwidth= 1,font=('Arial','10'),bg='#898282')
        mybutton.pack(side=tk.LEFT,fill='x')

class maintable():
    def __init__(self):
        style = ttk.Style()
        style.theme_use('clam')
        #self.tree =ttk.Treeview(app.frame2, column=("Date_Time", "ID", "Fish_Type","Fish_Length","Latitude","Longitude"), show='headings', height=20)
        self.tree =ttk.Treeview(app.frame2, column=("Date_Time", "ID", "Fish_Type","Fish_Length","Latitude","Longitude","RF-ID"), show='headings', height=20)
        self.createtable()

    def createtable(self):
        self.tree.column("# 1", anchor=tk.CENTER,width=150)
        self.tree.heading("# 1", text="Date_Time")
        self.tree.column("# 2", anchor=tk.CENTER,width=50)
        self.tree.heading("# 2", text="ID")
        self.tree.column("# 3", anchor=tk.CENTER,width=100)
        self.tree.heading("# 3", text="Fish_Type")
        self.tree.column("# 4", anchor=tk.CENTER,width=100)
        self.tree.heading("# 4", text="Fish_Length")
        self.tree.column("# 5", anchor=tk.CENTER,width=100)
        self.tree.heading("# 5", text="Latitude")
        self.tree.column("# 6", anchor=tk.CENTER,width=100)
        self.tree.heading("# 6", text="Longitude")
        self.tree.column("# 7", anchor=tk.CENTER,width=200)
        self.tree.heading("# 7", text="RF-ID")
        self.tree.pack()


class ipcam_detect2():
    
    def __init__(self):
        self.cap = cv2.VideoCapture("http://192.168.0.20/video1.mjpg")#162
        self.cnn =cnn_yolo_detect()
        temp=cv2.imread('python1s_grey.png')
        self.temp =cv2.Canny(temp,0,230)

    def imageprocess(self,img,img1):
        img1=img1-self.temp
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img1

    def imagetkprocess(self,img,img1):
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


    def get_image(self):
        def rotate_img(img,deg,hh,ww):
            (h, w,s) = img.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, deg, 1)
            rotate_img = cv2.warpAffine(img, M, (ww, hh))
    
            return rotate_img
        def get_degree(x1,y1,x2,y2):
            if (y1-y2)==0:
                return 0
            temp= (x1-x2)/(y1-y2)
            degree = math.atan(temp)
            return degree
            
        def rotate_matrix2(degree,box_array,frame=None):
            temp=np.ones((8,2))
            (h, w,g) = frame.shape
            center = (w // 2, h // 2)
            box_array[0]=box_array[0]-center[0]
            box_array[1]=box_array[1]-center[1]

            theta = np.radians(degree)
            c, s = np.cos(theta), np.sin(theta)
            r = np.array(((c, -s), (s, c)))

            for i in range(box_array.shape[1]):
                x=np.array([box_array[0][i],box_array[1][i]])
                tt=np.int_(np.dot(r,x))
                tt[0]=tt[0]+center[0]
                tt[1]=tt[1]+center[1]
                temp[i]=tt
            #print((temp))
            return np.transpose(temp)

        def rotate_matrix3(degree,a,b,frame=None):
            temp=np.ones((8,2))
            (h, w,g) = frame.shape
            center = (w // 2, h // 2)
            a[0]=a[0]+a[2]/2-center[0]
            a[1]=a[1]+a[3]/2-center[1]
            b[0]=b[0]+b[2]/2-center[0]
            b[1]=b[1]+b[3]/2-center[1]
            theta = np.radians(degree)
            c, s = np.cos(theta), np.sin(theta)
            r = np.array(((c, -s), (s, c)))

            
            x=np.array([a[0],a[1]])
            tt=np.int_(np.dot(r,x))
            tt[0]=tt[0]+center[0]
            tt[1]=tt[1]+center[1]
            a[0]=tt[0]
            a[1]=tt[1]
            x=np.array([b[0],b[1]])
            tt=np.int_(np.dot(r,x))
            tt[0]=tt[0]+center[0]
            tt[1]=tt[1]+center[1]
            b[0]=tt[0]
            b[1]=tt[1]

            #print((temp))
            return a,b       


        def test1(a,b,degree,frame=None):
            aa=[]
            bb=[]
            for i in range(len(a)):
                aa.append(a[i])
                bb.append(b[i])
            aa,bb=rotate_matrix3(degree,aa,bb,frame)
            
            _ax_array=[aa[2]/2,aa[2]/2,-aa[2]/2,-aa[2]/2]
            _ay_array=[aa[3]/2,-aa[3]/2,-aa[3]/2,aa[3]/2]
            _bx_array=[bb[2]/2,bb[2]/2,-bb[2]/2,-bb[2]/2]
            _by_array=[bb[3]/2,-bb[3]/2,-bb[3]/2,bb[3]/2]
            ax_array=np.ones(4)*aa[0]+_ax_array
            ay_array=np.ones(4)*aa[1]+_ay_array
            bx_array=np.ones(4)*bb[0]+_bx_array
            by_array=np.ones(4)*bb[1]+_by_array
            a_array = np.vstack((ax_array,ay_array))
            b_array= np.vstack((bx_array,by_array))
    
            test = np.hstack((a_array,b_array))
            f=lambda a: (abs(a)+a)/2 
            max_x = int(f(np.max(test[0])))
            min_x = int(f(np.min(test[0])))
            max_y = int(f(np.max(test[1])))
            min_y = int(f(np.min(test[1])))
              
            cv2.circle(frame,(int(aa[0]), int(aa[1])), 10, (0, 255, 0), -1)
            cv2.circle(frame,(int(bb[0]), int(bb[1])), 10, (0, 0, 255), -1)
            return frame[min_y:max_y,min_x:max_x]

        def caculate_angle(a_box,b_box):
            degree =get_degree(a_box[0],a_box[1],b_box[0],b_box[1])
            return degree/math.pi*180

        def image_transpose(frame):
            if frame.shape[0]>=frame.shape[1]:
                frame=np.rot90(frame)
            return frame

        def determin_the_length(head_position,tail_position,head_name,frame):
            classes = ["fish-tail","tuna","marlin","shark"]
            temp=[]
            ttemp=[]
            information=[]
            for i in range(len(head_position)):
                #temp=[]
                #ans=np.array()
                var_array=np.ones(len(tail_position))*100000
                sum_array=np.ones(len(tail_position))*100000
                boxx =[]
                tt =0
                for j in range(len(tail_position)):
                    (h, w,s) = frame.shape
                    degree = caculate_angle(head_position[i],tail_position[j])
                    imgg=rotate_img(frame,-degree,h,w)
                    fframe = test1(head_position[i],tail_position[j],degree,imgg)
                    ffframe = image_transpose(fframe)
                    fframe =cv2.Canny(ffframe,180,200)
                    fcframe = np.rot90(fframe)
                    fcframe=fcframe[int(fcframe.shape[0]/4):int(fcframe.shape[0]*3/4)][:]
                    fcframe = np.rot90(fcframe)
                    sum =np.sum(fframe/255, axis=0)
                    var= np.var(sum/frame.shape[0])-7
                    sum= np.sum(fcframe/255)
                    temp.append(ffframe)
                    if fframe.shape[1] >170 and fframe.shape[1] <250:                    
                        var_array[tt]=1*var-10*sum
                        sum_array[tt]=sum-10*var
                    #ffframe = np.rot90(fframe)
                        boxx.append(ffframe)
                        tt+=1 
                if(len(boxx)!=0):
                    #result = time.localtime(time.time())
                    #now_timr=str(result.tm_year)+"-"+str(result.tm_mon)+"-"+str(result.tm_mday)+" "+str(result.tm_hour)+":"+str(result.tm_min)+":"+str(result.tm_sec)
                    at =datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                    info_fish = ["Date","ID","Fish_type","length","latitude","longtitude"]
                    imagen=boxx[np.argmin(var_array)]
                    tail_position.pop(np.argmin(var_array))
                    info_fish[0]=at
                    info_fish[1]="lab611"
                    info_fish[2]=classes[head_name[i]]
                    info_fish[3]=str(imagen.shape[1]/5*4)
                    info_fish[4]="22.4576"
                    info_fish[5]="123.112"
                    information.append(info_fish) 
            return information
        ret, imgg = self.cap.read()
        img11= cv2.Canny(imgg,180,210)
        img,img1,name,center= self.cnn.integrate2(imgg,img11)
        filter1 = head_tail()
        filter1.split_name(name,center)
        tt= determin_the_length(filter1.head_position,filter1.tail_position,filter1.head_name,imgg)
        #img1= cv2.Canny(imgg,200,210)
        imgtk = self.imagetkprocess(imgg,img1)
        return imgtk,tt


class head_tail():
    def __init__(self):
        self.head_name=[]
        self.head_position=[]
        self.tail_position=[]

    def split_name(self,name,center):
        for count,i in enumerate(name):
            if i == 0:
                self.tail_position.append(center[count])
            else:
                self.head_name.append(i)
                self.head_position.append(center[count])
his=[]
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

