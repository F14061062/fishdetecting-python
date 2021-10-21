import numpy as np
import math
import cv2
from numpy import ma
from numpy.core.defchararray import center
class cnn_yolo_detect():
    def __init__(self):
        
        self.weights_path =  "1020.weights"
        self.cfg_path = "123.cfg"
        self.classes = ["tuna-head","fish-tail","marlin-head","shark-head"]
        self.net = cv2.dnn.readNet(self.weights_path,self.cfg_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def integrate(self,img,img1):
        outs , height , width =self.start_detect(img)
        outs = np.array(outs)
        boxes,confidences,class_ids = self.detect_output_arrange(outs,height,width)
        #img=cv2.Canny(img,0,255)
        #img = np.hstack((v1,v2))
        img ,img1,name, center = self.output_arrange_convert_frame1(boxes,confidences,class_ids,img,img1)
        return img , img1,name ,center


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
                cv2.circle(img,(int(x+w/2), int(y+h/2)), 5, (255, 255, 0), -1)
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
                cv2.circle(img,(int(x+w/2), int(y+h/2)), 10, (255, 255, 0), -1)
                cv2.circle(img1,(int(x+w/2), int(y+h/2)), 10, (255, 255, 0), -1)
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

class head_tail():
    def __init__(self):
        self.head_name=[]
        self.head_position=[]
        self.tail_position=[]

    def split_name(self,name,center):
        for count,i in enumerate(name):
            if i == 1:
                self.tail_position.append(center[count])
            else:
                self.head_name.append(i)
                self.head_position.append(center[count])

        



class ipcam_detect():
    
    def __init__(self):
        self.cap = cv2.VideoCapture("http://192.168.0.20/video1.mjpg")#162
        self.cnn =cnn_yolo_detect()
        temp=cv2.imread('python1s_grey.png')
        self.temp =cv2.Canny(temp,0,230)
        

    def get_image(self):
        def rotate_img(img,deg):
            (h, w,s) = img.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, deg, 1)
            rotate_img = cv2.warpAffine(img, M, (w, h))
    
            return rotate_img

        def rotating_matrix():
            
            pass
    

        def get_degree(x1,y1,x2,y2):
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
        
        def test1(a,b,degree,frame=None):
            
            _ax_array=[0,a[2],0,a[2]]
            _ay_array=[0,0,a[3],a[3]]
            _bx_array=[0,b[2],0,b[2]]
            _by_array=[0,0,b[3],b[3]]
            ax_array=np.ones(4)*a[0]+_ax_array
            ay_array=np.ones(4)*a[1]+_ay_array
            bx_array=np.ones(4)*b[0]+_bx_array
            by_array=np.ones(4)*b[1]+_by_array
            a_array = np.vstack((ax_array,ay_array))
            b_array= np.vstack((bx_array,by_array))
    
            test = np.hstack((a_array,b_array))
            test=rotate_matrix2(degree,test,frame)
            #for i in range(test.shape[1]):
                #cv2.circle(frame,(int(test[0][i]), int(test[1][i])), 4, (255, 255, 0), -1)
            max_x = int(np.max(test[0]))
            min_x = int(np.min(test[0]))
            max_y = int(np.max(test[1]))
            min_y = int(np.min(test[1]))
            return frame[min_y:max_y,min_x:max_x]

        def caculate_angle(a_box,b_box):
            degree =get_degree(a_box[0]+a_box[2]/2,a_box[1]+a_box[3]/2,b_box[0]+b_box[2]/2,b_box[1]+b_box[3]/2)
            return degree/math.pi*180-90

        def image_transpose(frame):
            if frame.shape[0]>=frame.shape[1]:
                frame=np.rot90(frame)
            return frame

        def determin_the_length(head_position,tail_position,frame):
            temp=[]
            ttemp=[]
            for i in head_position:
                #ans=np.array()
                var_array=np.ones(len(tail_position))
                sum_array=np.ones(len(tail_position))
                boxx =[]
                tt =0
                for j in tail_position:
                    degree = caculate_angle(i,j)
                    #degree=0
                    imgg=rotate_img(frame,degree)
                    fframe = test1(i,j,-degree,imgg)
                    ffframe = image_transpose(fframe)
                    fframe =cv2.Canny(ffframe,50,200)
                    sum =np.sum(fframe, axis=0)
                    var= np.var(sum/frame.shape[0])
                    sum= np.sum(fframe/255)
                    print("var",var)
                    print("sum",sum)
                    temp.append(fframe)
                    var_array[tt]=var
                    sum_array[tt]=sum
                    boxx.append(ffframe)
                    tt+=1
                ttemp.append(boxx[np.argmin(var_array)])
                tail_position.pop(np.argmin(var_array))
            return ttemp
        ret, imgg = self.cap.read()
        img11= cv2.Canny(imgg,180,210)
        img,img1,name,center= self.cnn.integrate(imgg,img11)
        img = self.imageprocess(img,img1)

        ret, imgg = self.cap.read()
        #print(name)
        #print(center)
        filter1 = head_tail()
        filter1.split_name(name,center)
        #print(filter1.head_name)
        print("ss")
        print("here")
        tt= determin_the_length(filter1.head_position,filter1.tail_position,imgg)
        print(len(tt))
        print("ss1")
        #print(tt[1].shape)
        
        return imgg,tt
        ret, imgg = self.cap.read()
        a,b=filter1.head_position
        c,d=filter1.tail_position
        degree1 =get_degree(b[0]+b[2]/2,b[1]+b[3]/2,d[0]+d[2]/2,d[1]+d [3]/2)
        degree =get_degree(a[0]+a[2]/2,a[1]+a[3]/2,d[0]+d[2]/2,d[1]+d [3]/2)
        degree2=degree1/math.pi*180-90
        #degree2=degree1/math.pi*180-90
        print("sssssss")
        print(int(degree2)+360)
        imgg=rotate_img(imgg,-degree2)
        print(-int(degree2)) 
        fframe = test1(b,d,degree2,imgg)
        print(fframe.shape)
        fframe= cv2.Canny(fframe,100,210)
        '''
        ret, imgg = self.cap.read()
        img11= cv2.Canny(imgg,100,210)
        testframe = img11[int(b[1]):int(b[1]+b[3]),int(b[0]):int(d[0]+b[2])]
        t1 =np.sum(testframe, axis=0)
        print("ss")
        print(np.var(t1))
        print(np.sum(testframe))
        testframe2 = img11[int(c[1]):int(b[1]+b[3]),int(b[0]):int(b[0]+b[2])]
        
        print("ss")
        testframe2=np.transpose(testframe2)
        t2 =np.sum(testframe2, axis=0)
        print(np.var(t2))
        print(np.sum(testframe2))
        print(degree/math.pi*180)
        print(img.shape)
        #img = rotate_img(img,degree/math.pi*180-90)
        print(testframe2.shape)
        #for i in filter1.head_position:
        #    cv2.circle(img,(int(i[0]+i[2]/2), int(i[1]+i[3]/2)),30, (255, 255, 0), -1)
        '''
        return fframe#imgg

    def imageprocess(self,img,img1):
        img1=img1-self.temp
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(np.sum(img1)/255)
        #   img = np.hstack((img,img1))
        return img1


def rotate_img(img):
    (h, w) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotate_img = cv2.warpAffine(img, M, (w, h))
    
    return rotate_img

#print(np.argmax())
a=ipcam_detect()
frame,tt = a.get_image()
#frame = rotate_img(frame)
for i in tt:
    print(i.shape)

while(True):
    cv2.imshow('frame', tt[0] )
    cv2.imshow('frame1',tt[1])
    #cv2.imshow('frame2',tt[2])
    #cv2.imshow('frame3',tt[3])
    cv2.imshow('frame4',frame)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        cv2.imwrite('python_grey.png',frame)
        break

cv2.destroyAllWindows()
    