from tkinter.constants import BOTH

import tkinter as tk
from tkinter import Label, ttk
import socket
import threading
import time
import tkinter.messagebox 
import os
from tkinter import simpledialog
class udp_read():
    def __init__(self):
        self.UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.UDP_server_create()
        self.temp =0
        
    def UDP_server_create(self):
        localIP     = "192.168.0.188"
        localPort   = 20001
        bufferSize  = 1024
        self.UDPServerSocket.bind((localIP, localPort))
    def messange_convert(self,message):
        convert=""
        for count,c in enumerate(message):
            if count >=10 and count <=21:
                convert+='%02x' %c
        return convert
    def get_message(self):
        message=self.UDPServerSocket.recvfrom(4096)
        message = self.messange_convert(message[0])
        return message

    def check_repeat(self,message):
        for i in his:
            if i == message:
                return False
        return True

    def while_loop_udp(self):
        while True:
            self.temp +=1
            message = self.get_message()

            if self.check_repeat(message):
                app.label.config(text="成功讀取",bg="green")
                his.append(message)
                print(his)
                print(message)
                test.tree.insert('', 'end', text="1", values=(self.temp,app.entryLabel1.get(), message,roost.USER_INP,app.entryLabel2.get()))
            else:
                app.label.config(text="請掃描下一個",bg="red")
                pass
            time.sleep(1)

        


class mainapp(tk.Tk):
    def __init__(self):
        super().__init__() 
        self.set_time=0
        self.temp=0
        self.USER_INP = simpledialog.askstring(title="請先輸入",
                          prompt="漁船")
        print("mosdasd")
        if self.USER_INP==None or self.USER_INP=="":
            os._exit(0)

        self.udp = udp_read()
        #self.refresh_pic()
        #t = threading.Thread(target = udp.rrr())

        
    def refresh_pic(self):
        pass
        #self.after(10,lambda:self.test())
        #self.Sent_detect_information1(signal,infor)

        #self.Sent_read_information1(signal,self.temp,message)

    #def test(self):
    #    t = threading.Thread(target = self.udp.rrr())
        
    
    def Sent_read_information1(self,signal,num,message):
        signal.insert('', 'end', text="1", values=(num, message))


class Application(tk.Frame):
    def __init__(self):
        super().__init__()
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        fileMenu = tk.Menu(menu,tearoff=0)
        fileMenu.add_command(label="Exit", command=self.exitProgram)
        menu.add_cascade(label="base", menu=fileMenu)
        self.intui()
    def exitProgram(self):
        os._exit(0)

    def write_to_text(self):
        
        filename = roost.USER_INP+(".txt")
        print(filename)
        f = open(filename, 'w')
        for i in his:
            f.write(i)
            f.write('\n')
        f.close()
    

    def output(self):
        tkinter.messagebox.showinfo(title='Message', message='Write to the text and exit!')
        self.write_to_text()
        os._exit(0)

    def intui(self):
        self.master.title("Read-label")
        self.frame1 = tk.Frame(self.master, relief=tk.RAISED, borderwidth=1,height=600)
        self.frame1.pack(side=tk.TOP,fill='x')
        self.frame2 = tk.Frame(self.frame1, relief=tk.RAISED, borderwidth=1,height=450)
        self.frame2.pack(side=tk.TOP,fill='x')
        self.frame3 = tk.Frame(self.frame1, relief=tk.RAISED, borderwidth=5,height=170)
        self.frame3.pack(side=tk.BOTTOM,fill='x')
        self.frame3.pack_propagate(0)
        mybutton = tk.Button(self.frame3, text='Save_data_and_close', command=self.output,width='20',height='5',borderwidth= 1,font=('Arial','10'),bg='#898282')
        self.label = tk.Label( self.frame3, text="Read",bg='green',font=('Arial','16'),width='15',height='3')
        self.label.pack(side=tk.LEFT,fill='x')
        mybutton.pack(side=tk.LEFT,fill='x')
        self.frame4 = tk.Frame(self.frame3, relief=tk.RAISED, borderwidth=5,height=170,width=340)
        self.frame4.pack(side=tk.RIGHT,fill='x')
        self.frame4.grid_propagate(0)
        Label1 = tk.Label( self.frame4, text="Department : ",font=('Arial','15'))
        label2 = tk.Label( self.frame4, text="Statue : ",font=('Arial','15'))
        Label1.grid(column=0, row=0, ipadx=5, pady=15, sticky=tk.W+tk.N)
        label2.grid(column=0, row=1, ipadx=5, pady=15, sticky=tk.W+tk.N)
        self.entryLabel1 = tk.Entry(self.frame4, width=15,bg='#D2D2D2',font=('Arial','15'))
        self.entryLabel2 = tk.Entry(self.frame4, width=15,bg='#D2D2D2',font=('Arial','15'))
        self.entryLabel1.insert(-1, '漁業署')
        self.entryLabel2.insert(-1, '年度正常配給')
        self.entryLabel1.grid(column=1, row=0, ipadx=5, pady=17,ipady=5, sticky=tk.W+tk.N)
        self.entryLabel2.grid(column=1, row=1, ipadx=5, pady=17,ipady=5, sticky=tk.W+tk.N)
class maintable():
    def __init__(self):
        style = ttk.Style()
        style.theme_use('clam')
        self.tree =ttk.Treeview(app.frame2, column=("NUM","Department", "RF-ID","Company","Statue"), show='headings', height=20)
        self.createtable()

    def createtable(self):
        self.tree.column("# 1", anchor=tk.CENTER,width=100)
        self.tree.heading("# 1", text="NUM")
        self.tree.column("# 2", anchor=tk.CENTER,width=100)
        self.tree.heading("# 2", text="Department")
        self.tree.column("# 3", anchor=tk.CENTER,width=200)
        self.tree.heading("# 3", text="RF-ID")
        self.tree.column("# 4", anchor=tk.CENTER,width=100)
        self.tree.heading("# 4", text="Company")
        self.tree.column("# 5", anchor=tk.CENTER,width=200)
        self.tree.heading("# 5", text="Statue")
        self.tree.pack()



his=[]
roost = mainapp()
roost.geometry('700x600')
roost.resizable(width=False, height=False)
app=Application()
#udp = udp_read()
test = maintable()


#p = threading.Thread(target = app.mainloop())
c= threading.Thread(target = roost.udp.while_loop_udp)
#p.start()
c.start()

app.mainloop()
#test.tree.insert('', 'end', text="1", values=("0", "message"))

