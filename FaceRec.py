from tkinter import *
from tkinter import font
import os
from PIL import ImageTk,Image

win=Tk()
win.title("FaceRecog")
win.configure(background="#a1dbcd")
img = ImageTk.PhotoImage(Image.open("single.png"))

def dataset(event):
    os.system('python image_based.py --image sample_computer.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt')
    b1.config(state=DISABLED)
                
def trainr(event):
    os.system('python main_video.py -i 240p1.mp4')
    b2.config(state=DISABLED)

def detect(event):
    os.system('python webcam_code.py')
    b1.config(state=NORMAL)
    b2.config(state=NORMAL)

def close_window():
    win.destroy()
    

f=Frame(win,width=500,height=500)
win.geometry('{}x{}'.format(400,450))

fnt1=font.Font(family='Helvectica', size=15,weight='bold')
Label(f,text="Detection and Classification of motion sensor Images using Machine Learning",font=fnt1).grid(row=1,column=2)
Label(f, image=img).grid(row=2, column=2)

fnt2=font.Font(family='Helvectica', size=10,weight='bold')
b1=Button(f,text="Image Based Object Detection",fg="#31dbcd",bg="#313a39",font=fnt2)
b1.grid(row=3,column=2,padx=10,pady=10,sticky='EWNS')
b1.bind("<Button 1>",dataset)

b2=Button(f,text="Video Based Object Detection",fg="#31dbcd",bg="#383a39",font=fnt2)
b2.grid(row=6,column=2,padx=10,pady=10,sticky='EWNS')
b2.bind("<Button 1>",trainr)

b3=Button(f,text="Real time Object Detection",fg="#31dbcd",bg="#383a39",font=fnt2)
b3.grid(row=9,column=2,padx=10,pady=10,sticky='EWNS')
b3.bind("<Button 1>",detect)

b4=Button(f,text="Exit",fg="#31dbcd",bg="#383a39",command=close_window,font=fnt2)
b4.grid(row=10,column=2,padx=10,pady=10,sticky='EWNS')

f.pack()
win.mainloop()

    
