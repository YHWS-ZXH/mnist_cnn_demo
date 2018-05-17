import sys
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
import cv2
import copy
import numpy as np
import tensorflow as tf
import test
import test

def imageprepare(file_name):
    img=cv2.imread(file_name)
    imgwith,imgheight=img.shape[0],img.shape[1]
    ret=[0.0 for i in range(imgwith*imgheight)]
    for i in range(imgwith):
        for j in range(imgheight):
            ret[j*imgwith+i]=(255-img[j][i][0])*1.0/255.0 
    return ret
    
class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.mnist=test.mnist_test()
        self.pixel=28
        self.blocks=[[0 for i in range(self.pixel)] for j in range(self.pixel)]
        #self.blocks[0][1]=1
        self.pos_x = 0
        self.pos_y = 0
        
        self.statusLabel = QLabel("描述")
        self.statusLabel.move(380,100)
        
        self.initUI()

    def initUI(self):
        self.setGeometry(10, 100, 480, 320)
        self.setWindowTitle("Points")

        self.btn1 = QPushButton("Button 1", self)
        self.btn1.move(400, 50)

        self.btn2 = QPushButton("Button 2", self)
        self.btn2.move(400, 70)

        # create textbox
        #self.textbox = QLineEdit(self)
        #self.textbox.move(400, 90)
        #self.textbox.resize(50, 20)

        self.label = QLabel(self)
        self.label.setFixedWidth(400)
        self.label.setFixedHeight(400)
        self.label.move(200, 100)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setText(u'数字是')


        self.btn1.clicked.connect(self.get_picture)
        self.btn2.clicked.connect(self.clear)

        self.show()

    def paintEvent(self, e):
        qp =QPainter()
        qp.begin(self)
        self.drawblock(qp)
        self.drawline(qp)
        qp.end()

    def drawline(self, qp):
        qp.setPen(Qt.white)
        size = self.size()
        
        if size.width()>size.height():
            kuan=size.height()
        else:
            kuan=size.width()
        for i in range(1,self.pixel+1):
            x=int(i*kuan/self.pixel)
            y=int(i*kuan/self.pixel)
            qp.drawLine(0, y, kuan, y)
            qp.drawLine(x, 0, x, kuan)
        
        x=int(1.0*kuan/10+kuan)
        qp.drawLine(x, 0, x, kuan)
        for i in range(10):
            y=int(i*kuan/10)
            #qp.drawLine(kuan, y, x, y)

    def drawblock(self, qp):
        qp.setPen(Qt.white)
        size = self.size()
        qp.setBrush(QColor(0, 0, 0))
        
        if size.width()>size.height():
            kuan=size.height()
        else:
            kuan=size.width()
        
        for i in range(self.pixel):
            for j in range(self.pixel):
                if self.blocks[i][j]==1:
                    qp.drawRect(i*kuan/self.pixel, j*kuan/self.pixel, kuan/self.pixel, kuan/self.pixel)
        
        data=[0.0 for i in range(self.pixel*self.pixel)]
        for y in range(self.pixel):
            for x in range(self.pixel):
                if self.blocks[x][y]==1:
                    data[x+y*self.pixel]=1.0
                else:
                    data[x+y*self.pixel]=0.0
        result=self.mnist.classify(data)
        for i in range(10):
            cvalue=int(255.0-result[i]*255.0)
            #qp.setBrush(QColor(cvalue,cvalue,cvalue))
            qp.setBrush(QColor(0,0,0))
            #qp.drawRect(kuan,i*kuan/10, kuan/10, kuan/10)
            qp.drawRect(kuan,i*kuan/10, (255-cvalue)/2, kuan/10)
    def mouseMoveEvent(self, event):
        
        self.pos_x = event.pos().x()
        self.pos_y = event.pos().y()
        self.blocks_set(self.pos_x,self.pos_y)

    def mousePressEvent(self,event):
        self.pos_x = event.pos().x()
        self.pos_y = event.pos().y()
        if event.button() == Qt.LeftButton:
            self.blocks_set(self.pos_x,self.pos_y)
        elif event.button() == Qt.RightButton:
            self.blocks_set(self.pos_x,self.pos_y,0)
        else:
            self.blocks_set(self.pos_x,self.pos_y,1)
        #self.get_picture()
    def blocks_set(self,pos_x,pos_y,value=1):
        size = self.size()
        if size.width()>size.height():
            kuan=size.height()
        else:
            kuan=size.width()
        if pos_x<0:
            tx=0
        elif pos_x>=kuan:
            tx=self.pixel-1
        else:
            tx=int(pos_x*self.pixel/kuan)
        if pos_y<0:
            ty=0
        elif pos_y>=kuan:
            ty=self.pixel-1
        else:
            ty=int(pos_y*self.pixel/kuan)
        self.blocks[tx][ty]=value
        self.update()


    def get_picture(self):
        a = copy.deepcopy(self.blocks)
        b = np.reshape(a,784,order ='F')
        print(b)
        self.max_idx = np.argmax(self.mnist.classify(b))
        #print('#### predict number is: %s' % (self.max_idx))
        self.label.setText(u'数字是%s'% (self.max_idx))

    def clear(self):
        self.blocks = [[0 for i in range(self.pixel)] for j in range(self.pixel)]
        self.label.setText(u'数字是')
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())