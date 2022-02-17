from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAbstractButton

from PyQt5.QtWidgets import (
    QAbstractButton,
    QApplication,
    QHBoxLayout,
    QSizePolicy,
    QWidget,
)

class SwitchButton(QAbstractButton):
    def __init__(self, parent=None , w1='On' , l1=12 , w2='Off', l2=33 , width=60 , height = 25):
        super().__init__(parent)
        self.onT = w1
        self.offT = w2
        self.onX = l1
        self.offX = l2
        self.setCheckable(True)
        # self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        # self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(width, height)
        self.__labeloff = QtWidgets.QLabel(self)
        self.__labeloff.setText(self.offT)
        self.__labeloff.setStyleSheet('color: rgb(120, 120, 120); font-weight: bold;')
        self.__background  = Background(self)
        self.__labelon = QtWidgets.QLabel(self)
        self.__labelon.setText(self.onT)
        self.__labelon.setStyleSheet('color: rgb(255, 255, 255); font-weight: bold;')
        self.__circle   = Circle(self)

        self.__background.resize(20, 20)
        self.__labelon.move(self.onX , height/2 - 6)
        self.__labeloff.move(self.offX , height/2 - 6)
        self.__duration    = 100
        

    def setDuration(self, time):
        self.__duration = time

    def setChecked(self, checked):
        super().setChecked(checked)

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def paintEvent(self, event):
        s = self.size()
        w = s.width()
        h = s.height()

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen = QtGui.QPen(QtCore.Qt.NoPen)
        p.setPen(pen)
        p.setBrush(QtGui.QColor(120, 120, 120))
        p.drawRoundedRect(0, 0, w , h, h/2 , h/2)
        lg = QtGui.QLinearGradient(35, 30, 35, 0)
        lg.setColorAt(0, QtGui.QColor(210, 210, 210, 255))
        lg.setColorAt(0.25, QtGui.QColor(255, 255, 255, 255))
        lg.setColorAt(0.82, QtGui.QColor(255, 255, 255, 255))
        lg.setColorAt(1, QtGui.QColor(210, 210, 210, 255))
        p.setBrush(lg)
        ed = 1
        p.drawRoundedRect(ed , ed , w - 2*ed , h - 2*ed, h/2 - ed , h/2 - ed)

        p.setBrush(QtGui.QColor(210, 210, 210))
        ed = 2
        p.drawRoundedRect(ed , ed , w - 2*ed , h - 2*ed, h/2 - ed , h/2 - ed)

        ed = 3
        if self.isEnabled():
            lg = QtGui.QLinearGradient(50, 30, 35, 0)
            lg.setColorAt(0, QtGui.QColor(230, 230, 230, 255))
            lg.setColorAt(0.25, QtGui.QColor(255, 255, 255, 255))
            lg.setColorAt(0.82, QtGui.QColor(255, 255, 255, 255))
            lg.setColorAt(1, QtGui.QColor(230, 230, 230, 255))
            p.setBrush(lg)
            p.drawRoundedRect(ed , ed , w - 2*ed , h - 2*ed, h/2 - ed , h/2 - ed)
        else:
            lg = QtGui.QLinearGradient(50, 30, 35, 0)
            lg.setColorAt(0, QtGui.QColor(200, 200, 200, 255))
            lg.setColorAt(0.25, QtGui.QColor(230, 230, 230, 255))
            lg.setColorAt(0.82, QtGui.QColor(230, 230, 230, 255))
            lg.setColorAt(1, QtGui.QColor(200, 200, 200, 255))
            p.setBrush(lg)
            p.drawRoundedRect(ed , ed , w - 2*ed , h - 2*ed, h/2 - ed , h/2 - ed)

    def mouseReleaseEvent(self, event):  # pylint: disable=invalid-name
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
            self.__circlemove = QtCore.QPropertyAnimation(self.__circle, b"pos")
            self.__circlemove.setDuration(self.__duration)

            self.__ellipsemove = QtCore.QPropertyAnimation(self.__background, b"size")
            self.__ellipsemove.setDuration(self.__duration)
          
            y =0
            xs = 0
            xf = self.width() - self.height() 
            Ssize = QtCore.QSize(self.height()-4, self.height()-4)
            Fsize = QtCore.QSize(self.width()-4, self.height()-4)
            if self.isChecked():
                xi = xs
                xf = xf
                si = Ssize
                sf = Fsize
            else:
                xi = xf
                xf = xs
                si = Fsize
                sf = Ssize

            self.__circlemove.setStartValue(QtCore.QPoint(xi, y))
            self.__circlemove.setEndValue(QtCore.QPoint(xf, y))

            self.__ellipsemove.setStartValue(si)
            self.__ellipsemove.setEndValue(sf)

            self.__circlemove.start()
            self.__ellipsemove.start()

    def enterEvent(self, event):  # pylint: disable=invalid-name
        self.setCursor(Qt.PointingHandCursor)
        super().enterEvent(event)

class Circle(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Circle, self).__init__(parent)
        self.ph = parent.height()
        self.pw = parent.width()
        self.setFixedSize(self.ph, self.ph)
        self.parent = parent


    def paintEvent(self, event):
        if self.parent.isChecked():
            self.move(self.pw - self.ph , 0)

        s = self.size()
        w = s.width()
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(QtGui.QColor(120, 120, 120))
        ed = 2
        qp.drawEllipse(ed , ed , w - 2*ed , w - 2*ed)
        rg = QtGui.QRadialGradient(int(self.width() / 2), int(self.height() / 2), 12)
        rg.setColorAt(0, QtGui.QColor(255, 255, 255))
        rg.setColorAt(0.6, QtGui.QColor(255, 255, 255))
        rg.setColorAt(1, QtGui.QColor(205, 205, 205))
        qp.setBrush(QtGui.QBrush(rg))
        ed = 3
        qp.drawEllipse(ed , ed , w - 2*ed , w - 2*ed)

        qp.setBrush(QtGui.QColor(210, 210, 210))
        ed = 4
        qp.drawEllipse(ed , ed , w - 2*ed , w - 2*ed)

        ed = 5
        if self.isEnabled():
            lg = QtGui.QLinearGradient(3, 18,20, 4)
            lg.setColorAt(0, QtGui.QColor(255, 255, 255, 255))
            lg.setColorAt(0.55, QtGui.QColor(230, 230, 230, 255))
            lg.setColorAt(0.72, QtGui.QColor(255, 255, 255, 255))
            lg.setColorAt(1, QtGui.QColor(255, 255, 255, 255))
            qp.setBrush(lg)
            qp.drawEllipse(ed , ed , w - 2*ed , w - 2*ed)
        else:
            lg = QtGui.QLinearGradient(3, 18, 20, 4)
            lg.setColorAt(0, QtGui.QColor(230, 230, 230))
            lg.setColorAt(0.55, QtGui.QColor(210, 210, 210))
            lg.setColorAt(0.72, QtGui.QColor(230, 230, 230))
            lg.setColorAt(1, QtGui.QColor(230, 230, 230))
            qp.setBrush(lg)
            qp.drawEllipse(ed , ed , w - 2*ed , w - 2*ed)

class Background(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Background, self).__init__(parent)
        self.ph = parent.height()
        self.pw = parent.width()
        self.setFixedHeight(self.ph)
        self.parent = parent
        

    def paintEvent(self, event):
        if self.parent.isChecked():
            self.setGeometry(0 , 0 , self.pw , self.height())
        s = self.size()
        w = s.width()
        h = s.height()
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen = QtGui.QPen(QtCore.Qt.NoPen)
        qp.setPen(pen)
        qp.setBrush(QtGui.QColor(154,205,50))
        ed = 2
        if self.isEnabled():
            qp.setBrush(QtGui.QColor(154, 190, 50))
            qp.drawRoundedRect(ed, ed, w - 2*ed , h - 2*ed , h/2 - ed , h/2 - ed)

            lg = QtGui.QLinearGradient(0, 25, 70, 0)
            lg.setColorAt(0, QtGui.QColor(154, 184, 50))
            lg.setColorAt(0.35, QtGui.QColor(154, 210, 50))
            lg.setColorAt(0.85, QtGui.QColor(154, 184, 50))
            qp.setBrush(lg)
            ed = 3
            qp.drawRoundedRect(ed, ed, w - 2*ed , h - 2*ed , h/2 - ed , h/2 - ed)
        else:
            qp.setBrush(QtGui.QColor(150, 150, 150))
            qp.drawRoundedRect(ed, ed, w - 2*ed , h - 2*ed , h/2 - ed , h/2 - ed)

            lg = QtGui.QLinearGradient(5, 25, 60, 0)
            lg.setColorAt(0, QtGui.QColor(190, 190, 190))
            lg.setColorAt(0.35, QtGui.QColor(230, 230, 230))
            lg.setColorAt(0.85, QtGui.QColor(190, 190, 190))
            qp.setBrush(lg)
            ed = 3
            qp.drawRoundedRect(ed, ed, w - 2*ed , h - 2*ed , h/2 - ed , h/2 - ed)

