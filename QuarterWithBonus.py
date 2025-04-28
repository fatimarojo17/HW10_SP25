#region imports
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import math
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
#endregion

#region class definitions
#region specialized graphic items
class MassBlock(qtw.QGraphicsItem):
    def __init__(self, CenterX, CenterY, width=30, height=10, parent=None, pen=None, brush=None, name='CarBody', mass=10):
        super().__init__(parent)
        self.x = CenterX
        self.y = CenterY
        self.pen = pen
        self.brush = brush
        self.width = width
        self.height = height
        self.rect = qtc.QRectF(-self.width/2, -self.height/2, self.width, self.height)
        self.name = name
        self.mass = mass
        self.setToolTip(f"{self.name}\nx={self.x:.3f}, y={self.y:.3f}\nmass = {self.mass:.3f}")
        self.setTransformOriginPoint(self.x, self.y)

    def boundingRect(self):
        return self.rect

    def paint(self, painter, option, widget=None):
        if self.pen:
            painter.setPen(self.pen)
        if self.brush:
            painter.setBrush(self.brush)
        painter.drawRect(self.rect)
        self.setPos(self.x, self.y)

class Wheel(qtw.QGraphicsItem):
    def __init__(self, CenterX, CenterY, radius=10, parent=None, pen=None, wheelBrush=None, massBrush=None, name='Wheel', mass=10):
        super().__init__(parent)
        self.x = CenterX
        self.y = CenterY
        self.pen = pen
        self.brush = wheelBrush
        self.radius = radius
        self.massBlock = MassBlock(0, 0, width=2*radius*0.85, height=radius/3, pen=pen, brush=massBrush, name="Wheel Mass", mass=mass)
        self.name = name
        self.mass = mass
        self.setToolTip(f"{self.name}\nx={self.x:.3f}, y={self.y:.3f}\nmass = {self.mass:.3f}")

    def boundingRect(self):
        return qtc.QRectF(-self.radius, -self.radius, 2*self.radius, 2*self.radius)

    def paint(self, painter, option, widget=None):
        if self.pen:
            painter.setPen(self.pen)
        if self.brush:
            painter.setBrush(self.brush)
        painter.drawEllipse(self.boundingRect())
        self.setPos(self.x, self.y)

    def addToScene(self, scene):
        scene.addItem(self)
        self.massBlock.setParentItem(self)
        scene.addItem(self.massBlock)

# Spring Class
class Spring(qtw.QGraphicsItem):
    def __init__(self, x1, y1, x2, y2, parent=None, pen=None):
        super().__init__(parent)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.pen = pen

    def boundingRect(self):
        return qtc.QRectF(min(self.x1, self.x2), min(self.y1, self.y2), abs(self.x2 - self.x1), abs(self.y2 - self.y1))

    def paint(self, painter, option, widget=None):
        if self.pen:
            painter.setPen(self.pen)
        # Draw a zig-zag spring
        num_coils = 6
        points = []
        dx = (self.x2 - self.x1) / (num_coils * 2)
        dy = (self.y2 - self.y1) / (num_coils)
        x = self.x1
        y = self.y1
        points.append(qtc.QPointF(x, y))
        for _ in range(num_coils):
            x += dx
            y += dy
            points.append(qtc.QPointF(x, y))
            x += dx
            y -= dy
            points.append(qtc.QPointF(x, y))
        points.append(qtc.QPointF(self.x2, self.y2))
        painter.drawPolyline(*points)

# Dashpot Class
class Dashpot(qtw.QGraphicsItem):
    def __init__(self, x1, y1, x2, y2, parent=None, pen=None):
        super().__init__(parent)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.pen = pen

    def boundingRect(self):
        return qtc.QRectF(min(self.x1, self.x2), min(self.y1, self.y2), abs(self.x2 - self.x1), abs(self.y2 - self.y1))

    def paint(self, painter, option, widget=None):
        if self.pen:
            painter.setPen(self.pen)
        midx = (self.x1 + self.x2) / 2
        midy = (self.y1 + self.y2) / 2
        rect_width = 10
        rect_height = 20
        # Top line
        painter.drawLine(self.x1, self.y1, midx, midy - rect_height/2)
        # Rectangle (damper body)
        painter.drawRect(midx - rect_width/2, midy - rect_height/2, rect_width, rect_height)
        # Bottom line
        painter.drawLine(midx, midy + rect_height/2, self.x2, self.y2)

#region MVC for quarter car model
class CarModel():
    """
    I re-wrote the quarter car model as an object oriented program
    and used the MVC pattern.  This is the quarter car model.  It just
    stores information about the car and results of the ode calculation.
    """
    def __init__(self):
        """
        self.results to hold results of odeint solution
        self.t time vector for odeint and for plotting
        self.tramp is time required to climb the ramp
        self.angrad is the ramp angle in radians
        self.ymag is the ramp height in m
        """
        self.results = []
        self.tmax = 3.0  # limit of timespan for simulation in seconds
        self.t = np.linspace(0, self.tmax, 200)
        self.tramp = 1.0  # time to traverse the ramp in seconds
        self.angrad = 0.1
        self.ymag = 6.0 / (12 * 3.3)  # ramp height in meters.  default is 0.1515 m
        self.yangdeg = 45.0  # ramp angle in degrees.  default is 45
        self.results = None

        #set default values for the properties of the quarter car model
        self.m1 = 450  # kg
        self.m2 = 20  # kg
        self.c1 = 4500  # Ns/m
        self.k1 = 15000  # N/m
        self.k2 = 90000  # N/m
        self.v = 120  # km/h

        g = 9.81
        self.mink1 = self.m1 * g / 0.1524
        self.maxk1 = self.m1 * g / 0.0762
        self.mink2 = self.m2 * g / 0.0381
        self.maxk2 = self.m2 * g / 0.01905

        self.accel = None
        self.accelMax = 0
        self.accelLim = 2.0
        self.SSE = 0.0


class CarView():
    def __init__(self, args):
        self.input_widgets, self.display_widgets = args
        # unpack widgets with same names as they have on the GUI
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, \
         self.le_tmax, self.chk_IncludeAccel = self.input_widgets

        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, \
        self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets

        # creating a canvas to draw a figure for the car model
        self.figure = Figure(tight_layout=True, frameon=True, facecolor='none')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout_horizontal_main.addWidget(self.canvas)

        # axes for the plotting using view
        self.ax = self.figure.add_subplot()
        if self.ax is not None:
            self.ax1 = self.ax.twinx()

        self.buildScene()

    def updateView(self, model=None):
        self.le_m1.setText("{:0.2f}".format(model.m1))
        self.le_k1.setText("{:0.2f}".format(model.k1))
        self.le_c1.setText("{:0.2f}".format(model.c1))
        self.le_m2.setText("{:0.2f}".format(model.m2))
        self.le_k2.setText("{:0.2f}".format(model.k2))
        self.le_ang.setText("{:0.2f}".format(model.yangdeg))
        self.le_tmax.setText("{:0.2f}".format(model.tmax))
        stTmp="k1_min = {:0.2f}, k1_max = {:0.2f}\nk2_min = {:0.2f}, k2_max = {:0.2f}\n".format(model.mink1, model.maxk1, model.mink2, model.maxk2)
        stTmp+="SSE = {:0.2f}".format(model.SSE)
        self.lbl_MaxMinInfo.setText(stTmp)
        self.doPlot(model)

    def buildScene(self):
        self.scene = qtw.QGraphicsScene()
        self.scene.setObjectName("MyScene")
        self.scene.setSceneRect(-200, -200, 400, 400)

        self.gv_Schematic.setScene(self.scene)
        self.setupPensAndBrushes()

        # Create the CarBody and Wheel
        self.Wheel = Wheel(0, 50, 50, pen=self.penWheel, wheelBrush=self.brushWheel, massBrush=self.brushMass,
                           name="Wheel")
        self.CarBody = MassBlock(0, -70, 100, 30, pen=self.penWheel, brush=self.brushMass, name="Car Body", mass=150)

        # New: Spring and Dashpot between CarBody and Wheel
        self.Spring1 = Spring(0, -40, 0, 20, pen=self.penWheel)  # Slightly left side
        self.Dashpot1 = Dashpot(30, -40, 30, 20, pen=self.penWheel)  # Slightly right side

        # Add to scene
        self.Wheel.addToScene(self.scene)
        self.scene.addItem(self.CarBody)
        self.scene.addItem(self.Spring1)
        self.scene.addItem(self.Dashpot1)

    def setupPensAndBrushes(self):
        self.penWheel = qtg.QPen(qtg.QColor("orange"))
        self.penWheel.setWidth(1)
        self.brushWheel = qtg.QBrush(qtg.QColor.fromHsv(35,255,255, 64))
        self.brushMass = qtg.QBrush(qtg.QColor(200,200,200, 128))

    def doPlot(self, model=None):
        if model.results is None:
            return
        ax = self.ax
        ax1=self.ax1
        # plot result of odeint solver
        QTPlotting = True  # assumes we are plotting onto a QT GUI form
        if ax == None:
            ax = plt.subplot()
            ax1=ax.twinx()
            QTPlotting = False  # actually, we are just using CLI and showing the plot
        ax.clear()
        ax1.clear()
        t=model.timeData
        ycar = model.results[:,0]
        ywheel=model.results[:,2]
        accel=model.accelData

        if self.chk_LogX.isChecked():
            ax.set_xlim(0.001,model.tmax)
            ax.set_xscale('log')
        else:
            ax.set_xlim(0.0, model.tmax)
            ax.set_xscale('linear')

        if self.chk_LogY.isChecked():
            ax.set_ylim(0.0001,max(ycar.max(), ywheel.max()*1.05))
            ax.set_yscale('log')
        else:
            ax.set_ylim(0.0, max(ycar.max(), ywheel.max()*1.05))
            ax.set_yscale('linear')

        ax.plot(t, ycar, 'b-', label='Body Position')
        ax.plot(t, ywheel, 'r-', label='Wheel Position')
        if self.chk_ShowAccel.isChecked():
            ax1.plot(t, accel, 'g-', label='Body Accel')
            ax1.axhline(y=accel.max(), color='orange')  # horizontal line at accel.max()
            ax1.set_yscale('log' if self.chk_LogAccel.isChecked() else 'linear')

        # add axis labels
        ax.set_ylabel("Vertical Position (m)", fontsize='large' if QTPlotting else 'medium')
        ax.set_xlabel("time (s)", fontsize='large' if QTPlotting else 'medium')
        ax1.set_ylabel("Y'' (g)", fontsize = 'large' if QTPlotting else 'medium')
        ax.legend()

        ax.axvline(x=model.tramp)  # vertical line at tramp
        ax.axhline(y=model.ymag)  # horizontal line at ymag
        # modify the tick marks
        ax.tick_params(axis='both', which='both', direction='in', top=True,
                       labelsize='large' if QTPlotting else 'medium')  # format tick marks
        ax1.tick_params(axis='both', which='both', direction='in', right=True,
                       labelsize='large' if QTPlotting else 'medium')  # format tick marks
        # show the plot
        if QTPlotting == False:
            plt.show()
        else:
            self.canvas.draw()

class CarController():
    def __init__(self, args):
        """
        This is the controller I am using for the quarter car model.
        """
        self.input_widgets, self.display_widgets = args
        #unpack widgets with same names as they have on the GUI
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, \
         self.le_tmax, self.chk_IncludeAccel = self.input_widgets

        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, \
        self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets

        self.model = CarModel()
        self.view = CarView(args)

        self.chk_IncludeAccel=qtw.QCheckBox()

    def ode_system(self, X, t):
        # define the forcing function equation for the linear ramp
        # It takes self.tramp time to climb the ramp, so y position is
        # a linear function of time.
        if t < self.model.tramp:
            y = self.model.ymag * (t / self.model.tramp)
        else:
            y = self.model.ymag

        x1, x1dot, x2, x2dot = X
        m1, m2 = self.model.m1, self.model.m2
        c1, k1, k2 = self.model.c1, self.model.k1, self.model.k2

        x1ddot = (-k1 * (x1 - x2) - c1 * (x1dot - x2dot)) / m1
        x2ddot = (k1 * (x1 - x2) + c1 * (x1dot - x2dot) - k2 * (x2 - y)) / m2

        # return the derivatives of the input state vector
        return [x1dot, x1ddot, x2dot, x2ddot]

    def calculate(self, doCalc=True):
        """
        I will first set the basic properties of the car model and then calculate the result
        in another function doCalc.
        """
        #Step 1.  Read from the widgets
        self.model.m1 = float(self.le_m1.text())
        self.model.m2 = float(self.le_m2.text())
        self.model.c1 = float(self.le_c1.text())
        self.model.k1 = float(self.le_k1.text())
        self.model.k2 = float(self.le_k2.text())
        self.model.v = float(self.le_v.text())

        g = 9.81
        self.model.mink1 = self.model.m1 * g / 0.1524
        self.model.maxk1 = self.model.m1 * g / 0.0762
        self.model.mink2 = self.model.m2 * g / 0.0381
        self.model.maxk2 = self.model.m2 * g / 0.01905

        ymag=6.0/(12.0*3.3)   #This is the height of the ramp in m
        if ymag is not None:
            self.model.ymag = ymag
        self.model.yangdeg = float(self.le_ang.text())
        self.model.tmax = float(self.le_tmax.text())
        if(doCalc):
            self.doCalc()
        self.SSE((self.model.k1, self.model.c1, self.model.k2), optimizing=False)
        self.view.updateView(self.model)

    def setWidgets(self, w):
        self.view.setWidgets(w)
        self.chk_IncludeAccel=self.view.chk_IncludeAccel

    def doCalc(self, doPlot=True, doAccel=True):
        """
        This solves the differential equations for the quarter car model.
        :param doPlot:
        :param doAccel:
        :return:
        """
        v = 1000 * self.model.v / 3600  # convert speed to m/s from kph
        self.model.angrad = self.model.yangdeg * math.pi / 180.0  # convert angle to radians
        self.model.tramp = self.model.ymag / (math.sin(self.model.angrad) * v)  # calculate time to traverse ramp

        self.model.t = np.linspace(0, self.model.tmax, 2000)
        ic = [0, 0, 0, 0]
        # run odeint solver
        self.model.results = odeint(self.ode_system, ic, self.model.t)

        if doAccel:
            self.calcAccel()
            self.model.timeData = self.model.t
            self.model.accelData = self.model.accel

        if doPlot:
            self.doPlot()

    def calcAccel(self):
        """
        Calculate the acceleration in the vertical direction using the forward difference formula.
        """
        N=len(self.model.t)
        self.model.accel=np.zeros(shape=N)
        vel=self.model.results[:,1]
        for i in range(N):
            if i==N-1:
                h = self.model.t[i] - self.model.t[i-1]
                self.model.accel[i]=(vel[i]-vel[i-1])/(9.81*h)  # backward difference of velocity
            else:
                h = self.model.t[i + 1] - self.model.t[i]
                self.model.accel[i] = (vel[i + 1] - vel[i]) / (9.81 * h)  # forward difference of velocity
            # else:
            #     self.model.accel[i]=(vel[i+1]-vel[i-1])/(9.81*2.0*h)  # central difference of velocity
        self.model.accelMax=self.model.accel.max()
        return True

    def OptimizeSuspension(self):
        """
        Step 1:  set parameters based on GUI inputs by calling self.set(doCalc=False)
        Step 2:  make an initial guess for k1, c1, k2
        Step 3:  optimize the suspension
        :return:
        """
        #Step 1:
        self.calculate(doCalc=False)
        x0 = np.array([self.model.k1, self.model.c1, self.model.k2])
        answer = minimize(self.SSE, x0, method='Nelder-Mead')
        self.model.k1, self.model.c1, self.model.k2 = answer.x
        self.doCalc()
        self.view.updateView(self.model)

    def SSE(self, vals, optimizing=True):
        """
        Calculates the sum of square errors between the contour of the road and the car body.
        :param vals:
        :param optimizing:
        :return:
        """
        k1, c1, k2=vals  #unpack the new values for k1, c1, k2
        self.model.k1=k1
        self.model.c1=c1
        self.model.k2=k2
        self.doCalc(doPlot=False)  #solve the odesystem with the new values of k1, c1, k2
        SSE=0
        for i in range(len(self.model.results[:,0])):
            t=self.model.t[i]
            y=self.model.results[:,0][i]
            if t < self.model.tramp:
                ytarget = self.model.ymag * (t / self.model.tramp)
            else:
                ytarget = self.model.ymag
            SSE+=(y-ytarget)**2

        #some penalty functions if the constants are too small
        if optimizing:
            if k1<self.model.mink1 or k1>self.model.maxk1:
                SSE+=100
            if c1<10:
                SSE+=100
            if k2<self.model.mink2 or k2>self.model.maxk2:
                SSE+=100

            # I'm overlaying a gradient in the acceleration limit that scales with distance from a target squared.
            if self.model.accelMax>self.model.accelLim and self.chk_IncludeAccel.isChecked():
                # need to soften suspension
                SSE+=(self.model.accelMax-self.model.accelLim)**2
        self.model.SSE=SSE
        return SSE

    def doPlot(self):
        self.view.doPlot(self.model)
#endregion
#endregion

def main():
    QCM = CarController()
    QCM.doCalc()

if __name__ == '__main__':
    main()
