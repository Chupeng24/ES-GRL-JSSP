


Machine=[
    [1,2,3],
    [2,3,1],
    [1,3,2],
]
#机器在地图中的坐标
Machine_site=[[4,1],[4,2],[4,3]]
Processing_time=[
    [4,7,8],
    [5,2,3],
    [1,3,4]
]
AGV_num=2

import simpy
import tkinter as tk

class Center:
    def __init__(self,Env,x=4,y=5):
        self.x=x
        self.y=y
        self.unit=70
        self.Origin = [1, 2]
        self.split = 5
        self.Hight = self.y - 1  # 4
        self.Width = self.x - 1  # 3
        self.build_JSP()

        self.color=["green","yellow","blue"]

    #start:开始时间，T:加工时间，Machine:所在机器
    def update_Gantt(self,start,T,Machine,Job):
        schdu_unit=20
        PD = [[2, 1], [2, 2], [2, 3]]
        #注：0时刻起点的左上角x坐标：2.7 * self.unit+40
        S_x=4.7*self.unit+30+start*schdu_unit
        S_y=(self.Origin[1] +self.Hight - PD[Machine][1]) * self.unit - 5-20
        E_x=S_x+T*schdu_unit
        E_y = (self.Origin[1] + self.Hight - PD[Machine][1]) * self.unit +5 +20

        self.canvas.create_rectangle(S_x,S_y,E_x,E_y,fill=self.color[Job])
        self.canvas.create_text(S_x+10,S_y+20, text=Job+1
                                , font=("arial", 11), fill="black")
        self.window.update()

    def build_JSP(self):
        self.window = tk.Tk()

        self.window.title("Job shop scheduling simulation")
        self.window.geometry("{1}x{1}".format((self.Hight + 4) * self.unit
                                              , (self.Width + 9) * self.unit))
        self.canvas = tk.Canvas(bg="white", height=(self.Hight + 4) \
                                                   * self.unit, width=(self.Width + 9) * self.unit)

        # Grid Layout
        for c in range(0, (self.Width * self.unit + 1), self.unit):
            x0, y0, x1, y1 = self.Origin[0] * self.unit + c, self.Origin[1] * self.unit \
                , self.Origin[0] * self.unit + c, (self.Hight + self.Origin[1]) \
                             * self.unit
            self.canvas.create_line(x0, y0, x1, y1)

        for r in range(0, (self.Hight * self.unit + 1), self.unit):
            x0, y0, x1, y1 = self \
                                 .Origin[0] * self.unit, self.Origin[1] * self.unit + r \
                , (self.Width + self.Origin[0]) * self.unit \
                , self.Origin[1] * self.unit + r
            self.canvas.create_line(x0, y0, x1, y1)

        # Loading Point
        L = [[0, 2]]
        for i in range(len(L)):
            p1 = [(self.Origin[0] + L[i][0]) * self.unit - 5, (self.Origin[1] + self.Hight - L[i][1]) \
                  * self.unit - 5]
            p2 = [(self.Origin[0] + L[i][0]) * self.unit + 5, (self.Origin[1] + self.Hight - L[i][1]) \
                  * self.unit + 5]
            self.canvas.create_oval(p1[0], p1[1], p2[0], p2[1], fill="blue")
        # Unloading Point
        U = [[0, 1]]
        for i in range(len(U)):
            p1 = [(self.Origin[0] + U[i][0]) * self.unit - 5, (self.Origin[1] + self.Hight - U[i][1]) \
                  * self.unit - 5]
            p2 = [(self.Origin[0] + U[i][0]) * self.unit + 5, (self.Origin[1] + self.Hight - U[i][1]) \
                  * self.unit + 5]
            self.canvas.create_oval(p1[0], p1[1], p2[0], p2[1], fill="green")
        # P/D Point
        PD = [[2,1],[2,2],[2,3]]
        Machine_name=["M1","M2","M3"]
        #scheduling的布局
        self.canvas.create_rectangle( 4.1* self.unit,  2* self.unit, 11* self.unit,  6* self.unit
                                      , fill="white")
        for i in range(len(PD)):
            p1 = [(self.Origin[0] + PD[i][0]) * self.unit - 5, (self.Origin[1]
                                                                + self.Hight - PD[i][1]) * self.unit - 5]
            p2 = [(self.Origin[0] + PD[i][0]) * self.unit + 5, (self.Origin[1]
                                                                + self.Hight - PD[i][1]) * self.unit + 5]

            self.canvas.create_rectangle(p1[0]-20, p1[1]-20, p2[0]+20, p2[1]+20
                                         , fill="orange")
            self.canvas.create_text(p1[0]+5, p1[1]+5, text=Machine_name[i]
                                    , font=("arial", 12), fill="black")
            self.canvas.create_rectangle(p1[0] - 20+1.7*self.unit, p1[1] - 20, p2[0] + 20+1.7*self.unit, p2[1] + 20
                                         , fill="red")
            self.canvas.create_text(p1[0] + 5+1.7*self.unit, p1[1] + 5, text=Machine_name[i]
                                    , font=("arial", 12), fill="black")
        # AS/RS]
        p = [0.5 * self.unit, 4.5 * self.unit]
        p1 = [0.1* self.unit, 5.5* self.unit]
        p2 = [1* self.unit,3.5* self.unit]
        self.canvas.create_rectangle(p1[0], p1[1], p2[0], p2[1]
                                     , fill="yellow")
        self.canvas.create_text(p[0], p[1], text="AS/RS"
                                , font=("arial", 12), fill="black")

        self.canvas.create_rectangle(0* self.unit
                                     , 0* self.unit, 1.6* self.unit
                                     , 0.8* self.unit
                                     , fill="gray")
        self.time = self.canvas.create_text(0.8 * self.unit
                                            , 0.4 * self.unit, text="00:00"
                                            , font=("arial", 20)
                                            , fill="Blue")
        self.canvas.create_text(2.5*self.unit, 1.5*self.unit, text="SIMULATION SIDE"
                                , font=("arial", 12), fill="black")
        self.canvas.create_text(7.5 * self.unit, 1.5 * self.unit, text="SCHEDULING SIDE"
                                , font=("arial", 12), fill="black")
        self.canvas.pack()
        # self.window.mainloop()

Env=simpy.Environment()
c=Center(Env)
for i in range(100):
    c.update_Gantt(0,4,0,0)
for i in range(100):
    c.update_Gantt(4, 5, 1, 0)
for i in range(100):
    c.update_Gantt(0, 3, 1, 1)
for i in range(1000):
    c.update_Gantt(3,5,2,1)