import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm
from fcns.veeMap import veeMap

class plot_results:
    def __init__(self):
        self.tList = []
        self.XtList = [[] for _ in range(12)]
        self.XdList = [[] for _ in range(12)]
        self.UtList = [[] for _ in range(12)]
        self.UList = [[] for _ in range(12)]
        self.FPtList = [[] for _ in range(12)]
        self.FPdList = [[] for _ in range(12)]
    
    def update(self, t, Xt, Xd, Ut, U):
        # Rotation matrix to euler angle
        COMAng_t = Xt[6:15].reshape((3, 3), order='F')
        COMAng_t = veeMap(logm(COMAng_t))
        # COMAng_t = np.array([0,1,2]) # Test
        Xt_res = np.concatenate((Xt[0:6], COMAng_t, Xt[15:18]))

        COMAng_d = Xd[6:15].reshape((3, 3), order='F')
        COMAng_d = veeMap(logm(COMAng_d))
        # COMAng_d = COMAng_t + 1 # Test
        Xd_res = np.concatenate((Xd[0:6], COMAng_d, Xd[15:18]))

        self.tList.append(t)
        for i in range(len(self.XtList)):
            self.XtList[i].append(Xt_res[i])
            self.XdList[i].append(Xd_res[i])
        for i in range(len(self.UtList)):
            self.UtList[i].append(Ut[i])
            self.UList[i].append(U[i])
        for i in range(len(self.FPtList)):
            self.FPtList[i].append(Xt[i+18])
            self.FPdList[i].append(Xd[i+18])
    
    def result_plot(self):
        Suptitles = ["Ground Reaction Force (GRF) [N]", "COM Informations", "Footend Position [m]"]
        COM_subtitles = ["Lin Pos [m]", "Lin Vel [m/s]", "Ang Pos [rad]", "Ang Vel [rad/s]"]
        Dim_Name = ["X", "Y", "Z"]
        Leg_Name = ["Front Left", "Front Right", "Rear Left", "Rear Right"]

        plt.figure(1) # Diagramm for GRF (desired/real)
        fig1, axs1 = plt.subplots(3, 4)
        fig1.suptitle(Suptitles[0], fontweight="bold", size=30)
        for i in range(3):
            for j in range(4):
                l1, = axs1[i,j].plot(self.tList, self.UtList[i+j*3], color="green", label='desired')
                l2, = axs1[i,j].plot(self.tList, self.UList[i+j*3], color="blue", label='real')
                axs1[i,j].grid()
                if i == 0:
                    axs1[i,j].set_title(Leg_Name[j], size=20)
                if i == 2:
                    axs1[i,j].set_xlabel('Time [s]', size=20)
                if j == 0:
                    ylabel_Name = Dim_Name[i] + "-axis"
                    axs1[i,j].set_ylabel(ylabel_Name, size=20)
                axs1[i,j].xaxis.set_tick_params(labelsize=15)
                axs1[i,j].yaxis.set_tick_params(labelsize=15)
        fig1.legend([l1, l2], ['desired', 'real'], bbox_to_anchor=(0.75,0.99), prop={'size': 20})
        
        plt.figure(2) # Diagramm for COM Information (desired/real)
        fig2, axs2 = plt.subplots(3, 4)
        fig2.suptitle(Suptitles[1], fontweight="bold", size=30)
        for i in range(3):
            for j in range(4):
                l1, = axs2[i,j].plot(self.tList, self.XdList[i+j*3], color="green", label='desired')
                l2, = axs2[i,j].plot(self.tList, self.XtList[i+j*3], color="blue", label='real')
                axs2[i,j].grid()
                if i == 0:
                    axs2[i,j].set_title(COM_subtitles[j], size=20)
                if i == 2:
                    axs2[i,j].set_xlabel('Time [s]', size=20)
                if j == 0:
                    ylabel_Name = Dim_Name[i] + "-axis"
                    axs2[i,j].set_ylabel(ylabel_Name, size=20)
                axs2[i,j].xaxis.set_tick_params(labelsize=15)
                axs2[i,j].yaxis.set_tick_params(labelsize=15)
        fig2.legend([l1, l2], ['desired', 'real'], bbox_to_anchor=(0.75,0.99), prop={'size': 20})

        plt.figure(3) # Diagramm for Foot Position (desired/real)
        fig3, axs3 = plt.subplots(3, 4)
        fig3.suptitle(Suptitles[2], fontweight="bold", size=30)
        for i in range(3):
            for j in range(4):
                l1, = axs3[i,j].plot(self.tList, self.FPdList[i+j*3], color="green", label='desired')
                l2, = axs3[i,j].plot(self.tList, self.FPtList[i+j*3], color="blue", label='real')
                axs3[i,j].grid()
                if i == 0:
                    axs3[i,j].set_title(Leg_Name[j], size=20)
                if i == 2:
                    axs3[i,j].set_xlabel('Time [s]', size=20)
                if j == 0:
                    ylabel_Name = Dim_Name[i] + "-axis"
                    axs3[i,j].set_ylabel(ylabel_Name, size=20)
                axs3[i,j].xaxis.set_tick_params(labelsize=15)
                axs3[i,j].yaxis.set_tick_params(labelsize=15)
        fig3.legend([l1, l2], ['desired', 'real'], bbox_to_anchor=(0.75,0.99), prop={'size': 20})

        plt.show()