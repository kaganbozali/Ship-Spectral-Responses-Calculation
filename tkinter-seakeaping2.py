#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:24:38 2023

@author: kagan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy import integrate
from tkinter import *
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import Tk, Label
from tkinter import ttk
from scipy.interpolate import interp1d
import csv
import sys
from PIL import Image, ImageTk
import matplotlib.image as mpimg


class MyWindow:
    def __init__(self,win):
        
        self.win = win
        self.win.title('Ship Spectra Calculator')
        self.win.geometry("1280x700+10+10")
        self.win.attributes('-fullscreen', True)
        self.win.configure(bg='white')
        
        # Close button
        self.close_btn = ttk.Button(win, text="X", command=self.close_window)
        self.close_btn.place(x=self.win.winfo_screenwidth() - 40, y=0, width=30, height=20)

        # Minimize button
        self.minimize_btn = ttk.Button(win, text="-", command=self.minimize_window)
        self.minimize_btn.place(x=self.win.winfo_screenwidth() - 80, y=0, width=30, height=20)
        
        # Python Console 
        self.console = scrolledtext.ScrolledText(win, width=120, height=15, bg='white', fg='black')
        self.console.place(x=10, y=570)

        sys.stdout = ConsoleRedirector(self.console, 'stdout')
        sys.stderr = ConsoleRedirector(self.console, 'stderr')    
        

        image_path = "icon_1.jpg"
        self.image = mpimg.imread(image_path)

        fig = plt.figure(figsize=(2, 2), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(self.image)
        ax.axis('off')

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().place(x=480, y=-20)
        
        image_path_2 = "logo.png"
        self.image_2 = mpimg.imread(image_path_2)
        
        fig_2 = plt.figure(figsize=(0.7, 0.7), dpi=100)
        ax = fig_2.add_subplot(111)
        ax.imshow(self.image_2)
        ax.axis('off')

        canvas = FigureCanvasTkAgg(fig_2, master=win)
        canvas.draw()
        canvas.get_tk_widget().place(x=-10, y=-20)        
        
        # Labels
        self.lbl1 = Label(win, text = 'L[m]',bg='white')
#        self.lbl2 = Label(win, text = 'B[m]')
#        self.lbl3 = Label(win, text = 'T[m]')
#        self.lbl4 = Label(win, text = 'Cb')
#        self.lbl5 = Label(win, text = 'Main Dimensions',bg='white')
#        self.lbl6 = Label(win, text = 'GM[m]')
#        self.lbl7 = Label(win, text = 'KG[m]')
#        self.lbl8 = Label(win, text = 'Model Scale',bg='white')
        self.lbl14 = Label(win, text = 'Model Speed[m/s]',bg='white')
        self.lbl15 = Label(win, text= 'Heading Angle[deg]',bg='white')
#        self.lbl9 = Label(win, text = 'Hydrostatic Values',bg='white')
        self.lbl10 = Label(win, text = 'Hs[m]',bg='white')
        self.lbl11 = Label(win, text = 'Tz[s]',bg='white')
        self.lbl12 = Label(win, text = 'Sea State',bg='white')
        self.lbl13 = Label(win,text = 'Interpolation Size',bg='white')
        # Entries
        self.t1=Entry(bd=1)
#        self.t2=Entry()
#        self.t3=Entry()
#        self.t4=Entry()
#        self.t6=Entry()
#        self.t7=Entry()
        self.t8=Entry()
        self.t10=Entry()
        self.t11=Entry()
        self.t13=Entry()
        self.t14=Entry()
        self.t15=Entry()
        # Hidden Entries
        self.t16=Entry()
        self.t17=Entry()
        self.t18=Entry()
        # Buttons
        self.btn = Button(win, text = 'Load Model Test Result')
        self.btn2 = Button(win, text = 'Interpolate Model Test Results')
        self.btn3 = Button(win, text = 'Plot the Spectra')
        self.btn4 = Button(win, text = 'Validate')
        self.btn5 = Button(win, text = 'Calculate RMS')
        self.btn6 = Button(win, text = 'RMS Polar')
        # Label and Entries Position
        self.lbl1.place(x=9, y=60)
        self.t1.place(x=150, y=60)
#        self.lbl2.place(x=60, y=75)
#        self.t2.place(x=100, y=75)
#        self.lbl3.place(x=60,y=100)
#        self.t3.place(x=100,y=100)
#        self.lbl4.place(x=60,y=125)
#        self.t4.place(x=100,y=125)
#        self.lbl5.place(x=120,y=30)
#        self.t6.place(x=390,y=50)
#        self.lbl6.place(x=260,y=50)
#        self.t7.place(x=390,y=75)
#        self.lbl7.place(x=260,y=75)
#        self.t8.place(x=390,y=100)
#        self.lbl8.place(x=260,y=100)        
#        self.lbl9.place(x=360,y=30)
        self.t10.place(x=350,y=60)
        self.lbl10.place(x=300,y=60)
        self.t11.place(x=350,y=85)
        self.lbl11.place(x=300,y=85)         
        self.lbl12.place(x=390,y=40)
        self.lbl13.place(x=9,y=35)
        self.t13.place(x=150,y=35)
        self.t14.place(x=150,y=85)
        self.lbl14.place(x=9,y=85)
        self.t15.place(x=150,y=110)
        self.lbl15.place(x=9,y=110) 
        # Button Position
        self.btn.place(x=50,y=0)
        self.btn2.place(x=240,y=0)    
        self.btn3.place(x=485,y=0)
        self.btn4.place(x=730,y=0)
        self.btn5.place(x=820,y=0)
        self.btn6.place(x=630,y=0)
        # Button Tasks
        self.btn.config(command=self.load_model_test_results)
        self.btn2.config(command=self.interp_results)
        self.btn3.config(command = self.plot_spectra)
        self.btn4.config(command = self.validate)
        self.btn5.config(command = self.calc_rms)
        self.btn6.config(command = self.RMS_Head)


    def load_model_test_results(self):
        file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if file_path:
            self.process_csv_file(file_path)
    
    def process_csv_file(self, file_path):
        global lamda_L
        global eta3_A
        global eta5_kA
        
        print(f"Processing CSV file: {file_path}")
    
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) > 1:
                data_rows = rows[1:]
                num_columns = len(data_rows[0])
    
                if num_columns != 3:
                    print("Error: The CSV file does not contain three columns of data.")
                    return
    
                lamda_L = []
                eta3_A = [] 
                eta5_kA = []  
    
                input_fields = [self.t16, self.t17, self.t18]
                for i, row in enumerate(data_rows):
                    for j, value in enumerate(row):
                        if j == 0:
                            lamda_L.append(float(value))
                        elif j == 1:
                            eta3_A.append(float(value))
                        elif j == 2:
                            eta5_kA.append(float(value))
                        if j < len(input_fields):
                            input_field = input_fields[j]
                            input_field.delete(0, END)
                            input_field.insert(0, value)
                            print(f"Assigned value '{value}' to input field at row {i + 2}, column {j + 1}")
    
                lamda_L = np.array(lamda_L)
                eta3_A = np.array(eta3_A)
                eta5_kA = np.array(eta5_kA)
    
                print("lamda_L values:", lamda_L)
                print("eta3/A values:", eta3_A)
                print("eta5/kA values:", eta5_kA)
            else:
                print("CSV file does not contain any data.")
    
        print("CSV file processing completed.")
    
    def interp_results(self):
        global x_1_new
        global x_2_new
        global x_3_new
        global n
        
        n = int(self.t13.get())
        f_1 = interp1d(lamda_L,eta3_A)
        f_2 = interp1d(lamda_L,eta5_kA)
        x_1_new = np.linspace(lamda_L[0],lamda_L[-1],n)
        x_2_new = f_1(x_1_new)
        x_3_new = f_2(x_1_new)
        print("Interpolation has completed")
    
    def plot_spectra(self):
        global S_ittc
        global w
        global S_we
        global we
        global h_3
        global h_5
        global k
        global g
        global V_m
        
        L_r = float(self.t1.get())
        V_m = float(self.t14.get())
        beta = float(self.t15.get())
        Hs = float(self.t10.get())
        Tz = float(self.t11.get())
        beta = np.pi*beta/180
        g = 9.81
        lamda = L_r * x_1_new
        k = 2*np.pi / lamda
        #ITTC Spectrum Calculation
        A= (123.8*Hs**2)/(Tz**4)
        B= 495/(Tz**4)
        w= (g*k)**(0.5)
        we = w - ( (w**2/g) * V_m*np.cos(beta))
        S_ittc = (A / w**5 ) * np.exp(-B/w**4)
        eta_3 = x_2_new*A
        eta_5 = x_3_new*k*A
        d_w=(1-(2*w*V_m*np.cos(beta)/g))
        S_we = S_ittc / d_w
        h_3 = S_we*x_2_new**2
        h_5 = S_we*x_3_new**2*k**2
        
        
        fig_3 = plt.figure(figsize=(5,2), dpi= 100)
        plt.plot(w,we)
        plt.xlabel('W(rad/s)')
        plt.ylabel('We(rad/s)')
        canvas = FigureCanvasTkAgg(fig_3, master=window)
        canvas.draw()
        canvas.get_tk_widget().place(x=880, y=60)
        
        fig = plt.figure(figsize=(5, 2), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(we, S_we, label='Ship Spectra')
        ax.plot(we, x_2_new, label = 'RAO(Heave)')
        ax.plot(we, x_3_new, label = 'RAO(Pitch)')
        ax.set_xlabel('Encounter Frequency')
        ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().place(x=880, y=270)
        
        fig_2 = plt.figure(figsize=(5,2),dpi=100)
        plt.plot(we,h_3,label='Heave')
        plt.plot(we,h_5*1000,label='Pitch(scale=1000)')
        plt.xlabel('Encounter Frequency')
        plt.legend()
        canvas = FigureCanvasTkAgg(fig_2, master=window)
        canvas.draw()
        canvas.get_tk_widget().place(x=880, y=490)
        
        
    def validate(self):
        m = []
        for i in range(6):
            m.append(np.abs(integrate.simpson(S_ittc*w**i,w)))
        c_1 = 4*(np.abs(m[0])**(0.5))
        print(c_1,"4 square root of m0 value for wave spectra")
        
        m_we = []
        for i in range(6):
            m_we.append(np.abs(integrate.simpson(S_we*we**i,we)))
        c_2 = 4*(np.abs(m_we[0])**(0.5))
        print(c_2,"4 square root of m0 value for ship spectra")
        
        
    def calc_rms(self):

        delta_we=[]
        for i in range(len(we)-1):
            delta_we.append(we[i]-we[i+1])
        delta_we = np.array(delta_we)

        RMS_3=np.zeros(n)
        RMS_3[0]=h_3[0]
        RMS_3[1:n]=list(h_3[1:n]*delta_we)
        RMS_3_val=np.sqrt(np.sum(RMS_3))
        print('RMS3 = ', RMS_3_val)

        RMS_5=np.zeros(n)
        RMS_5[0]=h_5[0]
        RMS_5[1:n]=list(h_5[1:n]*delta_we)
        RMS_5_val=np.sqrt(np.sum(RMS_5))
        print('RMS5 = ', RMS_5_val)

    
    def RMS_Head(self):
        beta_new = np.array([0,30,60,90,120,150,180,210,240,270,300,330])
        we_new = []
        dw_new = []
        S_we_new = []
        h_3_new = []
        h_5_new = []
        
        for i in range(len(beta_new)):
            we_new.append(w - ( (w**2/g) * V_m*np.cos(beta_new[i])))
            dw_new.append((1-(2*w*V_m*np.cos(beta_new[i])/g)))
            S_we_new.append(S_ittc / dw_new[i])
            h_3_new.append(S_we[i]*x_2_new**2)
            h_5_new.append(S_we[i]*x_3_new**2*k**2)


        delta_we_new=[]
        for j in range(len(beta_new)):
            for i in range(len(we_new[0])-1):
                delta_we_new.append(we_new[j][i]-we_new[j][i+1])
        #delta_we_new = np.array(delta_we_new)
        delta_we_new = np.array_split(delta_we_new, len(beta_new))

        
        RMS_3_new = np.zeros((len(beta_new), n))
        RMS_5_new = np.zeros((len(beta_new), n))
        RMS_3_polar = []
        RMS_5_polar = []
        
        for i in range(len(beta_new)):
            print('Heading Angle:', beta_new[i])
            RMS_3_new[i] = np.zeros(n)
            RMS_3_new[i][0] = (h_3_new[i][0])
            RMS_3_new[i][1:n] = (list(h_3_new[i][1:n]*delta_we_new[i]))
            RMS_3_val_new = (np.sqrt(np.sum(RMS_3_new[i])))
            RMS_3_polar.append(RMS_3_val_new)
            print('RMS3 = ', RMS_3_val_new)
            
            RMS_5_new[i] = np.zeros(n)
            RMS_5_new[i][0] = (h_5_new[i][0])
            RMS_5_new[i][1:n] = (list(h_5_new[i][1:n]*delta_we_new[i]))
            RMS_5_val_new = (np.sqrt(np.sum(RMS_5_new[i])))
            RMS_5_polar.append(RMS_5_val_new*20)
            print('RMS5 = ', RMS_5_val_new)

        fig_4 = plt.figure(figsize=(4.5, 4.5), dpi=100)
        ax = fig_4.add_subplot(111, polar=True)
        
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.set_rticks([])  

        theta = np.deg2rad(beta_new)  # Convert beta_new to radians
        ax.plot(theta, RMS_3_polar, 'o-', label='RMS3')
        ax.plot(theta, RMS_5_polar, 'o-', label='RMS5(Scale = 20)')
        
        # Set x-ticks at theta positions for each heading angle
        ax.set_xticks(theta)
        ax.set_xticklabels(beta_new)
        

        ax.legend()
        
        # Display the plot in the interface
        canvas = FigureCanvasTkAgg(fig_4, master=self.win)
        canvas.draw()
        canvas.get_tk_widget().place(x=350, y=120)
        
    
    def close_window(self):
        self.win.destroy()
    
    def minimize_window(self):
        self.win.iconify()
        
    def animate(self):
        self.canvas.move(self.ball, self.dx, 0)
        self.win.after(100, self.animate)  # Update animation every 10 milliseconds


class ConsoleRedirector:
    def __init__(self, console, tag):
        self.console = console
        self.tag = tag

    def write(self, message):
        self.console.insert('end', message, self.tag)
        self.console.see('end')

    def flush(self):
        pass

window=Tk()
mywin=MyWindow(window)
window.title('Ship Spectra Calculator')
window.geometry("1280x700+10+10")
window.mainloop()
