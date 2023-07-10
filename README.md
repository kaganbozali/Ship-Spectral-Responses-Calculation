# Ship-Spectral-Responses-Calculation(Heave-Pitch)

In order to run the code, first you need to satisfy the libraries required for the program in Python. Then you should clone the repository and unzip it. Afterwards, you have to run the tool.py file in the Python notebook then the interface will come out. You suppose to see the screen in Figure-1.
Everything you can calculate using the tool is given below in detail.


**Prerequisites:**
- Tkinter
- PIL


![image](https://github.com/kaganbozali/Ship-Spectral-Responses-Calculation-Heave-Pitch-/assets/104154215/59c83d18-c1a1-4aeb-ab90-9904ca15f10b)
Fig 1. General View of the System

The program allows you to import your model test results .csv file. You should click load model test results on the left-hand side and when you import the file successfully you will able to see "CSV File processing completed." text in the console on the bottom.

![image](https://github.com/kaganbozali/Ship-Spectral-Responses-Calculation-Heave-Pitch-/assets/104154215/b9285396-e713-4399-959e-c1578b13e08a)

Fig 2. Kernel when you import Model Results

Afterwards when you performed a model test probably you will not be able to perform it 200 times so in order to increase the convergence of the model test. RAO results that you imported as .csv file interpolated. In order to apply the tool you should put the desired number for interpolation to the interpolation size entry then you should click interpolate model test results button. At the end, you should able to see "Interpolation has complated" text in the console(Fig 3.)

![image](https://github.com/kaganbozali/Ship-Spectral-Responses-Calculation-Heave-Pitch-/assets/104154215/3e7f144c-1405-4c42-9966-7c9dd38cf8bf)
Fig 3. Interpolation of Results

Then there are several options available which includes calculating heave, pitch and ship spectral values with spesific significant wave height and zero crossing period(Fig 4.) or you can choose polar plot option which includes RMS values for every heading angle. (Fig 5.)

![image](https://github.com/kaganbozali/Ship-Spectral-Responses-Calculation-Heave-Pitch-/assets/104154215/45580b6d-8bfa-42e7-8981-34e8066d1297)
Fig 4. Heave and Pitch Graphs

![image](https://github.com/kaganbozali/Ship-Spectral-Responses-Calculation-Heave-Pitch-/assets/104154215/0b0a5139-9932-45a0-a9d1-65146de8004d)
Fig 5. Polar Plot for Different Heading Angles

Additionally you can validate results, which calculates 4mo^0.5 values for both wave and ship spectra which corresponds to significant wave height value approximately and you can also calculate your RMS values for spesific heading angle using calculate RMS button on the top.
