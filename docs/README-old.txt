DISCLAIMER
I cannot guarantee that the program will run on every system or that it is known how to run it. 
It is still possible to open all of the program files to read the code. I have included instructions below, if applicable.

FILE INFORMATION
'data' stores the test and train set of words used by the network.

'WandBsaves' has the network's final weights and biases after 20 iterations for each of 12 different networks styles. 
This is so a programmer can import them to see their accuracy and other features instead of retraining the network 
for many hours, like I did, to get to the same point.

'artefact.py' is the user interface for the network. It imports the generic feedforward neural network in 'feedforward.py'
 and instantiates the network class with various values such as the step size.

'confidences.csv' is the output of the guess confidence for each word in the test set

'feedforward.py' is the main body of code of the neural network

ARTEFACT FILE LOCATION
The main body of the artefact has location "\artefact and results\artefact.py".
This uses code stored in "\artefact and results\feedforward.py" which is also a part of the artefact.

DO NOT
Don't change the location of any files in the folder "artefact and results", or delete them, as the program will not work without them.

--- TO READ THE CODE ---

WINDOWS
1. Right click on any ".py" file 
2. Click "Open with..." and locate a text editing software, such as "Notepad".

OTHER
1. Open text editing software.
2. Click "File" then "Open" (usually at the top left).
3. Locate the ".py" file on the USB drive from the menu.

--- TO RUN THE ARTEFACT ---
If running the program is necessary, installation of Python 3.0 or later is required. 
You can find Python at https://www.python.org/downloads/. Remember to install version 3.0 or better and not version 2.7. 
The installation should be easy and not time consuming.

A GOOD METHOD
1. Run Python 3.0's (or later) IDLE software. This should be on your computer if you have used the Python installer on the website above.
If you do not know where this is on your computer, Windows can search for it by typing "IDLE" after pressing the home button.
Other systems have similar software search tools.
2. Go to File>Open at the top left of the screen in IDLE.
3. Use the file menu that opens to locate the "artefact.py" file and double-click to open.
4. While the editing window for "artefact.py" is open, press F5 on your keyboard or go to Run>Run Module at the top of the screen.
5. Type inputs into the window's field when the program asks for them.