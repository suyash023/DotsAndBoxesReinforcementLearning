Robot Learning. Assignment 3. Code Readme.
All codes are written in python and libraries that have been used are numpy and matplotlib.


1) Codes folder:

None of the code files take inputs from the user.
To run the code in the terminal please type:
For reproducibility of results each of the code files has a random seed. Changing the random seed could significantly effect the performance of the algorithm.
Note: grid3_qtable.py takes a significant amount of time to train.

python file_name.py

Contains the source files for each of the question.
grid2_qtable.py --> code for 2x2 grid using q table
grid2_nn.py --> code for 2x2 grid using neural network as a function approximator.
grid3_qtable.py --> code for 3x3 grid using q table
grid3_nn.py --> code for 3x3 grid using neural network as a function approximator

The code grid2_qtable.py generates a csv file called 'dict.csv' which is used by the grid3_table.py file.

To visualize the game please uncomment the lines draw_board_state() in the code to see one of the games.


2) Images folder

Contains images of the plots of draws and wins for each of the questions. Also attached in report. 
