Robot Learning. Assignment 3. Code Readme.
All codes are written in python and libraries that have been used are numpy and matplotlib.


1) Codes folder:

None of the code files take inputs from the user.
To run the code in the terminal please type:

python file_name.py


For reproducibility of results each of the code files has a random seed. Changing the random seed could significantly effect the performance of the algorithm.
Note 1: grid3_qtable.py takes a significant amount of time to train.
Note 2: grid3_qtable.py required the file dict.csv to be in the Codes directory, please transfer the file only then run the code otherwise it will throw an error saying the file is not there

Contains the source files for each of the question.
grid2_qtable.py --> code for 2x2 grid using q table
grid2_nn.py --> code for 2x2 grid using neural network as a function approximator.
grid3_qtable.py --> code for 3x3 grid using q table
grid3_nn.py --> code for 3x3 grid using neural network as a function approximator

The code grid2_qtable.py generates a csv file called 'dict.csv' which is used by the grid3_table.py file.

To visualize the game please uncomment the lines draw_board_state() in the code to see one of the games.

The plots for performance are generated finally after all the iterations have been run.


2) Images folder

Contains images of the plots of draws and wins for each of the questions. Also attached in report. 
