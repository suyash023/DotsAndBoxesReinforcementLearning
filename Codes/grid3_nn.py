import numpy as np
from matplotlib import pyplot as plt




eps = 0.1
learning_rate = 0.001
discount_factor = 0.3
players_box_count = [0,0]
time_step = 1
game_complete_flag = 0
num_iterations = range(100,10200,1000)
init_turn = 0
reward_flag =0
np.random.seed(10)
qlearning_rate = 0.8

def init_weights(layer_size,output_layer_size):
	weights = np.random.uniform(0,1,(layer_size,output_layer_size))
	return weights

def linear(z):
    #output_indices = np.where(z<=0)
    relu_output = z.copy()
    #relu_output[output_indices] = 0
    return relu_output

def layer_output(layer_input,weights,bias):
	out = np.dot(layer_input,weights) + bias.transpose()
	#print "layer input: ",np.dot(layer_input,weights).shape
	layer_output = linear(out)
	return layer_output

def prepare_input(state,action):
	nn_input = []
	for i in range(0,len(state)):
		number = int(state[i])
		nn_input.append(number)
	action_input = [0]*24
	action_input[action] = 1
	nn_input = nn_input + action_input
	nn_input_arr = np.array(nn_input)
	nn_input_arr = nn_input_arr.reshape((1,len(nn_input)))
	return nn_input_arr


def forward_prop(state,action,layer1_weights,layer2_weights,layer1_bias,layer2_bias):
	nn_input = prepare_input(state,action)
	layer1_output = layer_output(nn_input,layer1_weights,layer1_bias)
	#print layer1_output.shape
	layer2_output = layer_output(layer1_output,layer2_weights,layer2_bias)
	return layer2_output,layer1_output,nn_input


def linear_derivative(x):
	#greater_indices = np.where(x>0)
	#lesser_indices = np.where(x<=0)
	#output = x.copy()
	#output[lesser_indices] = 0
	#output[greater_indices] = 1
	output = np.ones((x.shape[0],x.shape[1]))
	return output


def back_prop(layer1_weights,layer1_output,layer2_weights,expected_output,output,nn_input):
	global learning_rate
	error = expected_output-output
	layer2_delta = error*linear_derivative(output)
	#print "Relu derivative", relu_derivative(output)
	layer1_error = layer2_delta*(layer2_weights.transpose())
	layer1_delta = layer1_error*linear_derivative(layer1_output)
	#print "Delta: ",(nn_input.transpose()).dot(layer1_delta)
	layer1_weights += learning_rate*(nn_input.transpose()).dot(layer1_delta)
	layer2_weights += learning_rate*(layer1_output.transpose()).dot(layer2_delta)
	return layer1_weights,layer2_weights




# def neural_network(state,action,layer1_weights,layer2_weights):
# 	layer1_weights = init_weights(48,24)
# 	layer2_weights = init_weights(24,24)
# 	layer1_bias = np.zeros((layer1_weights.shape[1],1))
# 	layer2_bias = np.zeros((1,1))
# 	#print layer2_bias.shape

# 	#nn_input = prepare_input(state,action)
# 	#print nn_input
# 	for i in range(1,15):
# 		output,layer1_output,nn_input = forward_prop(state,action,layer1_weights,layer2_weights,layer1_bias,layer2_bias)
# 		layer1_weights,layer2_weights = back_prop(layer1_weights,layer1_output,layer2_weights,expected_output,output,nn_input)
# 		print layer1_weights,layer2_weights
# 		print output



	# self.o_error = y - o # error in output
 #    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

 #    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
 #    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

 #    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
 #    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights




	





def init_board():
	board_state = '0'*24
	layer1_weights = init_weights(48,100)
	layer2_weights = init_weights(100,1)
	return board_state,layer1_weights,layer2_weights





def draw_board_state(board_state):
	points = []
	for i in range(1,5):
		for j in range(1,5):
			points.append([i,j])
	points_arr = np.array(points)
	plt.scatter(points_arr[:,0],points_arr[:,1])
	color = [0,1,1]
	counter = 0
	if counter == 0:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([1,2],[1,1],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==1:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([2,3],[1,1],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==2:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([3,4],[1,1],color=tuple(color))
		color=[0,1,1]
		counter = counter+1		
	if counter==3:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([1,1],[1,2],color=tuple(color))
		color=[0,1,1]
		counter = counter+1	
	if counter==4:
		color[int(board_state[counter])] =1 if board_state[counter] == '0' else 0
		plt.plot([2,2],[1,2],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==5:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([3,3],[1,2],color=tuple(color))
		color=[0,1,1]
		counter = counter+1	
	if counter==6:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([4,4],[1,2],color=tuple(color))
		color=[0,1,1]
		counter = counter+1	
	if counter==7:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([1,2],[2,2],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==8:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([2,3],[2,2],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==9:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([3,4],[2,2],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==10:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([1,1],[2,3],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==11:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([2,2],[2,3],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==12:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([3,3],[2,3],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==13:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([4,4],[2,3],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==14:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([1,2],[3,3],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==15:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([2,3],[3,3],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==16:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([3,4],[3,3],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==17:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([1,1],[3,4],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==18:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([2,2],[3,4],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==19:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([3,3],[3,4],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==20:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([4,4],[3,4],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==21:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([1,2],[4,4],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==22:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([2,3],[4,4],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	if counter==23:
		color[int(board_state[counter])] = 1 if board_state[counter] == '0' else 0
		plt.plot([3,4],[4,4],color=tuple(color))
		color=[0,1,1]
		counter = counter+1
	plt.show()




def epsilon_greedy():
	#global eps
	total_actions = 3**(24)*24
	eps = 0.2
	#eps = float(len(qtable))/total_actions 
 	prob = np.random.uniform(0,1,1)
	if prob < 1- eps:
		#print "Eps: ",1-eps
		#print "Exploring."
		return 1
	else:
		#print "Eps: ",eps
		#print "Exploiting"
		return 0


#def check_table(board_state):

def get_new_board_state(board_state,action_index,player):
	new_state = board_state[:action_index] + str(1) + board_state[action_index + 1:]
	return new_state


def check_reward(board_state,action_index,player):
	global players_box_count,game_complete_flag,init_turn,reward_flag
	indices_to_check = [0,1,2,7,8,9,14,15,16]
	indices_list = []
	reward = 0
	reward_flag = 0
	new_state = get_new_board_state(board_state,action_index,player)
	for i in range(0,len(board_state)):
		if new_state[i] != '0':
			indices_list.append(i)
		#print len(indices_list)
	if len(indices_list) == 24:
		game_complete_flag = 1

	for index in indices_to_check:
		box_indices = [index,index+4,index+7,index+3]
		counter = 0
		player_index = player-1
		completed = []
		for element in box_indices:
			if element in indices_list:
				completed.append(element)
		#print completed
		if len(completed) == 4 and action_index in completed :
			reward = reward + 1
			#print "Box completed! PLayer "+str(player)+" got reward of " + str(reward)
			#players_box_count[player_index] = players_box_count[player_index] + 1
	if reward > 0:
		#print "turn changed",init_turn
		reward_flag = 1
	player_index = player-1
	players_box_count[player_index] = players_box_count[player_index] + reward
	if players_box_count[0]+players_box_count[1] == 9:
		game_complete_flag =1 
		#print players_box_count
		#players_box_count[player_index] >= players_box_count[1-player_index]
		if players_box_count[player_index] > players_box_count[1-player_index]:
			reward = reward+5
			#print "Game complete! Player "+str(player)+" got reward of 5"
		else:
			reward = 0
	return reward 





# def update_q_table(board_state,qtable,action_index,reward,player):
# 	global learning_rate,discount_factor, time_step
# 	max_qvalue_action = action_index
# 	futureq_values = 0
# 	new_state = board_state
# 	for i in range(0,time_step):
# 		new_state = get_new_board_state(new_state,max_qvalue_action,player)
# 		new_state_max_qvalue = 0
# 		for action in range(0,12):
# 			if new_state in qtable.keys() and action in qtable[new_state].keys():
# 				if qtable[new_state][action] > new_state_max_qvalue:
# 					new_state_max_qvalue = qtable[new_state][action]
# 					max_qvalue_action = action
# 		futureq_values = futureq_values + discount_factor**(i+1) * new_state_max_qvalue

# 	if not ( board_state in qtable.keys() and action_index in qtable[board_state].keys()):
# 		qvalue = learning_rate * (reward+ futureq_values)
# 		print board_state,action_index
# 		qtable.update({board_state :{action_index: qvalue }})
# 	else:
# 		qvalue = (1-learning_rate)*qtable[board_state][action_index] + learning_rate*(reward+ futureq_values)
# 		qtable[board_state,action_index] = qvalue
# 	return qtable


def update_neural_network(board_state,action_index,layer1_weights,layer2_weights,reward,player):
	global learning_rate,discount_factor, time_step,qlearning_rate
	max_qvalue_action = action_index
	futureq_values = 0
	new_state = board_state
	layer1_bias = np.zeros((layer1_weights.shape[1],1))
	layer2_bias = np.zeros((1,1))
	for i in range(0,time_step):
		new_state = get_new_board_state(new_state,max_qvalue_action,player)
		new_state_max_qvalue = 0
		incomplete_indices = []
		for j in range(0,len(board_state)):
			if new_state[i]== '0':
				incomplete_indices.append(i)
		for action in incomplete_indices:
			#print layer1_weights.shape
			qvalue,layer1_output,nn_input = forward_prop(new_state,action,layer1_weights,layer2_weights,layer1_bias,layer2_bias)
			if qvalue >new_state_max_qvalue:
				new_state_max_qvalue = qvalue
				max_qvalue_action = action
		futureq_values = futureq_values + discount_factor**(i+1) * new_state_max_qvalue
	old_qvalue,layer1_output,nn_input = forward_prop(board_state,action_index,layer1_weights,layer2_weights,layer1_bias,layer2_bias)
	# if old_qvalue < 0:
	# 	old_qvalue = 0
	new_qvalue = (1-qlearning_rate)*old_qvalue+qlearning_rate* (reward + futureq_values)
	nn_qvalue = 10000
	while abs(new_qvalue - nn_qvalue)>0.01 :
		nn_qvalue,layer1_output,nn_input = forward_prop(board_state,action_index,layer1_weights,layer2_weights,layer1_bias,layer2_bias)
		layer1_weights,layer2_weights = back_prop(layer1_weights,layer1_output,layer2_weights,new_qvalue,nn_qvalue,nn_input)
		#print "input: ",nn_input
	#print "Comparing outputs: ",nn_qvalue,new_qvalue
	return layer1_weights,layer2_weights







def get_reward_maximizing_action(board_state,layer1_weights,layer2_weights):
	max_qvalue = 0
	action_index = -1
	layer1_bias = np.zeros((layer1_weights.shape[1],1))
	layer2_bias = np.zeros((1,1))
	incomplete_indices = []
	for i in range(0,len(board_state)):
		if board_state[i]== '0':
			incomplete_indices.append(i)
	for action in incomplete_indices:
		qvalue,layer1_output,nn_input = forward_prop(board_state,action,layer1_weights,layer2_weights,layer1_bias,layer2_bias)
		if qvalue> max_qvalue:
			#print "Found a better action:",qvalue
			max_qvalue = qvalue
			action_index = action
	if action_index == -1:
		list_index = np.random.randint(0,len(incomplete_indices),1)[0]
		#print 'Choosing random action'
		action_index = incomplete_indices[list_index]
	return action_index


def qlearning(board_state,layer1_weights,layer2_weights,player):
	exp_or_exploit = epsilon_greedy()
	incomplete_indices = []
	for i in range(0,len(board_state)):
		if board_state[i]== '0':
			incomplete_indices.append(i)
	#print incomplete_indices
	if exp_or_exploit == 1:
		list_index = np.random.randint(0,len(incomplete_indices),1)[0]
		action_index = incomplete_indices[list_index]
	else:
		#print "Layer1 shape:",layer1_weights.shape
		action_index = get_reward_maximizing_action(board_state,layer1_weights,layer2_weights)
	reward = check_reward(board_state,action_index,player)
	layer1_weights,layer2_weights = update_neural_network(board_state,action_index,layer1_weights,layer2_weights,reward,player)
	board_state = get_new_board_state(board_state,action_index,player)
	#draw_board_state(board_state)
	#print qtable	
	return layer1_weights,layer2_weights,board_state	




def self_play(board_state,layer1_weights,layer2_weights):
	global game_complete_flag,players_box_count,init_turn,reward_flag
	players_box_count = [0,0]
	game_complete_flag = 0
	init_turn = np.random.randint(0,2)
	reward_flag = 0
	if init_turn == 0:
		#print "Player 1's turn for the first time"
		turn_flag = 0
	else: 
		#print "Player 2's turn for the first time"
		turn_flag = 1
	#print init_turn
	while not game_complete_flag:
		if init_turn == 0:
			layer1_weights,layer2_weights,board_state = qlearning(board_state,layer1_weights,layer2_weights,1)
			while reward_flag == 1 and not game_complete_flag:
				layer1_weights,layer2_weights,board_state = qlearning(board_state,layer1_weights,layer2_weights,1)
			if not game_complete_flag:
				layer1_weights,layer2_weights,board_state = qlearning(board_state,layer1_weights,layer2_weights,2)
				while reward_flag == 1 and not game_complete_flag:
					layer1_weights,layer2_weights,board_state = qlearning(board_state,layer1_weights,layer2_weights,2)
		elif init_turn == 1:
			layer1_weights,layer2_weights,board_state = qlearning(board_state,layer1_weights,layer2_weights,2)
			while reward_flag == 1 and not game_complete_flag:
				layer1_weights,layer2_weights,board_state = qlearning(board_state,layer1_weights,layer2_weights,2)
			if not game_complete_flag:
				layer1_weights,layer2_weights,board_state = qlearning(board_state,layer1_weights,layer2_weights,1)
				while reward_flag == 1 and not game_complete_flag:
					layer1_weights,layer2_weights,board_state = qlearning(board_state,layer1_weights,layer2_weights,1)

	return layer1_weights,layer2_weights
	


def board_reset():
	board_state = '0'*24
	return board_state



def random_play(board_state,layer1_weights,layer2_weights):
	global game_complete_flag,players_box_count,init_turn,reward_flag
	players_box_count = [0,0]
	game_complete_flag = 0
	init_turn = np.random.randint(0,2)
	incomplete_indices = []
	reward_flag = 0
	if init_turn == 0:
		#print "Player 1's turn for the first time"
		turn_flag = 0
	else: 
		#print "Player 2's turn for the first time"
		turn_flag = 1
	while not game_complete_flag:
		incomplete_indices = []
		if init_turn == 0:
			player1_action_index = get_reward_maximizing_action(board_state,layer1_weights,layer2_weights)
			reward=check_reward(board_state,player1_action_index,1)
			board_state = get_new_board_state(board_state,player1_action_index,1)
			#draw_board_state(board_state)
			while reward_flag == 1 and not game_complete_flag:
				player1_action_index = get_reward_maximizing_action(board_state,layer1_weights,layer2_weights)
				reward=check_reward(board_state,player1_action_index,1)
				board_state = get_new_board_state(board_state,player1_action_index,1)
				#draw_board_state(board_state)
			if not game_complete_flag:
				incomplete_indices = []
				for i in range(0,len(board_state)):
					if board_state[i]== '0':
						incomplete_indices.append(i)
				list_index = np.random.randint(0,len(incomplete_indices),1)[0]
				player2_action_index = incomplete_indices[list_index]
				reward=check_reward(board_state,player2_action_index,2)
				board_state =get_new_board_state(board_state,player2_action_index,2)
				#draw_board_state(board_state)
				while reward_flag == 1 and not game_complete_flag:
					incomplete_indices = []
					for i in range(0,len(board_state)):
						if board_state[i]== '0':
							incomplete_indices.append(i)
					list_index = np.random.randint(0,len(incomplete_indices),1)[0]
					player2_action_index = incomplete_indices[list_index]
					reward=check_reward(board_state,player2_action_index,2)
					board_state =get_new_board_state(board_state,player2_action_index,2)
					#draw_board_state(board_state)
		elif init_turn == 1:
			incomplete_indices = []
			for i in range(0,len(board_state)):
				if board_state[i]== '0':
					incomplete_indices.append(i)
			list_index = np.random.randint(0,len(incomplete_indices),1)[0]
			player2_action_index = incomplete_indices[list_index]
			reward=check_reward(board_state,player2_action_index,2)
			board_state =get_new_board_state(board_state,player2_action_index,2)
			#draw_board_state(board_state)
			while reward_flag == 1 and not game_complete_flag:
				incomplete_indices = []
				for i in range(0,len(board_state)):
					if board_state[i]== '0':
						incomplete_indices.append(i)
				list_index = np.random.randint(0,len(incomplete_indices),1)[0]
				player2_action_index = incomplete_indices[list_index]
				reward=check_reward(board_state,player2_action_index,2)
				board_state =get_new_board_state(board_state,player2_action_index,2)
				#draw_board_state(board_state)
			if not game_complete_flag:
				player1_action_index = get_reward_maximizing_action(board_state,layer1_weights,layer2_weights)
				reward=check_reward(board_state,player1_action_index,1)
				board_state = get_new_board_state(board_state,player1_action_index,1)
				#draw_board_state(board_state)
				while reward_flag == 1 and not game_complete_flag:
					player1_action_index = get_reward_maximizing_action(board_state,layer1_weights,layer2_weights)
					reward=check_reward(board_state,player1_action_index,1)
					board_state = get_new_board_state(board_state,player1_action_index,1)
					#draw_board_state(board_state)

			#print player1_action_index
			#draw_board_state(board_state)
	if players_box_count[0] > players_box_count[1]:
		return 1,0,0
	elif players_box_count[0] == players_box_count[1]:
		return 0,1,0
	else:
		return 0,0,1

wins=[]
draws = []


for niters in num_iterations:
	board_state,layer1_weights,layer2_weights = init_board()
	print "Iterations: ",niters

	for i in range(0,niters):
		boad_state = board_reset()
		layer1_weights,layer2_weights=self_play(board_state,layer1_weights,layer2_weights)
	total_wins = 0
	total_draws = 0
	for i in range(0,100):
		board_state = board_reset()
		player1_win,game_draws,player2_win = random_play(board_state,layer1_weights,layer2_weights)
		total_wins = total_wins + player1_win
		total_draws = total_draws + game_draws
	wins.append(float(total_wins)/100)
	draws.append(float(total_draws)/100)
	print "accuracy of player1_win: ", wins
	print "Draws: ",draws


#print "Player 1 q table: ",player1_qtable
#print "Player 2 q table: ",player2_qtable

plt.plot(num_iterations,wins,color=(1,0,0),label="wins")
plt.plot(num_iterations,draws,color=(0,1,0),label="draws")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("game wins or draws percentages")
plt.show()





















































