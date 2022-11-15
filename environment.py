from gym import Env, utils
from gym.spaces import Discrete, Box
from sklearn.preprocessing import KBinsDiscretizer

import sys
import math
import random
import os.path
import pickle

sys.path.append('MacAPI')
import numpy as np
import sim


# Max movement along X
low, high = -0.05, 0.05

learning_rate = 0.1 #0.1
discount_rate = 0.99 #0.60
exploration_rate = 1.0 #0.05978456235635595 #0.1466885449377839
max_exploration_rate = 1.0 #0.05978456235635595 #0.1466885449377839 #0.37786092411182526
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

# Set state space and action space sizes
state_space_size = 7000 # 1000 + 50 + 50 (-0.05 to 0.05)
action_space_size = 5 # [-2, -1, 0, 1, 2]

num_episodes = 10

# Actions to move cup laterally
actions = [-2, -1, 0, 1, 2]


def setNumberOfBlocks(clientID, blocks, typeOf, mass, blockLength,
                      frictionCube, frictionCup):
    '''
        Function to set the number of blocks in the simulation
        '''
    emptyBuff = bytearray()
    res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(
        clientID, 'Table', sim.sim_scripttype_childscript, 'setNumberOfBlocks',
        [blocks], [mass, blockLength, frictionCube, frictionCup], [typeOf],
        emptyBuff, sim.simx_opmode_blocking)
    if res == sim.simx_return_ok:
        print(
            'Results: ', retStrings
        )  # display the reply from CoppeliaSim (in this case, the handle of the created dummy)
    else:
        print('Remote function call failed')

def triggerSim(clientID):
    e = sim.simxSynchronousTrigger(clientID)
    step_status = 'successful' if e == 0 else 'error'
    # print(f'Finished Step {step_status}')

def get_state(object_shapes_handles, clientID, pour):
    ''' Function to get the cubes and pouring cup position '''

    # Get position of the objects
    obj_pos = []
    for obj_handle in object_shapes_handles:
        # get the starting position of source
        returnCode, obj_position = sim.simxGetObjectPosition(
            clientID, obj_handle, -1, sim.simx_opmode_buffer)
        obj_pos.append(obj_position)

    returnCode, cup_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_buffer)

    return obj_pos, cup_position

def wait_(clientID):
    for _ in range(60):
        triggerSim(clientID)

def get_distance_3d(a, b):
    a_x, a_y, a_z = a[0], a[1], a[2]
    b_x, b_y, b_z = b[0], b[1], b[2]

    # Negative distance between source cup and receive cup
    return -math.sqrt((a_x - b_x)**2 + (a_y - b_y)**2 + (a_z - b_z)**2)

def get_reward(rewards, pos, low_x, high_x, res, j):

    if check_range(pos, low_x + 0*res, low_x + 50*res):
        reward = -5
    # Closer to receive cup -> less negative reward
    elif check_range(pos, low_x + 51*res, low_x + 100*res):
        reward = -3
    # Where we want the source cup to be -> positive reward
    elif check_range(pos, low_x + 201*res, low_x + 245*res):
        reward = 2
    # Dead center -> very positive reward
    elif check_range(pos, low_x + 246*res, low_x + 255*res):
        reward = 5
    # Where we want the source cup to be -> positive reward
    elif check_range(pos, low_x + 256*res, low_x + 300*res):
        reward = 2 
    # Outer bounds -> less negative reward
    elif check_range(pos, low_x + 301*res, low_x + 400*res):
        reward = -3
    # Outer bounds -> negative reward
    elif check_range(pos, low_x + 401*res, low_x + 500*res):
        reward = get_distance_3d()
    else: 
        reward = -5

def normalize(val, min_val, max_val):
    #zi = (xi - min(x)) / max(x) - min(x)) * Q, where Q = state_space_size (max value in range)
    #print(f"val: {val}, min_val: {min_val*1000}, max_val: {max_val*1000}")
    norm_val = (val - min_val) / (max_val - min_val) * 100# state space includes source cup position(500) and current frame number(1000)
    #print(f"NORMALIZED VALUE: {norm_val}, ROUNDED: {int(norm_val)}")
    return norm_val

def check_range(val, low, high):
    #print(f"value: {val}, range: {low} to {high}")
    return low <= val <= high


class CubesCups(Env):
    def __init__(self):
        #500 for source_x, 100 for speed
        self.bins = (500, 100)
        # [-2, -1, 0, 1, 2]
        self.action_space = Discrete(5)
        # Observation space has to be discrete in order to work with Q-table
        # Discretize state space values
        self.source_x_low = -0.80 + low
        self.source_x_high = -0.80 + high
        self.velReal_low = -0.7361215932167728
        self.velReal_high = 0.8499989492543077 

        self.lower_bounds = np.array([self.source_x_low, self.velReal_low])
        self.upper_bounds = np.array([self.source_x_high, self.velReal_high])
        self.observation_space = Box(low, high) 

        # Initialize Q-table
        #q_table = np.zeros((state_space_size, action_space_size))
        self.q_table = np.zeros(self.bins + (self.action_space.n,))

        self.state = None
        self.total_frames = 1000
        #j in range(velReal.shape[0])
        self.current_frame = 0
        self.speed = None
        self.offset = None

        self.clientID = None
        self.source_cup_handle = None
        self.receive_cup_handle = None
        self.cubes_handles = []
        
        self.source_cup_position = None
        self.receive_cup_position = None
        self.cubes_positions = []
        self.center_position = None
        self.joint_position = None


    def step(self, action):

        # Calculate reward as the negative distance between the source up and the receiving cup
        reward = get_distance_3d(self.source_cup_position, self.receive_cup_position) * 1000

        r_x, r_y = self.receive_cup_position[0], self.receive_cup_position[1]
        flag = 0
        for cube_pos in self.cubes_positions:
            cube_x, cube_y, cube_z = cube_pos[0], cube_pos[1], cube_pos[2]
            # Cubes are in the receive cube
            if (cube_x < r_x + 0.04 and cube_x > r_x - 0.04) and (cube_y < r_y + 0.04 and cube_y > r_y - 0.04 \
                    and (cube_z > 0.20 and cube_z < 0.60)):
                flag += 1                
            elif((cube_x < r_x + 0.04 and cube_x > r_x - 0.04) and (cube_z > 0.60)):
            # Cubes are within the x bounds of the receive cup (on trajectory to make it into the receive cup)
                reward += 250
        # Give big reward for both cubes landing in the receive cup            
        if flag == 2:
            reward += 1000
    
        # Rotate cup based on speed value
        self.rotate_cup()
        # Move cup laterally based on selected action in Q-table
        self.move_cup(action)

        # Get the position of both cubes
        for cube, i in zip(self.cubes_handles, range(0, 2)):
            returnCode, cube_position = sim.simxGetObjectPosition(
                self.clientID, cube, -1, sim.simx_opmode_streaming)
            self.cubes_positions[i] = cube_position

        '''if self.current_frame > 10 and self.joint_position > 0:
            done = True
        else:
            done = False'''

        # Keep track of current frame number
        self.current_frame += 1

        info = {}
        # Get state after step completed, to return
        self.state = np.array([self.source_cup_position[0], self.speed])

        #Return step info
        return self.state, reward, info

    def reset(self, rng):
        # Reset source cup position (sets self.source_cup_position, 
        # self.receive_cup_position, and self.center_positions)
        self.cubes_handles = []
        self.cubes_positions = []
        self.set_random_cup_position(rng)
        # Current frame is j
        self.current_frame = 0
        # Speed is velReal[j]
        self.speed = 0
        #self.clientID = None
        self.joint_position = None

        # Update state for new episode
        self.state = np.array([self.source_cup_position[0], self.speed])

        return self.state

    def render(self, action, reward):
        pass

    def start_simulation(self):
        ''' Function to communicate with Coppelia Remote API and start the simulation '''
        sim.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = sim.simxStart('127.0.0.1', 19000, True, True, 5000,
                                5)  # Connect to CoppeliaSim
        if self.clientID != -1:
            print('Connected to remote API server')
        else:
            print("fail")
            sys.exit()

        returnCode = sim.simxSynchronous(self.clientID, True)
        returnCode = sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)

        if returnCode != 0 and returnCode != 1:
            print("something is wrong")
            print(returnCode)
            exit(0)

        triggerSim(self.clientID)

        # get the handle for the source container
        res, self.source_cup_handle = sim.simxGetObjectHandle(self.clientID, 'joint',
                                            sim.simx_opmode_blocking)
        res, self.receive_cup_handle = sim.simxGetObjectHandle(self.clientID, 'receive',
                                            sim.simx_opmode_blocking)
        # start streaming the data
        returnCode, original_position = sim.simxGetObjectPosition(
            self.clientID, self.source_cup_handle, -1, sim.simx_opmode_streaming)
        returnCode, original_position = sim.simxGetObjectPosition(
            self.clientID, self.receive_cup_handle, -1, sim.simx_opmode_streaming)
        returnCode, original_position = sim.simxGetJointPosition(
            self.clientID, self.source_cup_handle, sim.simx_opmode_streaming)

        #get object handles
        self.get_cubes()

    def stop_simulation(self):
        ''' Function to stop the episode '''
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
        sim.simxFinish(self.clientID)
        print("Simulation stopped.")

    def get_cubes(self):
        # Drop blocks in source container
        triggerSim(self.clientID)
        number_of_blocks = 2
        print('Initial number of blocks=', number_of_blocks)
        setNumberOfBlocks(self.clientID,
                        blocks=number_of_blocks,
                        typeOf='cube',
                        mass=0.002,
                        blockLength=0.025,
                        frictionCube=0.06,
                        frictionCup=0.8)
        triggerSim(self.clientID)

        # Get handles of cubes created
        obj_type = "Cuboid"
        for cube in range(number_of_blocks):
            res, obj_handle = sim.simxGetObjectHandle(self.clientID,
                                                    f'{obj_type}{cube}',
                                                    sim.simx_opmode_blocking)
            self.cubes_handles.append(obj_handle)

        triggerSim(self.clientID)

        for cube in self.cubes_handles:
            # get the starting position of cubes
            returnCode, cube_position = sim.simxGetObjectPosition(
                self.clientID, cube, -1, sim.simx_opmode_streaming)
            self.cubes_positions.append(cube_position)
        
        # Give time for the cubes to finish falling
        wait_(self.clientID)

    def rotation_velocity(self, rng):
        ''' Set rotation velocity randomly, rotation velocity is a composition of two sinusoidal velocities '''
        #Sinusoidal velocity
        forward = [-0.3, -0.35, -0.4]
        backward = [0.75, 0.8, 0.85]
        freq = 60
        ts = np.linspace(0, 1000 / freq, 1000)
        velFor = rng.choice(forward) * np.sin(2 * np.pi * 1 / 20 * ts)
        velBack = rng.choice(backward) * np.sin(2 * np.pi * 1 / 10 * ts)

        #print(f"velFor: {velFor}\n\n\nvelBack: {velBack}")
        velSin = velFor
        idxFor = np.argmax(velFor > 0)
        velSin[idxFor:] = velBack[idxFor:]
        velReal = velSin
        #print(f"velFor min: {np.min(velFor)} velBack max: {np.max(velBack)}")
        print(f"velreal min: {np.min(velReal)}, velreal max: {np.max(velReal)}")

        return velReal

    def set_random_cup_position(self, rng):
        # Print source cup position before random move
        returnCode, self.source_cup_position = sim.simxGetObjectPosition(
            self.clientID, self.source_cup_handle, -1, sim.simx_opmode_buffer)
        print(f'Source Cup Initial Position:{self.source_cup_position}')

        # Move cup along x axis
        global low, high
        rng_var = rng.random()
        self.offset = low + (high - low) * rng_var
        self.source_cup_position[0] = self.source_cup_position[0] + self.offset

        returnCode = sim.simxSetObjectPosition(self.clientID, self.source_cup_handle, -1, self.source_cup_position,
                                            sim.simx_opmode_blocking)
        triggerSim(self.clientID)

        returnCode, self.source_cup_position = sim.simxGetObjectPosition(
            self.clientID, self.source_cup_handle, -1, sim.simx_opmode_buffer)
        print(f'Pouring cup randomly set position:{self.source_cup_position}')

        returnCode, self.receive_cup_position = sim.simxGetObjectPosition(
            self.clientID, self.receive_cup_handle, -1, sim.simx_opmode_buffer)
        print(f'Receiving cup position:{self.receive_cup_position}')

        obj_type = "Cuboid"
        number_of_blocks = 2
        for cube in range(number_of_blocks):
            res, obj_handle = sim.simxGetObjectHandle(self.clientID,
                                                    f'{obj_type}{cube}',
                                                    sim.simx_opmode_blocking)
            self.cubes_handles.append(obj_handle)

        for cube in self.cubes_handles:
            # get the starting position of cubes
            returnCode, cube_position = sim.simxGetObjectPosition(
                self.clientID, cube, -1, sim.simx_opmode_streaming)
            self.cubes_positions.append(cube_position)

        wait_(self.clientID)

        self.center_position = self.source_cup_position[0]

    def get_cup_offset(self, rng):
        return self.offset

    def step_chores(self):
        # 60Hz
        triggerSim(self.clientID)
        # Make sure simulation step finishes
        returnCode, pingTime = sim.simxGetPingTime(self.clientID)

    def rotate_cup(self):
        ''' Function to rotate cup '''
        errorCode = sim.simxSetJointTargetVelocity(self.clientID, self.source_cup_handle, self.speed,
                                                sim.simx_opmode_oneshot)
        returnCode, self.joint_position = sim.simxGetJointPosition(self.clientID, self.source_cup_handle,
                                                        sim.simx_opmode_buffer)

    def move_cup(self, action):
        ''' Function to move the pouring cup laterally during the rotation '''
        global low, high
        resolution = 0.001
        move_x = resolution * action
        movement = self.source_cup_position[0] + move_x
        if self.center_position + low < movement < self.center_position + high:
            self.source_cup_position[0] = movement
            returnCode = sim.simxSetObjectPosition(self.clientID, self.source_cup_handle, -1,
                                                self.source_cup_position,
                                                sim.simx_opmode_blocking)

    def discretize_state(self, obs):
        print(f"observation to be discretized: {obs}")
        discretized = list()
        for i in range(len(obs)):
            scaling = ((obs[i] + abs(self.lower_bounds[i]))
                / (self.upper_bounds[i] - self.lower_bounds[i]))
            new_obs = int(round((self.bins[i] - 1) * scaling))
            new_obs = min(self.bins[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def update_q(self, state, action, new_state, reward):
        # Add 2 to action variable to map correctly to Q-table indices
        action += 2
        #print(f"action update q: {action}")
        # Update Q-table for Q(s,a)
        #q_table[state][action] = q_table[state][action] * (1 - learning_rate) + \
        #            learning_rate * (reward + discount_rate * np.max(q_table[new_state]))
        #print(f"new_state: {new_state}, state: {state}")
        #print(f"q_table[new_state]: {self.q_table[new_state]}, \nq_table[state][action]: {self.q_table[state][action]}")
        self.q_table[state][action] += (learning_rate *
                    (reward
                    + discount_rate * np.max(self.q_table[new_state])
                    - self.q_table[state][action]))
    
    def pick_action(self, state):
        #Exploration-exploitation trade-off
        exploration_rate_threshold = np.random.uniform(0, 1)
        # If exploitation is picked, select action where max Q-value exists within state's row in the q-table
        if exploration_rate_threshold > exploration_rate:
            # Select the largest Q-value
            #norm = normalize(norm_state, norm_low, norm_high)# -0.85 + low + offset, -0.85 + high + offset)
            return np.argmax(self.q_table[state]) - 2
        # If exploration is picked, select a random action from the current state's row in the q-table
        else:
            return self.action_space.sample() - 2
            #print(f"action: {action}")

    '''def get_state(self):
        #self.state = math.floor(self.source_cup_position[0] * -1000) + (self.current_frame * 5)
        state = np.array([self.source_cup_position[0], self.speed])
        return state'''
