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
np.set_printoptions(threshold=sys.maxsize)
import sim


# Max movement along X
low, high = -0.05, 0.05


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


class CubesCups(Env):
    def __init__(self, num_episodes=500,
                min_lr=0.1, min_epsilon=0.1, 
                discount=0.99, decay=25):

        #1000 for source_x, 100 for speed
        self.bins = (1000, 100)
        # [-2, -1, 0, 1, 2]
        self.action_space = Discrete(5)
        # Observation space has to be discrete in order to work with Q-table
        # Discretize state space values
        self.source_x_low = -0.95 # -0.85 + low
        self.source_x_high = -0.75 # -0.85 + high
        self.velReal_low = -0.7361215932167728
        self.velReal_high = 0.8499989492543077 

        print(f"min: {self.source_x_low}, max: {self.source_x_high}")

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

        self.clientID = None
        self.source_cup_handle = None
        self.receive_cup_handle = None
        self.cubes_handles = []
        
        self.source_cup_position = None
        self.receive_cup_position = None
        self.cubes_positions = []
        self.center_position = None
        self.joint_position = None

        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

        self.cur_distance = None
        self.prev_distance = None

    def step(self, action):

        # Calculate reward as the negative distance between the source up and the receiving cup
        #self.prev_distance = self.cur_distance
        #self.cur_distance = get_distance_3d(self.source_cup_position, self.receive_cup_position) * 1000
        '''
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
        '''

        '''if self.cur_distance < self.prev_distance:
            reward = 1
        elif self.cur_distance == self.prev_distance:
            reward = 0
        else:
            reward = -1'''

        reward = get_distance_3d(self.source_cup_position, self.receive_cup_position)
    
        # Rotate cup based on speed value
        self.rotate_cup()
        # Move cup laterally based on selected action in Q-table
        self.move_cup(action)

        # Get the position of both cubes
        for cube, i in zip(self.cubes_handles, range(0, 2)):
            returnCode, cube_position = sim.simxGetObjectPosition(
                self.clientID, cube, -1, sim.simx_opmode_streaming)
            self.cubes_positions[i] = cube_position

        done = bool(self.current_frame > 10 and self.joint_position > 0)

        # Keep track of current frame number
        self.current_frame += 1

        info = {}
        # Get state after step completed, to return
        self.state = np.array([self.source_cup_position[0], self.speed])

        #Return step info
        return self.state, reward, done, info

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
        # Set cur distance between source cup and receive cup for reward calculation
        self.cur_distance = get_distance_3d(self.source_cup_position, self.receive_cup_position) * 1000

        return self.state

    def train(self):
        rewards_all_episodes = []
        rewards_filename = "rewards_history.txt"
        q_table_filename = "q_table.npy"

        # Load q_table.pkl for updating, if it exists
        '''if(os.path.exists(q_table_filename)):
            with open(q_table_filename, 'rb') as f:
                self.q_table = np.load(f)
            print("Q-table loaded.")'''

        for e in range(self.num_episodes):
            print(f"Episode {e+1}:")

            # Set rotation velocity randomly
            rng = np.random.default_rng()
            velReal = self.rotation_velocity(rng)

            # Start simulation, process objects/handles
            self.start_simulation()

            # Set initial position of the source cup and initialize state
            state = self.discretize_state(self.reset(rng))

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False
            # Keep track of rewards for each episode, to be stored in a list for analysis
            rewards_current_episode = 0

            for j in range(velReal.shape[0]):
                self.step_chores()
                # Initialize the speed of the source cup at this frame
                self.speed = velReal[j]

                # Pick next action, greedy epsilon
                action = self.pick_action(state)

                # Take next action
                obs, reward, done, _ = self.step(action)

                # Normalize speed value for q_table
                self.speed = self.normalize(velReal[j], self.velReal_low, self.velReal_high)
                # Calculate new state, continuous -> discrete
                new_state = self.discretize_state(obs)
            
                # Update Q-table for Q(s,a)
                print(f"source pos: {self.source_cup_position[0]}")
                print(f"STATE: {state}, new_state: {new_state} REWARD: {reward}, ACTION: {action}")
                self.update_q(state, action, new_state, reward)

                # Update state variable
                state = new_state

                # Keep track of rewards for current episode
                rewards_current_episode += reward
                
                # Break if cup goes back to vertical position
                if done:
                    break
            #end for

            # Stop simulation
            self.stop_simulation()

            # Append current episode's reward to total rewards list for later
            rewards_all_episodes.append(rewards_current_episode)

        # Save the Q-table to a .npy file
        with open(q_table_filename, 'wb') as f:
            np.save(f, self.q_table)
        print(f"Q-table saved to {q_table_filename}")

        # Append this episodes rewards to a .txt file
        with open(rewards_filename, 'wb') as f:
            np.savetxt(f, rewards_all_episodes)
        print(f"Saved rewards history to {rewards_filename}")

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
        move_x = low + (high - low) * rng.random()
        self.source_cup_position[0] = self.source_cup_position[0] + move_x

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
        #print(f"observation to be discretized: {obs}")
        discretized = list()
        for i in range(len(obs)):
            scaling = ((obs[i] + abs(self.lower_bounds[i]))
                / (self.upper_bounds[i] - self.lower_bounds[i]))
            new_obs = int(round((self.bins[i] - 1) * scaling))
            new_obs = min(self.bins[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def normalize(self, val, min_val, max_val):
        # norm_i = (x_i - min(x)) / max(x) - min(x)) * Q
        # Normalize values between 0 and Q = 100
        norm_val = (val - min_val) / (max_val - min_val) * 100
        return norm_val

    def update_q(self, state, action, new_state, reward):
        # Add 2 to action variable to map correctly to Q-table indices
        action += 2
        # Update Q-table for Q(s,a)
        self.q_table[state][action] = self.q_table[state][action] * (1 - self.learning_rate) + \
                    self.learning_rate * (reward + self.discount * np.max(self.q_table[new_state]))

        '''self.q_table[state][action] += (self.learning_rate *
                    (reward
                    + self.discount * np.max(self.q_table[new_state])
                    - self.q_table[state][action]))'''
    
    def pick_action(self, state):
        #Exploration-exploitation trade-off
        # If exploration is picked, select a random action from the current state's row in the q-table
        if (np.random.random() < self.epsilon):
            # Select the largest Q-value
            return self.action_space.sample() - 2
        # If exploitation is picked, select action where max Q-value exists within state's row in the q-table
        else:
            return np.argmax(self.q_table[state]) - 2

    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))
    
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def run(self):
        t = 0
        done = False
        q_table_filename = "q_table.npy"

        # Load q_table.pkl for updating, if it exists
        if(os.path.exists(q_table_filename)):
            with open(q_table_filename, 'rb') as f:
                self.q_table = np.load(f)
            print("Q-table loaded.")
        
        # Set rotation velocity randomly
        rng = np.random.default_rng()
        velReal = self.rotation_velocity(rng)

        # Start simulation, process objects/handles
        self.start_simulation()

        # Set initial position of the source cup and initialize state
        state = self.discretize_state(self.reset(rng))

        for j in range(velReal.shape[0]):
            self.step_chores()
            # Initialize the speed of the source cup at this frame
            self.speed = velReal[j]

            # Pick next action, greedy epsilon
            action = np.argmax(self.q_table[state]) - 2

            # Take next action
            obs, reward, done, _ = self.step(action)

            # Calculate new state, continuous -> discrete
            new_state = self.discretize_state(obs)

            print(f"STATE: {state}, new_state: {new_state} REWARD: {reward}, ACTION: {action}")
        
            # Update state variable
            state = new_state
            
            # Break if cup goes back to vertical position
            if done:
                break
        #end for
        
        # Stop simulation
        self.stop_simulation()
        
        return t