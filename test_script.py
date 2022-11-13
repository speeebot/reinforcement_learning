"""
    This code communicated with the coppeliaSim simulation and simulates pouring aiming to a destination container
    
    WINDOWS:
    ./Applications/coppeliaSim.app/Contents/MacOSMacOS/coppeliaSim -gREMOTEAPISERVERSERVICE_19000_FALSE_TRUE ~/path/to/file/2cups_Intro_to_AI.ttt

    MACOS:
    cd ~/Applications/coppeliaSim.app/Contents/MacOS
    ./coppeliaSim -gREMOTEAPISERVERSERVICE_19000_FALSE_TRUE ~/Documents/school/fall22/intro_ai/project2_part1/2cups_Intro_to_AI.ttt

"""

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

learning_rate = 0.1
discount_rate = 0.99
#exploration_rate = 1.0 #0.06079027722859994 #0.1466885449377839
max_exploration_rate = 1.0 #0.06079027722859994 #0.1466885449377839 #0.37786092411182526
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

# Set state space and action space sizes
state_space_size = 100 # Resolution of 0.0002 for 100 per fifth section of the x-axis (-0.5 to 0.5) -> (-500 to 500) -> totaling 1000
action_space_size = 5 # [-2, -1, 0, 1, 2]

num_episodes = 100

# Actions to move cup laterally
actions = [-2, -1, 0, 1, 2]

# Lists of possible reward structures
rewards = [[-5, -3, 2, 5, 2, -3, -5],
            [-3, 2, 5, 2, -3, -5, -5],
            [2, 5, 2, -3, -5, -5, -5],
            [5, 2, -3, -5, -5, -5, -5],
            [-5, -5, -3, 2, 5, 2, -3],
            [-5, -5, -5, -3, 2, 5, 2],
            [-5, -5, -5, -5, -3, 2, 5]]

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


def rotation_velocity(rng):
    ''' Set rotation velocity randomly, rotation velocity is a composition of two sinusoidal velocities '''
    #Sinusoidal velocity
    forward = [-0.3, -0.35, -0.4]
    backward = [0.75, 0.8, 0.85]
    freq = 60
    ts = np.linspace(0, 1000 / freq, 1000)
    velFor = rng.choice(forward) * np.sin(2 * np.pi * 1 / 20 * ts)
    velBack = rng.choice(backward) * np.sin(2 * np.pi * 1 / 10 * ts)
    velSin = velFor
    idxFor = np.argmax(velFor > 0)
    velSin[idxFor:] = velBack[idxFor:]
    velReal = velSin
    return velReal


def start_simulation():
    ''' Function to communicate with Coppelia Remote API and start the simulation '''
    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1', 19000, True, True, 5000,
                             5)  # Connect to CoppeliaSim
    if clientID != -1:
        print('Connected to remote API server')
    else:
        print("fail")
        sys.exit()

    returnCode = sim.simxSynchronous(clientID, True)
    returnCode = sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    if returnCode != 0 and returnCode != 1:
        print("something is wrong")
        print(returnCode)
        exit(0)

    triggerSim(clientID)

    # get the handle for the source container
    res, pour = sim.simxGetObjectHandle(clientID, 'joint',
                                        sim.simx_opmode_blocking)
    res, receive = sim.simxGetObjectHandle(clientID, 'receive',
                                           sim.simx_opmode_blocking)
    # start streaming the data
    returnCode, original_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_streaming)
    returnCode, original_position = sim.simxGetObjectPosition(
        clientID, receive, -1, sim.simx_opmode_streaming)
    returnCode, original_position = sim.simxGetJointPosition(
        clientID, pour, sim.simx_opmode_streaming)

    return clientID, pour, receive


def stop_simulation(clientID):
    ''' Function to stop the episode '''
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    sim.simxFinish(clientID)


def get_object_handles(clientID, pour):
    # Drop blocks in source container
    triggerSim(clientID)
    number_of_blocks = 2
    print('Initial number of blocks=', number_of_blocks)
    setNumberOfBlocks(clientID,
                      blocks=number_of_blocks,
                      typeOf='cube',
                      mass=0.002,
                      blockLength=0.025,
                      frictionCube=0.06,
                      frictionCup=0.8)
    triggerSim(clientID)

    # Get handles of cubes created
    object_shapes_handles = []
    obj_type = "Cuboid"
    for obj_idx in range(number_of_blocks):
        res, obj_handle = sim.simxGetObjectHandle(clientID,
                                                  f'{obj_type}{obj_idx}',
                                                  sim.simx_opmode_blocking)
        object_shapes_handles.append(obj_handle)

    triggerSim(clientID)

    for obj_handle in object_shapes_handles:
        # get the starting position of source
        returnCode, obj_position = sim.simxGetObjectPosition(
            clientID, obj_handle, -1, sim.simx_opmode_streaming)

    returnCode, position = sim.simxGetJointPosition(clientID, pour,
                                                    sim.simx_opmode_buffer)
    returnCode, obj_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_buffer)
    print(f'Pouring Cup Initial Position:{obj_position}')
    # Give time for the cubes to finish falling
    _wait(clientID)
    return object_shapes_handles, obj_position


def set_cup_initial_position(clientID, pour, receive, cup_position, rng):

    # Move cup along x axis
    global low, high
    move_x = low + (high - low) * rng.random()
    cup_position[0] = cup_position[0] + move_x

    returnCode = sim.simxSetObjectPosition(clientID, pour, -1, cup_position,
                                           sim.simx_opmode_blocking)
    triggerSim(clientID)
    returnCode, pour_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_buffer)
    print(f'Pouring Cup Moved Position:{pour_position}')
    returnCode, receive_position = sim.simxGetObjectPosition(
        clientID, receive, -1, sim.simx_opmode_buffer)
    print(f'Receiving Cup Position:{receive_position}')
    return pour_position, receive_position


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


def move_cup(clientID, pour, action, cup_position, center_position):
    ''' Function to move the pouring cup laterally during the rotation '''

    global low, high
    resolution = 0.001
    move_x = resolution * action
    movement = cup_position[0] + move_x
    if center_position + low < movement < center_position + high:
        cup_position[0] = movement
        returnCode = sim.simxSetObjectPosition(clientID, pour, -1,
                                               cup_position,
                                               sim.simx_opmode_blocking)


def rotate_cup(clientID, speed, pour):
    ''' Function to rotate cup '''
    errorCode = sim.simxSetJointTargetVelocity(clientID, pour, speed,
                                               sim.simx_opmode_oneshot)
    returnCode, position = sim.simxGetJointPosition(clientID, pour,
                                                    sim.simx_opmode_buffer)
    return position


def _wait(clientID):
    for _ in range(60):
        triggerSim(clientID)


def update_q_table(q_table, state, action, new_state, reward, low, high, j):
    # Add 2 to action variable to map correctly to Q-table indices
    action += 2
    # Normalize state values for q_table
    norm_state = normalize(state, low, high)
    norm_new_state = normalize(new_state, low, high)
    print(f"state values: {state}, {new_state}, normalized state values: {norm_state}, {norm_new_state}")
    # Update Q-table for Q(s,a)
    q_table[norm_state, action] = q_table[norm_state, action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_rate * np.max(q_table[norm_new_state, :]))

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
    #print(f"val: {val}, min_val: {min_val}, max_val: {max_val}")
    norm_val = ((val - min_val) / (max_val - min_val)) * state_space_size # state space includes source cup position(500) and current frame number(1000)
    #print(f"NORMALIZED VALUE: {norm_val}, ROUNDED: {int(norm_val)}")
    return int(norm_val)

def check_range(val, low, high):
    #print(f"value: {val}, range: {low} to {high}")
    return low <= val <= high

def main():
    q_table_filename = "q_table.txt"

        
    # Set rotation velocity randomly
    rng = np.random.default_rng()
    velReal = rotation_velocity(rng)

    # Start simulation
    clientID, source_cup, receive_cup = start_simulation()
    object_shapes_handles, source_cup_position = get_object_handles(clientID, source_cup)

    # Get initial position of the cups
    source_position, receive_position = set_cup_initial_position(clientID, source_cup, receive_cup, source_cup_position, rng)
    _wait(clientID)

    center_position = source_position[0]

    # Get starting state
    old_cubes_position, old_source_cup_position = get_state(object_shapes_handles,
                                                clientID, source_cup)
    
    state = old_source_cup_position[0]

    if(os.path.exists(q_table_filename)):
        q_table = np.loadtxt(q_table_filename)
    print("Q-table loaded.")

    rewards_current_episode = 0

    for j in range(velReal.shape[0]):
        # 60HZ
        triggerSim(clientID)
        # Make sure simulation step finishes
        returnCode, pingTime = sim.simxGetPingTime(clientID)

        #initialize the speed of the source cup at this frame
        speed = velReal[j]

        # Get current state
        cubes_position, source_cup_position = get_state(object_shapes_handles,
                                                clientID, source_cup)
        
        # Update state
        new_state = source_cup_position[0]

        source_low_x = center_position + low
        source_high_x = center_position + high
                                                        
        # Select the largest Q-value
        norm_state = normalize(state, source_low_x, source_high_x)
        action = np.argmax(q_table[norm_state,:]) - 2
        print(action)

        # Rotate cup
        position = rotate_cup(clientID, speed, source_cup)

        # call move_cup function
        move_cup(clientID, source_cup, action, source_cup_position, center_position)

        #update state variable
        state = new_state
        old_source_cup_position = source_cup_position
        
        #print(f"Step {j}, Cubes position: {cubes_position}, Action taken: {action}")

        # Break if cup goes back to vertical position
        if j > 10 and position > 0:
            break

    #end for
        
    # Stop simulation
    stop_simulation(clientID)
    print("Simulation stopped.")

if __name__ == '__main__':

    main()
