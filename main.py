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

num_episodes = 300

#actions to move cup laterally
actions = [-2, -1, 0, 1, 2]

# Initialize Q-table
q_table = dict()


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


def update_q_table(state, action, new_state, reward):
    # Create an ID for each state
    state_id = '/'.join([str(x) for x in state])
    new_state_id = '/'.join([str(x) for x in new_state])

    if state_id not in q_table:
        q_table[state_id] = {action: 0 for action in actions}
    if new_state_id not in q_table:
        q_table[new_state_id] = {action: 0 for action in actions}


    if state_id in q_table:
        print(f"{state_id}, {action}")
        print(f"HELLO: {q_table[state_id].values()}")
        q_table[state_id][action] = q_table[state_id][action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * max(q_table[new_state_id].values()))
                                                    #max(p[new_state_id] for p in q_table.values())
    else:
        value = 0 * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * max(q_table[new_state_id].values()))

        q_table[state_id] = {action: value}

    
    return state_id, new_state_id


def main():
    rewards_all_episodes = []
    exploration_rate = 1.0
    global q_table

    for episode in range(num_episodes):
        print(f"Episode {episode}:")
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

        #map of actions to Q-table indices
        actions_map = {0:-2, 1:-1, 2:0, 3:1, 4:2}

        # Get starting state
        cubes_position, source_cup_position = get_state(object_shapes_handles,
                                                    clientID, source_cup)
        
        #state = math.floor(abs(velReal[0] + source_cup_position[0]) * 10000)
        state = [round(velReal[0], 4), round(source_cup_position[0], 4), round(cubes_position[0][0], 4), \
            round(cubes_position[0][1], 4), round(cubes_position[0][2], 4), round(cubes_position[1][0], 4), \
            round(cubes_position[1][1], 4), round(cubes_position[1][2], 4)]
        
        state_id = '/'.join([str(x) for x in state])
        print(state_id)

        #print(f"source cup pos: {source_cup_position}, state: {state}")
        #print(f"cubes pos: {cubes_position}, source_cup_pos: {source_cup_position}")

        #load q_table.txt
        if(os.path.exists("q_table.pkl")):
            #q_table = np.loadtxt("q_table4.txt")
            with open('q_table.pkl', 'rb') as f:
                q_table = pickle.load(f)
            print("Q-table loaded.")

        rewards_current_episode = 0

        for j in range(velReal.shape[0]):
            # 60HZ
            triggerSim(clientID)
            # Make sure simulation step finishes
            returnCode, pingTime = sim.simxGetPingTime(clientID)

            #initialize the speed of this frame
            speed = velReal[j]

            # Get current state
            cubes_position, source_cup_position = get_state(object_shapes_handles,
                                                    clientID, source_cup)

            #print(f"Step {j}, Cubes position: {cubes_position}, Cup position: {cup_position}")

            # Initialize variables for x,y,z of source, receive, and both cubes
            receive_x, receive_y, receive_z = receive_position[0], receive_position[1], receive_position[2]
            source_x, source_y, source_z = source_cup_position[0], source_cup_position[1], source_cup_position[2]
            cube1_x, cube1_y, cube1_z = cubes_position[0][0], cubes_position[0][1], cubes_position[0][2]
            cube2_x, cube2_y, cube2_z = cubes_position[1][0], cubes_position[1][1], cubes_position[1][2]

            # Receiving cup radius is around 0.05, check if cubes made it into receiving cup
            '''flag = 0
            if (cube_x < receive_x + 0.05 and cube_x > receive_x - 0.05) and (cube_y < receive_y + 0.05 and cube_y > receive_y - 0.05):
                    #Both cubes are within the radius of the receiving cup
                    new_state = math.floor(abs(speed + source_x) * 1000 * 2)
            else:
                #Atleast one cube is not within the radius of the receiving cup
                new_state = math.floor(abs(speed + source_x) * 1000 * 1)
                flag = 1'''
                #negative distance between cubes and receive cup, bigger reward is better -> closer to receive cup
            reward = -math.sqrt((cube1_x - receive_x)**2 + (cube1_y - receive_y)**2 + (cube1_z - receive_z)**2) + \
                -math.sqrt((cube2_x - receive_x)**2 + (cube2_y - receive_y)**2 + (cube2_z - receive_z)**2) 
            #print(f"STATE: {state}, REWARD: {reward}")
                
            # Update state
            #new_state = math.floor(abs(speed + source_x) * 10000)
            new_state = [round(speed, 4), round(source_cup_position[0], 4), round(cubes_position[0][0], 4), \
                    round(cubes_position[0][1], 4), round(cubes_position[0][2], 4), round(cubes_position[1][0], 4), \
                    round(cubes_position[1][1], 4), round(cubes_position[1][2], 4)]

            exploration_rate_threshold = np.random.uniform(0, 1)
            # If exploitation is picked, select action where max Q-value exists within state's row in the q-table
            if exploration_rate_threshold > exploration_rate:
                # Select the largest Q-value
                if state_id in q_table:
                    print(f"KEY ERROR?: {q_table[state_id]}, type: {type(q_table[state_id][0])}")
                    action = max(q_table[state_id], key=q_table[state_id].get)
                    print(f"EXPLOITED action selected: {action}")
                else:   
                    action = 0
                    print(f"NO ACTION SELECTED: 0")
            # If exploration is picked, select a random action from the current state's row in the q-table
            else:
                #action = random.choice(q_table[state,:])
                action = random.choice(actions)
                print(f"random action selected: {action}")

            # Rotate cup
            position = rotate_cup(clientID, speed, source_cup)
            #print(position)

            #move cup randomly (for part 1)
            #action = np.random.choice(actions)
            # call move_cup function
            move_cup(clientID, source_cup, action, source_cup_position, center_position)

            # Add 2 to action to properly map to Q-table indices
            #action += 2
            #print(f"state: {state}, action: {action}")

            #update Q-table for Q(s,a)
            state_id, new_state_id = update_q_table(state, action, new_state, reward)
            print(f"STATE: {state_id}, ACTION: {action}")

            #update state variable
            state = new_state
            rewards_current_episode += reward
            
            #print(f"Step {j}, Cubes position: {cubes_position}, Action taken: {action}")

            # Break if cup goes back to vertical position
            if j > 10 and position > 0:
                break

        #end for

        #Exploration rate decay
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
        
        #Append current episode's reward to total rewards list for later
        rewards_all_episodes.append(rewards_current_episode)
        #print(f"Cubes final position: {cubes_position}")

        # Stop simulation
        stop_simulation(clientID)
        print("Simulation stopped.")

        #np.savetxt('q_table4.txt', q_table)
        #print("Q-table4 saved.")
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(q_table, f)
    
    np.savetxt('rewards_listv4.txt', rewards_all_episodes)
    print(f"final_exploration_rate: {exploration_rate}")
    print("Saved rewards for each episode to rewards_listv4_2.txt")

if __name__ == '__main__':

    main()
