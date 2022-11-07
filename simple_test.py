"""
    This code communicated with the coppeliaSim simulation and simulates pouring aiming to a destination container
    
    ./Applications/coppeliaSim.app/Contents/MacOSMacOS/coppeliaSim -gREMOTEAPISERVERSERVICE_19000_FALSE_TRUE ~/path/to/file/2cups_Intro_to_AI.ttt
"""

import sys

sys.path.append('MacAPI')
import sim


def triggerSim(clientID):
    e = sim.simxSynchronousTrigger(clientID)
    step_status = 'successful' if e == 0 else 'error'
    # print(f'Finished Step {step_status}')


def stop_simulation(clientID):
    ''' Function to stop the episode '''
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    sim.simxFinish(clientID)


if __name__ == "__main__":
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

    triggerSim(clientID)
    res, receive = sim.simxGetObjectHandle(clientID, 'receive',
                                           sim.simx_opmode_blocking)

    triggerSim(clientID)
    # get the handle for the source container
    res, sphere = sim.simxGetObjectHandle(clientID, 'Sphere',
                                          sim.simx_opmode_blocking)

    # start streaming the data
    returnCode, original_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_streaming)
    returnCode, original_position = sim.simxGetObjectPosition(
        clientID, receive, -1, sim.simx_opmode_streaming)
    returnCode, original_position = sim.simxGetJointPosition(
        clientID, pour, sim.simx_opmode_streaming)
    returnCode, original_position = sim.simxGetObjectPosition(
        clientID, sphere, -1, sim.simx_opmode_streaming)

    triggerSim(clientID)

    returnCode, source_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_buffer)
    print(f'Pouring Cup Initial Position:{source_position}')
    triggerSim(clientID)
    returnCode, receiver_position = sim.simxGetObjectPosition(
        clientID, receive, -1, sim.simx_opmode_buffer)
    print(f'Receiving Cup Initial Position:{receiver_position}')

    triggerSim(clientID)

    # change position of the source cup
    #source_position[0] -= 0.1
    source_position[0] = receiver_position[0] - 0.05
    returnCode = sim.simxSetObjectPosition(clientID, pour, -1, source_position,
                                           sim.simx_opmode_blocking)

    triggerSim(clientID)
    returnCode, source_position = sim.simxGetObjectPosition(
        clientID, pour, -1, sim.simx_opmode_buffer)
    print(f'Pouring Cup Moved Position:{source_position}')

    triggerSim(clientID)
    returnCode, sphere_position = sim.simxGetObjectPosition(
        clientID, sphere, -1, sim.simx_opmode_buffer)
    print(f'Sphere Initial Position:{sphere_position}')

    triggerSim(clientID)
    sphere_position[2] = receiver_position[2]
    # change position of the sphere
    returnCode = sim.simxSetObjectPosition(clientID, sphere, -1,
                                           sphere_position,
                                           sim.simx_opmode_blocking)
    print(f'Sphere moved Position:{sphere_position}')

    # # Stop simulation
    #stop_simulation(clientID)
