"""
    This code communicated with the coppeliaSim simulation and simulates pouring aiming to a destination container
    
    WINDOWS:
    ./Applications/coppeliaSim.app/Contents/MacOSMacOS/coppeliaSim -gREMOTEAPISERVERSERVICE_19000_FALSE_TRUE ~/path/to/file/2cups_Intro_to_AI.ttt

    MACOS:
    cd ~/Applications/coppeliaSim.app/Contents/MacOS
    ./coppeliaSim -gREMOTEAPISERVERSERVICE_19000_FALSE_TRUE ~/Documents/school/fall22/intro_ai/project2_part1/2cups_Intro_to_AI.ttt

"""

from environment import *

def main():
    env = CubesCups()

    q_table_filename = "q_table.txt"

    # Set rotation velocity randomly
    rng = np.random.default_rng()
    velReal = env.rotation_velocity(rng)

    # Start simulation, process objects/handles
    env.start_simulation()

    # Set initial position of the source cup
        # State is x coordinate of source cup
    state = env.reset(rng)

    done = False

    # Get source cup offset
    offset = env.get_cup_offset(rng)
    
    if(os.path.exists(q_table_filename)):
        q_table = np.loadtxt(q_table_filename)
        print(f"{q_table_filename} loaded.")
    else:
        print(f"Was unable to load {q_table_filename}")

    for j in range(velReal.shape[0]):
        env.step_chores()
        # Initialize the speed of the source cup at this frame
        env.speed = velReal[j]

        # Offset is in range (-0.05, 0.05) -> -1000 will always normalize to positive value

        # Select the largest Q-value
        action = np.argmax(q_table[state,:]) - 2

        #print(f"index: {norm_state+1}, {q_table[norm_state,:]}")

        #Take next action
        new_state, reward, info = env.step(action)

        print(f"state: {state}, action selected: {action}")
        
        #update state variable
        state = new_state
        
        # Break if cup goes back to vertical position
        '''if done == True:
            print("done")
            break'''
        if j > 10 and env.joint_position > 0:
            break
    #end for

    # Stop simulation
    env.stop_simulation()

if __name__ == '__main__':

    main()

