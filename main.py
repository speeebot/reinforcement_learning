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

    global exploration_rate
    rewards_all_episodes = []
    rewards_filename = "rewards_history.txt"
    q_table_filename = "q_table.txt"

    #source_x_low = -0.80 + low
    #source_x_high = -0.80 + high
    #velReal_low = -0.7361215932167728
    #velReal_high = 0.8499989492543077 
    #sect = np.linspace(-0.80 + low, -0.80 + high, 500)
    #sect2 = np.linspace(velReal_low, velReal_high, 500)
    #print(sect2)

    for episode in range(num_episodes):
        print(f"Episode {episode+1}:")

        # Set rotation velocity randomly
        rng = np.random.default_rng()
        velReal = env.rotation_velocity(rng)

        # Start simulation, process objects/handles
        env.start_simulation()

        # Set initial position of the source cup
            # State is x coordinate of source cup
        state = env.discretize_state(env.reset(rng))
        print(f"state: {state}")
        done = False
        # Keep track of rewards for each episode, to be stored in a list for analysis
        rewards_current_episode = 0

        # Get source cup offset
        #offset = env.get_cup_offset(rng)
        #print(f"offset: {offset}")

        if(os.path.exists(q_table_filename)):
            q_table = np.loadtxt(q_table_filename)
        print("Q-table loaded.")

        for j in range(velReal.shape[0]):
            env.step_chores()
            # Initialize the speed of the source cup at this frame
            env.speed = velReal[j]
            print(env.speed)
            # Offset is in range (-0.05, 0.05) -> -1000 will always normalize to positive value
            #norm_low = math.floor((-0.85 + low + offset) * -1000)
            #norm_high = math.floor((-0.85 + high + offset) * -1000)

            action = env.pick_action(state)

            #print(f"norm_low: {norm_low}, norm_high: {norm_high}, norm_state: {norm_state}")
            
            #print(f"HERE: {-0.85 + low + offset}, {-0.85 + high + offset}")
            #Take next action
            obs, reward, info = env.step(action)

            new_state = env.discretize_state(obs)
            
            #print(offset)
            # Update Q-table for Q(s,a)
            #print(f"velReal[j]: {velReal[j]}, source_pos: {env.source_cup_position[0]}")
            print(f"STATE: {state}, new_state: {new_state} REWARD: {reward}, ACTION: {action}")
            env.update_q(state, action, new_state, reward)

            #update state variable
            state = new_state
            rewards_current_episode += reward
            
            # Break if cup goes back to vertical position
            '''if done == True:
                print("done")
                break'''
            if j > 10 and env.joint_position > 0:
                break
        #end for

        # Stop simulation
        env.stop_simulation()

        #Exploration rate decay
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
        
        #Append current episode's reward to total rewards list for later
        rewards_all_episodes.append(rewards_current_episode)

        np.savetxt(q_table_filename, q_table)
        print(f"Q-table saved to {q_table_filename}")

    np.savetxt(rewards_filename, rewards_all_episodes)
    print(f"Saved rewards for each episode to {rewards_filename}")
    print(f"Final exploration rate: {exploration_rate}")

if __name__ == '__main__':

    main()
