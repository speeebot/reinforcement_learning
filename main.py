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

    # Initialize Q-table
    q_table = np.zeros((state_space_size, action_space_size))
    #q_table = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(num_episodes):
        print(f"Episode {episode+1}:")

        # Set rotation velocity randomly
        rng = np.random.default_rng()
        velReal = env.rotation_velocity(rng)

        # Start simulation, process objects/handles
        env.start_simulation()

        # Set initial position of the source cup
            # State is x coordinate of source cup
        state = env.reset(rng)
        done = 0
        # Keep track of rewards for each episode, to be stored in a list for analysis
        rewards_current_episode = 0

        # Get starting state
        #cubes_position, source_cup_position = get_state(object_shapes_handles,
        #                                            clientID, source_cup)
        
        if(os.path.exists(q_table_filename)):
            q_table = np.loadtxt(q_table_filename)
        print("Q-table loaded.")

        for j in range(velReal.shape[0]):

            env.step_chores()
            #initialize the speed of the source cup at this frame
            env.speed = velReal[j]

            # Get current state
            #cubes_position, source_cup_position = get_state(object_shapes_handles,
            #                                        clientID, source_cup)
            
            # Update state
            #new_state = source_cup_position[0]

            '''source_low_x = center_position + low
            source_high_x = center_position + high
            receive_low_x = receive_position[0] + low
            receive_high_x = receive_position[0] + high
            source_x = round(source_cup_position[0], 3)'''

            '''print(f"RANGE: {source_low_x + 0*res} to {(source_low_x + 100)*res}") # Outer bounds -> negative reward
            print(f"RANGE: {source_low_x + 101*res} to {source_low_x + 200*res}") # Closer to receive cup -> less negative reward
            print(f"RANGE: {source_low_x + 201*res} to {source_low_x + 300*res}") # Where we want the source cup to be -> positive reward
            print(f"RANGE: {source_low_x + 301*res} to {source_low_x + 400*res}") # Outer bounds -> less negative reward
            print(f"RANGE: {source_low_x + 401*res} to {source_low_x + 500*res}") # Outer bounds -> negative reward
            #print(f"{source_high_x}, {source_low_x + 500*res}")'''

            # Define ranges to map state space to a given action
            # Outer bounds -> negative reward
            '''if check_range(source_x, receive_low_x + 0*res, receive_low_x + 50*res):
                reward = get_distance_3d(old_source_cup_position, receive_position) \
                     - get_distance_3d(source_cup_position, receive_position)'''

            exploration_rate_threshold = np.random.uniform(0, 1)
            # If exploitation is picked, select action where max Q-value exists within state's row in the q-table
            if exploration_rate_threshold > exploration_rate:
                # Select the largest Q-value
                action = np.argmax(q_table[state,:]) - 2
            # If exploration is picked, select a random action from the current state's row in the q-table
            else:
                action = env.action_space.sample() - 2

            #Take next action
            new_state, reward, done, info = env.step(action)

            # Calculate reward as the negative distance between the source up and the receiving cup
            #reward = get_distance_3d(source_cup_position, receive_position)

            # Rotate cup based on speed value
            #env.rotate_cup()
            # Move cup laterally based on selected action in Q-table
            #env.move_cup(action)
            # Update Q-table for Q(s,a)
            print(f"STATE: {normalize(state, -0.85 + low, -0.85 + high)}, REWARD: {reward}, ACTION: {action}")
            update_q_table(q_table, state, action, new_state, reward)


            #update state variable
            state = new_state
            rewards_current_episode += reward
            
            # Break if cup goes back to vertical position
            if done:
                break
        #end for

        #Exploration rate decay
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
        
        #Append current episode's reward to total rewards list for later
        rewards_all_episodes.append(rewards_current_episode)

        # Stop simulation
        env.stop_simulation()

        np.savetxt(q_table_filename, q_table)
        print(f"Q-table saved to {q_table_filename}")

    np.savetxt(rewards_filename, rewards_all_episodes)
    print(f"Saved rewards for each episode to {rewards_filename}")
    print(f"Final exploration rate: {exploration_rate}")

if __name__ == '__main__':

    main()
