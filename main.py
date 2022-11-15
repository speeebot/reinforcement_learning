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
    q_table_filename = "q_table.pkl"

    # Load q_table.pkl for updating, if it exists
    if(os.path.exists("q_table.pkl")):
        with open('q_table.pkl', 'rb') as f:
            env.q_table = pickle.load(f)
        print("Q-table loaded.")

    for episode in range(env.num_episodes):
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
            env.learning_rate = env.get_learning_rate(j)
            env.epsilon = env.get_epsilon(j)

            action = env.pick_action(state)

            #print(f"norm_low: {norm_low}, norm_high: {norm_high}, norm_state: {norm_state}")
            
            #print(f"HERE: {-0.85 + low + offset}, {-0.85 + high + offset}")
            #Take next action
            obs, reward, info = env.step(action)

            new_state = env.discretize_state(obs)
            
            #print(offset)
            # Update Q-table for Q(s,a)
            print(f"STATE: {state}, new_state: {new_state} REWARD: {reward}, ACTION: {action}")
            env.update_q(state, action, new_state, reward)

            #update state variable
            state = new_state

            # Keep track of rewards for current episode
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
        
        #Append current episode's reward to total rewards list for later
        rewards_all_episodes.append(rewards_current_episode)

        with open(q_table_filename, 'wb') as f:
            pickle.dump(env.q_table, f)
        print(f"Q-table saved to {q_table_filename}")

    np.savetxt(rewards_filename, rewards_all_episodes)
    print(f"Saved rewards for each episode to {rewards_filename}")

if __name__ == '__main__':

    main()
