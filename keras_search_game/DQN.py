from maze_env import Maze
from blogs import DQN


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()

#代码来自以下大佬
#https://blog.csdn.net/senjie_wang/article/details/82708381?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162374335516780271513336%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=162374335516780271513336&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~baidu_landing_v2~default-1-82708381.nonecase&utm_term=python+keras+dqn+&spm=1018.2226.3001.4450

if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DQN(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=4000,
                      output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    # RL.plot_cost()