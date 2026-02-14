from env.BoardEnv import BaseBoardEnv
from agents.RandomAgent import RandomAgent
from gameSimulation.randomVSrandom import SimulateRandomVSrandom

if __name__ == "__main__":
    env = BaseBoardEnv(4)
    agent1 = RandomAgent(env)
    agent2 = RandomAgent(env)
    simulator = SimulateRandomVSrandom(env)
    simulator.set_agents(agent1,agent2)
    score = simulator.simulate_actions_display()
    print("Agent1: ",score[0])
    print("Agent2: ",score[1])
    if(score[0]>score[1]):
        print("Agent1 wins")
    elif(score[0]<score[1]):
        print("Agent2 wins")
    else:
        print("Draw")