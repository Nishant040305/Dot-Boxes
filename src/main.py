from env.BoardEnv import BaseBoardEnv
from agents.RandomAgent import RandomAgent
from gameSimulation.simulation import SimulateRandomVSrandom, SimulateRandomVSgreedy,SimulateGreedyVSgreedy,SimulateMinMaxVSgreedy

if __name__ == "__main__":
    env = BaseBoardEnv(4)
    simulator = SimulateMinMaxVSgreedy(env)
    score = simulator.simulate()
    print("Agent1: ",score[0])
    print("Agent2: ",score[1])
    if(score[0]>score[1]):
        print("Agent1 wins")
    elif(score[0]<score[1]):
        print("Agent2 wins")
    else:
        print("Draw")