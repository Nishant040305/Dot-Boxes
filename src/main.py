from env.BoardEnv import BaseBoardEnv
from agents.RandomAgent import RandomAgent
from gameSimulation.simulation import SimulateRandomVSrandom, SimulateRandomVSgreedy,SimulateGreedyVSgreedy,SimulateMinMaxVSgreedy,SimulateMinMaxVSMinMax,SimulateAlphaBetaVSAlphaBeta,SimulateAlphaBetaVSgreedy,SimulateAlphaBetaVSMinMax

if __name__ == "__main__":
    env = BaseBoardEnv(5)
    simulator = SimulateAlphaBetaVSAlphaBeta(env,depth1=5,depth2=3,parallel=True)
    score = simulator.simulate()
    print("Agent1: ",score[0])
    print("Agent2: ",score[1])
    if(score[0]>score[1]):
        print("Agent1 wins")
    elif(score[0]<score[1]):
        print("Agent2 wins")
    else:
        print("Draw")