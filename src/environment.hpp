#pragma once
#include <vector>
#include <cstdint>
#include "assert.h"
#include <algorithm>

#define ENV_HEIGHT 47
#define ENV_WIDTH 115
#define ENV_CAPACITY (ENV_HEIGHT * ENV_WIDTH)

class Environment{ 
    private:
        int m_stepCount;
        int m_boxesLeft;
        std::array<int, ENV_CAPACITY> m_matrix;
        std::vector< std::pair<int, int > > m_agentPositions; 
        std::vector<std::pair<int, int>> m_availableBoxes;

    public:
        Environment(int wallOffset, int boxOffset, int n, int agentCount);
        int &envMat(int n, int m);
        void printMatrix(std::vector<std::pair<int, int>> dropOffPoints, bool redraw);
        int getNumOfAgents();
        std::array<int, ENV_CAPACITY>& getMatrixArray();
        bool isDone();
        int getStepCount();
        int getBoxesLeft();

        int getMatrixIndex(int agentIdx); 
        std::vector<int> getAgentValues();
        std::vector<std::pair<int,int>> &getAvailableBoxes();

        // Returns the cost of a given set of actions as well as updates the environment
        double step(std::vector<int> &actions, std::vector< std::pair<int, int> > &targets); 


		void forceFinish();
};
