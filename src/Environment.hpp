#pragma once
#include <vector>
#include <cstdint>
#include "assert.h"
#include <algorithm>
#include <unordered_set>

class Environment{ 
    private:
        int m_stepCount;
        int m_height;
        int m_width;
        int m_boxesLeft;
        int *m_matrix;
        std::vector< std::pair<int, int > > m_agentPositions; 
        std::vector<std::pair<int, int>> m_availableBoxes;
		std::unordered_set<int> m_brokenRobots;

    public:
        Environment(int wallOffset, int boxOffset, int n, int agentCount);
        Environment(Environment &other); // Copy constructor
        Environment &operator=(const Environment &other);
        ~Environment(); // Destructor

        int &envMat(int n, int m);
        void printMatrix(std::vector<std::pair<int, int>> dropOffPoints, bool redraw);
        int getNumOfAgents();
        int getHeight();
        int getWidth();
        int* getMatPtr();
        bool isDone();
        int getStepCount();
        int getBoxesLeft();

        int getMatrixIndex(int agentIdx); 
        std::vector<int> getAgentValues();
        std::vector<std::pair<int,int>> &getAvailableBoxes();

        // Returns the cost of a given set of actions as well as updates the environment
        double step(std::vector<int> &actions, std::vector< std::pair<int, int> > &targets); 

		void breakRobot(int robotIndex, std::vector<std::pair<int,int>> &targets);

		void forceFinish();
};
