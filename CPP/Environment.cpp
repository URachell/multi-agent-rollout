#include "Environment.hpp"
#include <iostream>
#include "Astar.cpp"

// Objects in matrix
enum object {space, wall, box, dropOff, firstAgent};

// Costs
static const double c_step = 1.0, c_pickUp = -100.0, c_dropOff = -1000.0, c_collision = 1e10;

static constexpr double discountFactor = 0.99;

template<int N>
struct Discount {
    double arr[N];
    constexpr Discount () : arr() {
        arr[0] = 1.0;
        for (int i = 1; i <  N; ++i)
            arr[i] = arr[i - 1] * discountFactor;
    }
};
static constexpr auto discountFactors = Discount<10000>();




Environment::Environment(int wallOffset, int boxOffset, int n, int agentCount) : m_stepCount(0) {
    m_dim = 2 + 2 * wallOffset + (n - 1) * boxOffset + 2 * n;
    m_matrix = new int[m_dim * m_dim]();
    m_boxesLeft = 0;

    // Populate matrix with walls and boxes
    for(int i = 0; i < m_dim; ++i){
        int fill_value = (i == 0 || i == m_dim - 1) ? wall : space;
        for(int j = 0; j < m_dim; ++j){
            envMat(i, j) = fill_value;
        }
        envMat(i, 0) = wall;
        envMat(i, m_dim - 1) = wall;
    }

    // Populate matrix with agents
    for(int i = 0; i < agentCount; ++i){ 
        envMat(1, 1 + i) = -firstAgent - i;
        m_agentPositions.push_back(std::pair<unsigned int, unsigned int>(1, 1 + i));
    }

    // Populate matrix with boxes
    for(int n_i = 0; n_i < n; ++n_i){
        for(int n_j = 0; n_j < n; ++n_j){
            int i = 1 + wallOffset + n_i * (2 + boxOffset);
            int j = 1 + wallOffset + n_j * (2 + boxOffset);
            envMat(i, j) = box; 
            envMat(i + 1, j) = box; 
            envMat(i, j + 1) = box; 
            envMat(i + 1, j + 1) = box; 
            m_boxesLeft += 4;
        }
    }

}

Environment::Environment(Environment &other) : m_stepCount(other.m_stepCount), m_dim(other.m_dim), m_agentPositions(other.m_agentPositions), m_boxesLeft(other.m_boxesLeft){
    m_matrix = new int[m_dim * m_dim]();
    for (size_t i = 0; i < m_dim * m_dim; ++i)
        m_matrix[i] = other.m_matrix[i];
}
Environment& Environment::operator=(const Environment &other){
    if (this != &other) 
    {
        int *new_m_matrix = new int[other.m_dim * other.m_dim];
        for (size_t i = 0; i < (other.m_dim * other.m_dim); ++i)
            new_m_matrix[i] = other.m_matrix[i];

        m_stepCount = other.m_stepCount;
        m_dim = other.m_dim;
        m_boxesLeft = other.m_boxesLeft;
        m_agentPositions = other.m_agentPositions;

        delete[] m_matrix;
        m_matrix = new_m_matrix; // Point to new place in memory 
    }

    return *this;
}
Environment::~Environment(){
    if(m_matrix != nullptr)
        delete[] m_matrix;
}

int &Environment::envMat(int n, int m){
    return m_matrix[m + n * m_dim];
}
void Environment::printMatrix(){
    for(int i = 0; i < m_dim; i++){
        for(int j = 0; j < m_dim; j++){
            int el = envMat(i, j);
            if(el < 0){
                std::cout << " " << el;
            }
            else
            {
                std::cout << "  ";
                switch(el){
                    case 0:
                        std::cout << " ";
                        break;
                    case 1:
                        std::cout << 'X';
                        break;
                    default:
                        std::cout << el;
                        break;
                }
            }
        }
        std::cout << '\n';
    }
}

int getNumOfAgents(){
    return m_agentPositions.size();
}

int getDim(){
    return m_dim;
}

int getMatPtr(){
    return m_matrix;

int* getMatPtr(){
    return m_matrix;
}
}

bool isDone(){
    return m_boxesLeft == 0 ? true : false;
}

/*
// Getter for environment matrix
std::vector< std::vector< int > > Environment::getMatrix(){
    return m_matrix;
}

*/
// Returns the cost of a given set of actions as well as updates the environment
double Environment::step(std::vector<uint8_t> actions, std::vector<std::pair<unsigned int, unsigned int>> targets){
    assert(actions.size() == m_agentPositions.size());
    assert(targets.size() == m_agentPositions.size());

    std::vector< std::pair<unsigned int, unsigned int> > newPositions(m_agentPositions.size());

    double cost;
    const int tot_dim = m_dim * m_dim;
    int coll_mat[tot_dim];
    for(int i = 0; i < tot_dim; ++i)
        coll_mat[i] = 0;

    // Find new positions 
    for(size_t agent_index = 0; agent_index < m_agentPositions.size(); ++agent_index){
        std::pair agentPos = m_agentPositions[agent_index];
        std::pair target = targets[agent_index];
        uint8_t action = actions[agent_index];

        std::pair<unsigned int, unsigned int> newPos(agentPos.first, agentPos.second); // Stand still

        switch(action){
            case 1: // Move up
                newPos.first = agentPos.first - 1;
                break;
            case 2: // Move down
                newPos.first = agentPos.first + 1;
                break;
            case 3: // Move left
                newPos.second = agentPos.second - 1;
                break;
            case 4: // Move right
                newPos.second = agentPos.second + 1;
                break;
        }
        int newPosEl = envMat(newPos.first, newPos.second);

        // If the new position is a wall or an undesired box, stand still, otherwise move
        if(newPosEl == wall || (newPosEl == box && newPos != targets[agent_index])){
            newPositions[agent_index] = m_agentPositions[agent_index];
        }
        else{
            newPositions[agent_index] = newPos;
        }

        // Check for and penalize collisions
        if(coll_mat[m_dim * newPositions[agent_index].first + newPositions[agent_index].second] != 0){
            cost += c_collision * discountFactors.arr[m_stepCount];
        }else{
            coll_mat[m_dim * newPositions[agent_index].first + newPositions[agent_index].second] = 1;
        }
    }

    // Check for and penalize swaps 
    for(size_t i = 0; i < m_agentPositions.size(); ++i){
        int agent_index = envMat(newPositions[i].first, newPositions[i].second); // Which agent stood here before?
        agent_index = (agent_index>=0) ? agent_index : -agent_index; 

        if(newPositions[i] == m_agentPositions[firstAgent - agent_index])
            cost += c_collision * discountFactors.arr[m_stepCount];
    }

    std::vector<int> oldEl(m_agentPositions.size());
    for(size_t agent_index = 0; agent_index < m_agentPositions.size(); ++agent_index){
        oldEl[agent_index] = envMat(m_agentPositions[agent_index].first, m_agentPositions[agent_index].second);
        envMat(m_agentPositions[agent_index].first, m_agentPositions[agent_index].second) = space;
    }
    for(size_t agent_index = 0; agent_index < m_agentPositions.size(); ++agent_index){
        
        if(actions[agent_index] != 0)
            cost += c_step;

        // Update matrix
        envMat(newPositions[agent_index].first, newPositions[agent_index].second) = oldEl[agent_index];

        // If an agent reaches its target
        if(newPositions[agent_index] == targets[agent_index]){
            if(envMat(newPositions[agent_index].first, newPositions[agent_index].second) > 0){ // If the agent has a box
                cost += c_dropOff;
                m_boxesLeft--;
            }else{
                cost += c_pickUp;
            }
            envMat(newPositions[agent_index].first, newPositions[agent_index].second) = -envMat(newPositions[agent_index].first, newPositions[agent_index].second);
        }
    }
    m_agentPositions = newPositions;
    m_stepCount += 1;

    return cost;
}