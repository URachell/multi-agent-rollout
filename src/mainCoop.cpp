#include <iostream>
#include "SimulateCoop.hpp"

int main(){
    int simulationCount = 0;
    int numOfSuccess = 0;
    while(simulationCount < 200){
        //bool success = simulate(10);
        bool success = simulateCoop(150);

        if(success)
            numOfSuccess++;

        simulationCount++;

        std::cout << "Number of simulations: " << simulationCount << ", number of successes: " << numOfSuccess << std::endl;
    }
    return 0;
}