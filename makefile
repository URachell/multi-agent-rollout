all:
	g++ -std=c++17 -O2 src/rollout.cpp src/indextopair.cpp src/environment.cpp src/basepolicy.cpp src/coststocontrol.cpp src/boxpicker.cpp src/updatetargets.cpp src/updatebasepolicy.cpp src/controlpicker.cpp src/simulate.cpp src/astar.cpp src/initastar.cpp -o rollout

coop:
	g++ -std=c++17 -O2 src/coop.cpp src/indextopair.cpp src/environment.cpp src/boxpicker.cpp src/updatetargets.cpp  src/simulatecoop.cpp src/coopalgorithm.cpp -o coop

clean:
	rm -f rollout
	rm -f coop
