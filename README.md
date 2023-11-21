# CS5446 Project

Group: P22

## Classical Planning
This project is to implement a classical planning agent that can solve planning problems in the deterministic, fully observable environment. 

We implemented the following algorithms on CartPole-v1 and Pendulum-v1 environments:
* DDPG  
* PPO
* SAC
* TD3
* DQN
* A2C
* FQF
* Rainbow
* PG
* C51
* QR-DQN
* Gail

### Installation
To install the required packages, please use the following command:
```
pip install -r requirements.txt
```
### Usage
To run the code, please use the following command:
```
python <dir>
```
where `<dir>` is the directory of the algorithm you want to run. For example, to run DQN for CartPole-v1, please use the following command:
```
python algorithms/discrete/dqn.py
```

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.