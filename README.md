# Mancala AI

## Example Usage
```py
# main.py

from Simulator import Simulator
from agents import *

# Define the two agents that will play the game.
p1 = HumanAgent()
p2 = MinimaxAgent(9)
simulator = Simulator(p1, p2)
simulator.start()
```

```console
$ python main.py
```

## Simulator
- `Simulator(p1, p2)`: starts a simulator for the two agents, p1 and p2
- `Simulator(p1, p2, silent = True)`: same as above but output is suppressed

## Available Agents

#### `RandomAgent()`
An agent that chooses a random action.

#### `HumanAgent()`
An agent that prompts the user for every action.

#### `GreedyAgent()`
An agent that greedily chooses the best action every turn.

#### `MinimaxAgent(depth)`
An agent that uses the minimax algorithm to search through the game tree over a certain depth.
