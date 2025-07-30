# PokerGym

A poker simulation and reinforcement learning environment.
Uses [Gymnasium](https://gymnasium.farama.org/) (formerly OpenAI gym) and [PettingZoo](https://pettingzoo.farama.org/) API to create a poker environment for training agents.

Following poker rules from [Poker Tournament Directors Association](https://www.pokertda.com/poker-tda-rules/).

## Installation
To run the environment, you need to install the required dependencies currently available in `requirements.txt`.
Additionally, you can install the optional for testing dependencies in `requirements-dev.txt`.

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Current Status
PokerGym is in an early stage of development. The environment is functional, but the bots are currently simple deterministic or random agents. 
Reinforcement learning agents are planned for the next stages of development.


<!-- ### Features
- None -->

### Goals
| Task | Status |
|------|--------|
| Set-up Project | ✔ |
| Basic Poker Environment | ✔ |
| Deal with side-pots | ✔ |
| Basic Poker Agent | ✔ |
| Random Poker Agent | ✔ |
| Reinforcement Learning Poker Agent |  |
| Test Cases | ✔ |
| Additional Cases |  |
| Generated Standarized Game Log outputs |  |