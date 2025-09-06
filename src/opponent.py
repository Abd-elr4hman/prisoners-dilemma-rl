import numpy as np

# Opponent strategies
class BaseOpponent:
    def reset(self):
        pass
    
    def get_action(self, history, player_last_action):
        raise NotImplementedError


class RandomOpponent(BaseOpponent):
    def get_action(self, history, player_last_action):
        return np.random.choice([0, 1])

class TitForTatOpponent(BaseOpponent):
    def get_action(self, history, player_last_action):
        if len(history) == 0:
            return 0  # Start by cooperating
        else:
            # Copy player's last action
            return history[-1][0]

class AlwaysDefectOpponent(BaseOpponent):
    def get_action(self, history, player_last_action):
        return 1
    
class AlwaysCooperateOpponent(BaseOpponent):
    def get_action(self, history, player_last_action):
        return 0