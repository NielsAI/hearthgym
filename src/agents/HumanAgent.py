from agents.HearthstoneAgent import HearthstoneAgent

class HumanAgent(HearthstoneAgent):
    """
    Random agent implementation that selects actions uniformly at random.
    """
    def __init__(self):
        super().__init__()

    def act(self, observation, valid_actions: list = None, action_mask=None, env=None):
        """
        Prompt the human player to select an action from the valid actions.
        If no valid actions are available, ask the human player to input a value for each action type.

        :param observation: The current observation from the environment.
        :param valid_actions: List of valid actions (not used in this implementation).
        :param action_mask: Mask for valid actions (not used in this implementation).
        :param env: The environment (not used in this implementation).
        :return: The selected action.
        """
                
        # Sample one of the valid actions
        if valid_actions:
            print("\nValid actions:")
            if env:
                # If the environment is provided, render the valid actions for the human player in a user-friendly way
                env.render_actions(valid_actions)
            else:
                # Present the valid actions to the human player as numbered options
                for i, action in enumerate(valid_actions):
                    print(f"{i}: {action}")
                
            # Get the human player's choice
            choice = int(input("Enter the index of the action you would like to take: "))
            return valid_actions[choice]
        else:
            # Ask the human player to input a value for each action type
            action = {}
            for key, space in self.action_space.items():
                max_value = space.n - 1
                action[key] = int(input(f"Enter the index of the {key}: "))