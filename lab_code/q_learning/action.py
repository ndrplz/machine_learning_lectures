class Action:
    """
    Class to encapsulate allowable actions
    """
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @staticmethod
    def to_arrow(action):
        """
        Represent the action as string.
        
        Parameters
        ----------
        action: Action
            An instance of `Action` class
            
        Returns
        -------
        str_arrow: str
            String description of action `action`
        """
        str_arrow = ''
        if action == Action.UP:
            str_arrow = '^'
        elif action == Action.DOWN:
            str_arrow = 'v'
        elif action == Action.LEFT:
            str_arrow = '<'
        elif action == Action.RIGHT:
            str_arrow = '>'
        return str_arrow
