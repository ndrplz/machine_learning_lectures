class Action:

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @staticmethod
    def to_arrow(action):
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
