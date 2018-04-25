import random


class Tic(object):
    winning_combos = (
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6])

    winners = ('X-win', 'Draw', 'O-win')

    def __init__(self, squares=[]):
        if len(squares) == 0:
            self.squares = [None for i in range(9)]
        else:
            self.squares = squares

    def show(self):
        for element in [self.squares[i:i + 3] for i in range(0, len(self.squares), 3)]:
            print(element)

    def available_moves(self):
        """Empty Spots"""
        return [k for k, v in enumerate(self.squares) if v is None]

    def available_combos(self, player):
        """Available Combos"""
        return self.available_moves() + self.get_squares(player)

    def complete(self):
        """Is the game over?"""
        if None not in [v for v in self.squares]:
            return True
        if self.winner() != None:
            return True
        return False

    def X_won(self):
        return self.winner() == 'X'

    def O_won(self):
        return self.winner() == 'O'

    def tied(self):
        return self.complete() == True and self.winner() is None

    def winner(self):
        """Determine the Winner"""
        for player in ('X', 'O'):
            positions = self.get_squares(player)
            for combo in self.winning_combos:
                win = True
                for pos in combo:
                    if pos not in positions:
                        win = False
                if win:
                    return player
        return None

    def get_squares(self, player):
        """squares that belong to a player"""
        return [k for k, v in enumerate(self.squares) if v == player]

    def make_move(self, position, player):
        """place on square on the board"""
        self.squares[position] = player

    def minimax(self, board, player, alpha, beta):
        """Min Max Algorithm"""
        if board.complete():
            if board.X_won():
                return -10
            elif board.tied():
                return 0
            elif board.O_won():
                return 10
            best = None
        for move in board.available_moves():
            board.make_move(move, player)
            val = self.minimax(board, get_enemy(player), alpha, beta)
            board.make_move(move, None)
            if player == 'O':
                if val > best:
                    best = val
            else:
                if val < best:
                    best = val
            return best

    def alphabeta(self, board, player, alpha, beta):
        """Alpha Alpha-Beta Pruning"""
        if board.complete():
            if board.X_won():
                return -10
            elif board.tied():
                return 0
            elif board.O_won():
                return 10
        for move in board.available_moves():
            board.make_move(move, player)
            score = self.alphabeta(board, get_enemy(player), alpha, beta)
            board.make_move(move, None)
            if player == 'O':
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    return alpha
            else:
                if score < beta:
                    beta = score
                if beta <= alpha:
                    return beta
        if player == 'O':
            return alpha
        else:
            return beta


def determine(board, player):
    a = -2
    choices = []
    if len(board.available_moves()) == 9:
        return 4
    for move in board.available_moves():
        board.make_move(move, player)
        val = board.alphabeta(board, get_enemy(player), -2, 2)
        board.make_move(move, None)
        if val > a:
            a = val
            choices = [move]
        elif val == a:
            choices.append(move)
    return random.choice(choices)


def get_enemy(player):
    if player == 'X':
        return 'O'
    return 'X'


if __name__ == "__main__":
    board = Tic()
    board.show()

    while not board.complete():
        player = 'X'
        player_move = int(input("Your Move: ")) - 1
        print('\n')
        if not player_move in board.available_moves():
            continue
        board.make_move(player_move, player)
        board.show()

        print('\n')

        if board.complete():
            break
        player = get_enemy(player)
        computer_move = determine(board, player)
        board.make_move(computer_move, player)
        board.show()
    print("The Winner is: ", board.winner())
