import numpy as np
import heapq

MIN_BET = 25.0
TURN_NUMBER = 200
GAMES = 1000


class Roulette:
    def __init__(self):
        self.gain = 0
        self.last_number = -1
        self.board_history = []

    def shuffle(self, return_number=False):
        number = np.random.randint(0, 37)
        self.last_number = number
        if return_number:
            return number

    def pay_agent(self, agent, bet, gale=False):
        board = self.board_history[-1]
        agent.win = 0
        agent.lose = 0
        if gale:
            agent.won_last_game = False
        for i in range(13):
            if board[i] and bet[i]:
                if i == 0:
                    agent.win += 1
                    agent.cash += (MIN_BET / 10) * 36
                elif 0 < i < 7:
                    agent.win += 1
                    agent.cash += bet[i] * 2
                    if gale:
                        agent.won_last_game = True
                elif 7 <= i:
                    agent.win += 1
                    agent.cash += bet[i] * 3
                    if gale:
                        agent.won_last_game = True
            elif not board[i] and bet[i]:
                agent.lose += 1

    def tendency(self):
        trends = np.zeros(13)
        for board in self.board_history:
            trends = np.array(board) + trends

        trends = trends / len(self.board_history)
        return trends

    def check_board(self):
        zero = False
        odd = False
        even = False
        first_half = False
        second_half = False
        first_line = False
        second_line = False
        third_line = False
        first_column = False
        second_column = False
        third_column = False
        red = False
        black = False

        if self.last_number % 2:
            odd = True
        else:
            even = True

        if 0 < self.last_number <= 18:
            first_half = True
        elif 18 < self.last_number:
            second_half = True

        if self.last_number % 3 == 0:
            third_line = True
        elif self.last_number % 3 == 1:
            second_line = True
        elif self.last_number % 3 == 2:
            first_line = True

        if 0 < self.last_number <= 12:
            first_column = True
        elif 12 < self.last_number <= 24:
            second_column = True
        elif 24 < self.last_number:
            third_column = True

        if not self.last_number:
            zero = True
        elif any(np.array([1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]) == self.last_number):
            red = True
        else:
            black = True

        board = [zero, odd, even, first_half, second_half, red, black, first_line, second_line, third_line,
                 first_column, second_column, third_column]

        self.board_history.append(board)
        trends = self.tendency()
        return board, trends


class SingleTrendAgent:
    def __init__(self, cover_zero=False):
        self.cash = 1000
        self.cover_zero = cover_zero
        self.win = 0
        self.lose = 0

    def bet(self, board_trends):
        max_trend = np.argmax(board_trends[1:])
        bet = np.zeros(13)
        bet[max_trend + 1] = MIN_BET
        bet[0] = (MIN_BET / 10) if self.cover_zero else 0
        if self.cash < sum(bet):
            self.win = 0
            self.lose = 0
            return np.zeros(13)
        self.cash -= sum(bet)
        return bet


class MultiTrendAgent:
    def __init__(self, cover_zero=False):
        self.cash = 1000
        self.cover_zero = cover_zero
        self.win = 0
        self.lose = 0

    def bet(self, board_trends):
        bet = np.zeros(13)
        max_trends = np.array(heapq.nlargest(3, range(len(board_trends[1:7])), board_trends[1:7].take))
        bet[max_trends + 1] = MIN_BET
        bet[0] = (MIN_BET / 10) if self.cover_zero else 0
        if self.cash < sum(bet):
            self.win = 0
            self.lose = 0
            return np.zeros(13)
        self.cash -= sum(bet)
        return bet


class ReverseSingleTrendAgent:
    def __init__(self, cover_zero=False):
        self.cash = 1000
        self.cover_zero = cover_zero
        self.win = 0
        self.lose = 0

    def bet(self, board_trends):
        max_trend = np.argmin(board_trends[1:7])
        bet = np.zeros(13)
        bet[max_trend + 1] = MIN_BET
        bet[0] = (MIN_BET / 10) if self.cover_zero else 0
        if self.cash < sum(bet):
            self.win = 0
            self.lose = 0
            return np.zeros(13)
        self.cash -= sum(bet)
        return bet


class ReverseMultiTrendAgent:
    def __init__(self, cover_zero=False):
        self.cash = 1000
        self.cover_zero = cover_zero
        self.win = 0
        self.lose = 0

    def bet(self, board_trends):
        bet = np.zeros(13)
        max_trends = np.array(heapq.nsmallest(3, range(len(board_trends[1:7])), board_trends[1:7].take))
        bet[max_trends + 1] = MIN_BET
        bet[0] = (MIN_BET / 10) if self.cover_zero else 0
        if self.cash < sum(bet):
            self.win = 0
            self.lose = 0
            return np.zeros(13)
        self.cash -= sum(bet)
        return bet


class MartingaleAgent:
    def __init__(self, cover_zero=False):
        self.cash = 1000
        self.last_bet_value = 0
        self.won_last_game = True
        self.first_bet = True
        self.cover_zero = cover_zero
        self.win = 0
        self.lose = 0

    def bet(self, board_trends):
        max_trend = np.argmax(board_trends[1:])
        bet = np.zeros(13)
        self.last_bet_value = MIN_BET if self.won_last_game else (self.last_bet_value * 2)
        bet[max_trend + 1] = self.last_bet_value
        bet[0] = (MIN_BET / 10) if self.cover_zero else 0
        if self.cash < sum(bet):
            self.win = 0
            self.lose = 0
            return np.zeros(13)
        self.cash -= sum(bet)
        return bet


class ReverseMartingaleAgent:
    def __init__(self, cover_zero=False):
        self.cash = 1000
        self.last_bet_value = 0
        self.won_last_game = True
        self.first_bet = True
        self.cover_zero = cover_zero
        self.win = 0
        self.lose = 0

    def bet(self, board_trends):
        max_trend = np.argmin(board_trends[1:7])
        bet = np.zeros(13)
        self.last_bet_value = MIN_BET if self.won_last_game else (self.last_bet_value * 2)
        bet[max_trend + 1] = self.last_bet_value
        bet[0] = (MIN_BET / 10) if self.cover_zero else 0
        if self.cash < sum(bet):
            self.win = 0
            self.lose = 0
            return np.zeros(13)
        self.cash -= sum(bet)
        return bet


class MultiMartingaleAgent:
    def __init__(self, cover_zero=False):
        self.cash = 1000
        self.last_bet_value = 0
        self.won_last_game = True
        self.first_bet = True
        self.cover_zero = cover_zero
        self.win = 0
        self.lose = 0

    def bet(self, board_trends):
        bet = np.zeros(13)
        max_trends = np.array(heapq.nlargest(3, range(len(board_trends[1:7])), board_trends[1:7].take))
        self.last_bet_value = MIN_BET if self.won_last_game else (self.last_bet_value * 2)
        bet[max_trends + 1] = self.last_bet_value
        bet[0] = (MIN_BET / 10) if self.cover_zero else 0
        if self.cash < sum(bet):
            self.win = 0
            self.lose = 0
            return np.zeros(13)
        self.cash -= sum(bet)
        return bet


class ReverseMultiMartingaleAgent:
    def __init__(self, cover_zero=False):
        self.cash = 1000
        self.last_bet_value = 0
        self.won_last_game = True
        self.first_bet = True
        self.cover_zero = cover_zero
        self.win = 0
        self.lose = 0

    def bet(self, board_trends):
        bet = np.zeros(13)
        max_trends = np.array(heapq.nsmallest(3, range(len(board_trends[1:7])), board_trends[1:7].take))
        self.last_bet_value = MIN_BET if self.won_last_game else (self.last_bet_value * 2)
        bet[max_trends + 1] = self.last_bet_value
        bet[0] = (MIN_BET / 10) if self.cover_zero else 0
        if self.cash < sum(bet):
            self.win = 0
            self.lose = 0
            return np.zeros(13)
        self.cash -= sum(bet)
        return bet


first_game = True
cash_log = []
win_log = []
lose_log = []
bet_log = []

for g in range(GAMES):
    print("#### - RUNNING GAME {}/{} - ####".format(g, GAMES))

    # Create the environment
    roullete = Roulette()

    single_trend_agent = SingleTrendAgent()
    multi_trend_agent = MultiTrendAgent()
    reverse_single_trend_agent = ReverseSingleTrendAgent()
    reverse_multi_trend_agent = ReverseMultiTrendAgent()
    martingale_agent = MartingaleAgent()
    reverse_martingale_agent = ReverseMartingaleAgent()
    multi_martingale_agent = MultiMartingaleAgent()
    reverse_multi_martingale_agent = ReverseMultiMartingaleAgent()
    single_trend_agent_zero = SingleTrendAgent(cover_zero=True)
    multi_trend_agent_zero = MultiTrendAgent(cover_zero=True)
    reverse_single_trend_agent_zero = ReverseSingleTrendAgent(cover_zero=True)
    reverse_multi_trend_agent_zero = ReverseMultiTrendAgent(cover_zero=True)
    martingale_agent_zero = MartingaleAgent(cover_zero=True)
    reverse_martingale_agent_zero = ReverseMartingaleAgent(cover_zero=True)
    multi_martingale_agent_zero = MultiMartingaleAgent(cover_zero=True)
    reverse_multi_martingale_agent_zero = ReverseMultiMartingaleAgent(cover_zero=True)
    turn_cash_log = []
    turn_win_log = []
    turn_lose_log = []
    turn_bet_log = []

    for i in range(0, TURN_NUMBER):
        if roullete.last_number == -1:
            roullete.shuffle()  # First game just to fit the strategies
            last_board, board_tendency = roullete.check_board()

        # Agent's bets
        bet1 = single_trend_agent.bet(board_tendency)
        bet2 = multi_trend_agent.bet(board_tendency)
        bet3 = reverse_single_trend_agent.bet(board_tendency)
        bet4 = reverse_multi_trend_agent.bet(board_tendency)
        bet5 = martingale_agent.bet(board_tendency)
        bet6 = reverse_martingale_agent.bet(board_tendency)
        bet7 = multi_martingale_agent.bet(board_tendency)
        bet8 = reverse_multi_martingale_agent.bet(board_tendency)
        bet9 = single_trend_agent_zero.bet(board_tendency)
        bet10 = multi_trend_agent_zero.bet(board_tendency)
        bet11 = reverse_single_trend_agent_zero.bet(board_tendency)
        bet12 = reverse_multi_trend_agent_zero.bet(board_tendency)
        bet13 = martingale_agent_zero.bet(board_tendency)
        bet14 = reverse_martingale_agent_zero.bet(board_tendency)
        bet15 = multi_martingale_agent_zero.bet(board_tendency)
        bet16 = reverse_multi_martingale_agent_zero.bet(board_tendency)

        # Roll
        roullete.shuffle()
        last_board, board_tendency = roullete.check_board()

        # Pay the agents
        roullete.pay_agent(single_trend_agent, bet1)
        roullete.pay_agent(multi_trend_agent, bet2)
        roullete.pay_agent(reverse_single_trend_agent, bet3)
        roullete.pay_agent(reverse_multi_trend_agent, bet4)
        roullete.pay_agent(martingale_agent, bet5, gale=True)
        roullete.pay_agent(reverse_martingale_agent, bet6, gale=True)
        roullete.pay_agent(multi_martingale_agent, bet7, gale=True)
        roullete.pay_agent(reverse_multi_martingale_agent, bet8, gale=True)
        roullete.pay_agent(single_trend_agent_zero, bet9)
        roullete.pay_agent(multi_trend_agent_zero, bet10)
        roullete.pay_agent(reverse_single_trend_agent_zero, bet11)
        roullete.pay_agent(reverse_multi_trend_agent_zero, bet12)
        roullete.pay_agent(martingale_agent_zero, bet13, gale=True)
        roullete.pay_agent(reverse_martingale_agent_zero, bet14, gale=True)
        roullete.pay_agent(multi_martingale_agent_zero, bet15, gale=True)
        roullete.pay_agent(reverse_multi_martingale_agent_zero, bet16, gale=True)

        turn_cash_log.append([single_trend_agent.cash, multi_trend_agent.cash, reverse_single_trend_agent.cash,
                              reverse_multi_trend_agent.cash, martingale_agent.cash, reverse_martingale_agent.cash,
                              multi_martingale_agent.cash, reverse_multi_martingale_agent.cash,
                              single_trend_agent_zero.cash,
                              multi_trend_agent_zero.cash, reverse_single_trend_agent_zero.cash,
                              reverse_multi_trend_agent_zero.cash, martingale_agent_zero.cash,
                              reverse_martingale_agent_zero.cash,
                              multi_martingale_agent_zero.cash, reverse_multi_martingale_agent_zero.cash])

        turn_win_log.append([single_trend_agent.win, multi_trend_agent.win, reverse_single_trend_agent.win,
                             reverse_multi_trend_agent.win, martingale_agent.win, reverse_martingale_agent.win,
                             multi_martingale_agent.win, reverse_multi_martingale_agent.win,
                             single_trend_agent_zero.win, multi_trend_agent_zero.win,
                             reverse_single_trend_agent_zero.win, reverse_multi_trend_agent_zero.win,
                             martingale_agent_zero.win, reverse_martingale_agent_zero.win,
                             multi_martingale_agent_zero.win, reverse_multi_martingale_agent_zero.win])

        turn_lose_log.append([single_trend_agent.lose, multi_trend_agent.lose, reverse_single_trend_agent.lose,
                             reverse_multi_trend_agent.lose, martingale_agent.lose, reverse_martingale_agent.lose,
                             multi_martingale_agent.lose, reverse_multi_martingale_agent.lose,
                             single_trend_agent_zero.lose, multi_trend_agent_zero.lose,
                             reverse_single_trend_agent_zero.lose, reverse_multi_trend_agent_zero.lose,
                             martingale_agent_zero.lose, reverse_martingale_agent_zero.lose,
                             multi_martingale_agent_zero.lose, reverse_multi_martingale_agent_zero.lose])

        turn_bet_log.append([sum(bet1), sum(bet2), sum(bet3), sum(bet4), sum(bet5), sum(bet6), sum(bet7), sum(bet8),
                             sum(bet9), sum(bet10), sum(bet11), sum(bet12), sum(bet13), sum(bet14), sum(bet15),
                             sum(bet16)])

    turn_cash_log = np.array(turn_cash_log).reshape(TURN_NUMBER, 16)
    turn_win_log = np.array(turn_win_log).reshape(TURN_NUMBER, 16)
    turn_lose_log = np.array(turn_lose_log).reshape(TURN_NUMBER, 16)
    turn_bet_log = np.array(turn_bet_log).reshape(TURN_NUMBER, 16)

    if first_game:
        first_game = False
        cash_log = turn_cash_log
        win_log = turn_win_log
        lose_log = turn_lose_log
        bet_log = turn_bet_log
    else:
        lose_log = np.vstack([lose_log, turn_lose_log])
        win_log = np.vstack([win_log, turn_win_log])
        cash_log = np.vstack([cash_log, turn_cash_log])
        bet_log = np.vstack([bet_log, turn_bet_log])

with open('cash.npy', 'wb') as f:
    np.save(f, cash_log)
with open('win.npy', 'wb') as f:
    np.save(f, win_log)
with open('lose.npy', 'wb') as f:
    np.save(f, lose_log)
with open('bet.npy', 'wb') as f:
    np.save(f, bet_log)
