import numpy as np
from othello.OthelloUtil import getValidMoves, judge_next_game
from othello.OthelloGame import OthelloGame

class BOT():
    def __init__(self, n=8):
        self.game = OthelloGame(n)
        self.n = n
        self.player = None

    def getAction(self, board, color):
        self.player = color # 設定我方顏色
        # 判斷是否已經進行下一局
        if judge_next_game(self.game, board):
            self.game = OthelloGame(self.n)
        # 更新盤面(對手落子直到我方有合法步)
        # print(find_opp_move(self.game, board), self.player)
        self.game = board # 更新盤面(對手落子直到我方有合法步)
        # for opp_position in find_opp_move(self.game, board):
        #     self.game.do_move(opp_position, -self.player)

        self.game.current_player = self.player # 設定當前玩家

        valids = getValidMoves(board, color) # 取得合法步列表
        # print(valids, type(valids[0]))
        # print(valids[0][0], valids[0][1])
        position = np.random.choice(range(len(valids)), size=1)[0] # 隨機選擇其中一合法步
        position = valids[position]
        self.game.move(position) # 更新盤面(我方落子)
        # print(self.game)
        # print("---------------------------------------------------------")
        # print(position)
        # print(type(position), position.shape)
        return position
    
