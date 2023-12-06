import numpy as np
import copy, time
from othello.OthelloUtil import np_2_str, str_2_np, getValidMoves, isEndGame, judge_next_game
from othello.OthelloGame import *

Vmap = np.array([[ 1.00, -0.25, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, -0.25,  1.00],
                 [-0.25, -0.25, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, -0.25, -0.25],
                 [ 0.10,  0.01, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05,  0.01,  0.10],
                 [ 0.10,  0.01, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05,  0.01,  0.10],
                 [ 0.05,  0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02,  0.01,  0.05],
                 [ 0.05,  0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02,  0.01,  0.05],
                 [ 0.05,  0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02,  0.01,  0.05],
                 [ 0.05,  0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02,  0.01,  0.05],
                 [ 0.10,  0.01, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05,  0.01,  0.10],
                 [ 0.10,  0.01, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05,  0.01,  0.10],
                 [-0.25, -0.25, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, -0.25, -0.25],
                 [ 1.00, -0.25, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, -0.25,  1.00]])


class TreeNode(object):
    """
    為alpha beta tree的節點
    """

    def __init__(self, parent, curr_player, player_color, depth=0):
        """
        parent: 父節點
        children: 子節點, 為一個字典, key為action, value為TreeNode
        children_curr: 下一個節點的玩家
        values: 節點的值
        curr_player: 當前節點的玩家
        player_color: 玩家的顏色
        depth: 節點的深度
        """
        self._parent = parent
        self._children = {}
        self.children_curr = None
        self.values = 0
        self.curr_player = curr_player
        self.player_color = player_color 
        self.depth = depth

    def get_value(self, board):
        """
        計算當下節點的值，用盤面的子來計算。
        """
        black_idxs = np.where(board == BLACK)
        wiite_idxs = np.where(board == WHITE)
        if self.curr_player == BLACK:
            # 找出黑子的數量
            self.values = np.sum(black_idxs) - np.sum(wiite_idxs)
            # self.values = np.sum(Vmap[black_idxs]) - np.sum(Vmap[wiite_idxs])
            # self.values = 100 - len(wiite_idxs[0])
        else:
            # 找出白子的數量
            self.values = np.sum(wiite_idxs) - np.sum(black_idxs)
            # self.values = np.sum(Vmap[wiite_idxs]) - np.sum(Vmap[black_idxs])
            # self.values = 100 - len(black_idxs[0])

    def expand(self, actions, next_player):
        """
            actions: 下一步盤面的列表
            next_player: 下一個出牌的人
            根據以上參數擴展節點
        """
        self.children_curr = next_player
        for i in range(len(actions)):
            if np_2_str(actions[i]) not in self._children:
                self._children[np_2_str(actions[i])] = TreeNode(self, next_player, self.player_color, self.depth + 1)

        # 將children按照value排序
        # self._children = dict(sorted(self._children.items(), key=lambda act_node: act_node[1].values, reverse=True))

    def _ismax(self):
        """Check if it's max node (i.e. the node of the player who is maximizing
        the value).
        """
        return True if self.children_curr == self.player_color else False

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class alpha_beta(object):

    def __init__(self, depth=7, time_limit_sec=20):
        """
        root: 根節點
        player_color: 玩家的顏色
        depth: 搜尋深度
        time_limit_sec: 搜尋時間限制
        """
        self._root = None
        self.player_color = None
        self.depth = depth
        self.start_time = None
        self.time_limit_sec = time_limit_sec
        
    def reset_root(self, player_color):
        """
        重置根節點
        """
        self.player_color = player_color
        self._root = TreeNode(None, -player_color, player_color) # 第一個節點一定是對手的節點

    def expand_abtree(self, node: TreeNode, state: OthelloGame, alpha=float('-inf'), beta=float('inf')):
        """
        進行展開，直到指定的深度或是遊戲結束。 
        更新節點的值。
        更新alpha beta值。
        """
        if time.time() - self.start_time > self.time_limit_sec:
            node.get_value(state)
            
            return node.values
        
        if node.depth == self.depth or isEndGame(state):
            node.get_value(state)
            return node.values
        else:
            childrens = state.availables()
            node.expand(childrens, state.current_player)

        if node._ismax(): # 當前節點是我方節點
            for action, child in node._children.items():
                copy_state = copy.deepcopy(state)
                copy_state.move(str_2_np(action))
                alpha = max(alpha, self.expand_abtree(child, copy_state, alpha, beta))
                if alpha >= beta:
                    break

            node.values = alpha
            return alpha
        
        else:
            for action, child in node._children.items():
                copy_state = copy.deepcopy(state)
                copy_state.move(str_2_np(action))
                beta = min(beta, self.expand_abtree(child, copy_state, alpha, beta))
                if alpha >= beta:
                    break

            node.values = beta
            return beta

    def get_move(self, board):
        """
        得到最好的落子
        """
        self.start_time = time.time() # 設定開始時間
        self.expand_abtree(self._root, board) # 展開樹

        for action, child in self._root._children.items():
            print(action, child.values)
        print("=========================================================")

        return max(self._root._children.items(), key=lambda act_node: act_node[1].values)[0]

    def __str__(self):
        return "Alpha_Beta {}".format(self.player_color)


class Alpha_Beta_BOT(object):
    """AI player based on alpha beta""" # depth=[3, 6, 5], time_limit_sec=[1, 4, 3]
    def __init__(self, n, depth=[3, 7, 5], time_limit_sec=[1, 4, 3]):
        self.alpha_beta = [alpha_beta(depth=d, time_limit_sec=t) for d, t in zip(depth, time_limit_sec)]
        self.game = OthelloGame(n)
        self.player = None
        self.round = 0
        self.n = n

    def getAction(self, board, color):
        self.player = color # 設定我方顏色
        # 判斷是否已經進行下一局
        if judge_next_game(self.game, board):
            self.game = OthelloGame(n=self.n)
            self.round = 0
        self.round += 1
        # | 21 * 1 = 21 | 39 * 4 = 156 | 10 * 3 = 30 | 210
        # | 5 * 2 = 10  | 20 * 10 = 200 | 
        if self.round <= 21:
            ab = self.alpha_beta[0]  # 前盤搜
        elif self.round <= 60:
            ab = self.alpha_beta[1]  # 中盤搜
        else:
            ab = self.alpha_beta[2]  # 後盤搜
        
        self.game[:] = board[:] # 更新盤面
        
        self.game.current_player = self.player # 設定當前玩家
        # 重置tree
        ab.reset_root(color)

        move_str = ab.get_move(self.game) # 取得alpha beta的落子(str)

        move = str_2_np(move_str) # 轉換成np.array

        return move

    def __str__(self):
        return "alpha beta {}".format(self.player)
