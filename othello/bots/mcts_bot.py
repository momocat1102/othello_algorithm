import numpy as np
import copy, time
from operator import itemgetter
from othello.OthelloUtil import getValidMoves, find_opp_move, np_2_str, str_2_np
from othello.OthelloGame import *


show_detail = False

def rollout_policy_fn(board: OthelloGame):
    """a coarse, fast version of policy_fn used in the rollout phase."""

    valids = getValidMoves(board, board.current_player) # 取得合法步列表

    if len(valids) == 0:
        board.current_player = -1. * board.current_player
        valids = getValidMoves(board, board.current_player)
    position = np.random.choice(range(len(valids)), size=1)[0] # 隨機選擇其中一合法步
    position = valids[position]
    return position


class TreeNode(object):
    """
    為MCTS的節點
    """

    def __init__(self, parent, curr_player, player_color):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self.curr_player = curr_player
        self.player_color = player_color

    def expand(self, actions, next_player, n):
        """
            action_priors: (action, prob)， 所有可以出的牌和其機率
            next_player: 下一個出牌的人
            根據以上參數擴展節點
        """
        for action in actions:
            self._children[np_2_str(action)] = TreeNode(self, next_player, self.player_color)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        # if show_detail:
        #     for action, child in self._children.items():
        #         if child.curr_player == BLACK:
        #             curr = "BLACK"
        #         else:
        #             curr = "WHITE"
        #         if child.player_color == BLACK:
        #             color = "BLACK"
        #         else:
        #             color = "WHITE"
        #         print(f"child position: {action}")
        #         print(f"child curr_player: {curr}")
        #         print(f"child player_color: {color}")
        #         print(f"child n_visits: {self._children[action]._n_visits}")
        #         print(f"child Q: {self._children[action]._Q}")
        #         print(f"child u: {self._children[action]._u}")
        #         print(f"child P: {self._children[action]._P}")
        #         print("******************************************")

        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q.
        """
        self._u = (c_puct * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))

        return self._Q + self._u

    def back_update(self, leaf_value):
        """ 
            反向去更所有直系節點的Q值和訪問次數
            leaf_value: 從葉節點到當前節點的分數，leaf_value自己的分數，若當前節點為對手，則leaf_value為13-leaf_value   
        """
        curr_value = leaf_value
        self._n_visits += 1
        
        if self.curr_player != self.player_color:
            curr_value = -1. * curr_value
            
        # 更新Q值
        self._Q += 1.0 * (curr_value - self._Q) / self._n_visits
       
        if self._parent:
            self._parent.back_update(leaf_value)

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, c_puct=5, n_playout=1000, time_limit=2.5):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = None
        self.player_color = None
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.time_limit = time_limit
        

    def reset_root(self, player_color):
        self.player_color = player_color
        self._root = TreeNode(None, -player_color, player_color) # 第一個節點一定是對手的節點

    def _playout(self, state: OthelloGame):
        """
        一次模擬, 會選定一個葉節點, 並從葉節點開始模擬, 會先擴展一次, 直到遊戲結束, 最後更新分數。
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        # print(node.is_leaf())
        # print("+++++++++++++++++select node++++++++++++++++")
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.

            # if show_detail:
            #     if node.curr_player == BLACK:
            #         curr = "BLACK"
            #     else:
            #         curr = "WHITE"
            #     if state.current_player == BLACK:
            #         state_curr = "BLACK"
            #     else:
            #         state_curr = "WHITE"
            #     print(f"node 資訊:")
            #     print(f"node curr_player: {curr}")
            #     print(f"state curr_player: {state_curr}")
            #     state.showBoard()
            #     print("-----------------select----------------------")
            action, node = node.select(self._c_puct)
            state.move(str_2_np(action))
            # if show_detail:
            #     print("---------------------------------------------")
            #     state.showBoard()
            # print(state, state.current_player, self.player_color)
            # print("////////////////////////////////////////////////////////////////")

        end = isEndGame(state)
        if end is None:
            node.expand(state.availables(), state.current_player, state.n)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        node.back_update(leaf_value)

    def _evaluate_rollout(self, state: OthelloGame, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        for _ in range(limit):
            end = isEndGame(state)
            if end is not None:
                break
            position = rollout_policy_fn(state)
            state.move(position)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if end == 0:  # tie
            return 0
        else:
            return 1 if end == self.player_color else -1

    def get_move(self, state, time_):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        for n in range(self._n_playout):
            if time.time() - time_ >= self.time_limit:
                print(f"time out ! playout: {n}")
                break

            self._playout(copy.deepcopy(state))

        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]
    
    def update_with_move(self, last_move_str, player_color):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        last_move: 更新tree至當前狀態(str) (list)(可能不只一步)
        """
        # update = False
        if self._root is None:
            self.reset_root(player_color)
            return
        # print("last_move_str: ", last_move_str, type(last_move_str))
        if len(last_move_str) == 1 and self._root._children != {}:
            if last_move_str[0] in self._root._children:
                self._root = self._root._children[last_move_str[0]]
                self._root._parent = None
                return

        print("reset root")
        self.reset_root(player_color)

    def __str__(self):
        return "MCTS"


class MCTS_BOT(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=3, n_playout=300, n=12, time_limit=2.8):
        self.mcts = MCTS(c_puct, n_playout, time_limit)
        self.game = OthelloGame(n)
        self.player = None
        self.n = n
        self.time = None

    def getAction(self, board, color):
        self.time = time.time()
        self.player = color # 設定我方顏色
        # 判斷是否已經進行下一局
        # 更新tree至當前狀態
        self.mcts.update_with_move(find_opp_move(self.game, board), color)

        self.game[:] = board[:] # 更新盤面
        # print(self.game.showBoard())
        self.game.current_player = self.player # 設定當前玩家
        move_key = self.mcts.get_move(self.game, self.time) # 取得MCTS的落子(str)
        move = str_2_np(move_key) # 轉換成座標
        self.game.move(move) # 更新盤面(我方落子)
        
        # print(self.game.showBoard())
        # print(move)
        # print("=========================================================")
        
        self.mcts.update_with_move([move_key], color) # 更新tree至當前狀態
        return move

    def __str__(self):
        return "MCTS {}".format(self.player)
