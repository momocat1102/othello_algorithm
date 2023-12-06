
from .constant import *  
from . import Base_bot
import numpy as np
import tensorflow as tf
from make_tree import make_tree as mt
import copy
from bridge_game_functions import show_card, trump_table
PLAYER_SELF = 0
PLAYER_OPPO = 1

REVEALED = 4
OPPO_CARD = 5

p_map = {0:"我方", 1:"對方"}
show_detail = False

class VirtualState():
    def __init__(self, state, decraler, revealed_card, oppo_card, trump):
        
        self.trump    = trump
        self.decraler = decraler
        
        self.my_hand     = list(state[0])
        self.oppo_hand   = list(state[1])
        self.unknown     = list(state[2])
        
        # 當前玩家一開始一定是自己
        self.curr_player = PLAYER_SELF
        self.off_change  = oppo_card
        self.revealed_card = revealed_card
        
        self._deal_state()
        
    def _deal_state(self):


        self.oppo_hand.sort()
        np.random.shuffle(self.unknown)
        
    def do_move(self, move):
        
            
        if self.curr_player == PLAYER_SELF:
            self.my_hand.remove(move)
              
        elif self.curr_player == PLAYER_OPPO:
            self.oppo_hand.remove(move)  
        
        # 當前為先手
        if self.off_change == -1:
            
            self.off_change = move
            # 先手出牌後換後手
            self.curr_player = 1 - self.curr_player
        
        else:# 當前為後手 
            # 比較大小
            
            if self.cmp(self.off_change, move):
                winner = 1-self.curr_player
            else:
                winner = self.curr_player
            
            # 贏家拿走翻開的牌，輸家再抽一張牌 
            next_card = self.unknown.pop(0)
            
            if show_detail:
                print("winner:", p_map[winner])
                print("win get:", show_card(self.revealed_card))
                print("lose get:", show_card(next_card))
            
            if winner == PLAYER_SELF:
                self.my_hand.append(self.revealed_card)
                self.oppo_hand.append(next_card)
            else:
                self.my_hand.append(next_card)
                self.oppo_hand.append(self.revealed_card)
            self.my_hand.sort()
            self.oppo_hand.sort()
            if show_detail:
                print("my_hand:", show_card(self.my_hand).replace("\n", " "))
                print("oppo_hand:", show_card(self.oppo_hand).replace("\n", " "))
                print("unknown:", end=" ")
                for i in self.unknown:
                    print(show_card(i), end=" ")
                print()
            
            # 再翻一張牌並重置回合
            if len(self.unknown) > 0:
                self.revealed_card = self.unknown.pop(0)
            
            self.off_change  = -1
            self.curr_player = winner        
            
    def cmp(self, offensive, defensive):
        
        if defensive//13 == offensive//13: #同花
            return offensive > defensive
        else:
            return defensive//13 != self.trump #防守方不是trump，return True
        
    def get_avliable_moves(self):
        
        if self.curr_player == PLAYER_SELF:    
            if self.off_change != -1:
                can_play = [x for x in self.my_hand if self.off_change//13 == x//13]
                
                if len(can_play) != 0:
                    return can_play
        
            return list(self.my_hand)
        
        else:# 當前為對手
            if self.off_change != -1:
                can_play = [x for x in self.oppo_hand if self.off_change//13 == x//13]
                
                if len(can_play) != 0:
                    return can_play
                
            return list(self.oppo_hand)
        
    def get_inputs_for_model(self):
        
        state = np.ones((55)) * USED
        state[self.unknown] = UNKNOWN
        if self.curr_player == PLAYER_SELF:
            state[self.my_hand] = ON_MYHAND
            state[self.oppo_hand] = ON_OPPONENTHAND
        else:
            state[self.my_hand] = ON_OPPONENTHAND
            state[self.oppo_hand] = ON_MYHAND
        
        state[-3] = self.off_change
        state[-2] = self.revealed_card
        state[-1] = self.trump
        
        return state[np.newaxis, :]
    
    def game_end(self):
        """
            判斷遊戲是否結束
            return: (is_end, my_value)
            is_end為True表示是否遊戲結束，my_value為(自己的墩數-對手的墩數)/13，如果遊戲沒結束，my_value為None
        """
        if len(self.unknown) == 0:
            #print("遊戲結束")
            if self.decraler == PLAYER_SELF:
                my_score, oppo_score = mt.mk_tree_score_only(self.my_hand, self.oppo_hand, self.trump, {})
            else:
                oppo_score, my_score = mt.mk_tree_score_only(self.oppo_hand, self.my_hand, self.trump, {})
                
            return True, (my_score - oppo_score)/13
        
        return False, None
        

class TreeNode():
    """
    mcts搜索樹中的節點。樹的子節點字典中，key為動作，值為TreeNode。每個節點跟踪其自身的
    Q，先驗概率 P 及其訪問次數調整的 u。
    """

    def __init__(self, parent, prior_p, curr_player, act):
        """
            parent: 當前節點的父節點
            prior_p:  當前節點的被選擇的機率，由模型預測得到
        """
        self._parent = parent
        self._children = {} # 動作 -> TreeNode
        self._n_visits = 0  # 當前節點被訪問的次數
        self._Q = 0         # 當前節點的平均價值
        self._P = prior_p
        self.curr_player = curr_player
        self.act = act # 當前節點的動作，目前只是觀察過程時會print
        
    def expand(self, action_priors, next_player):
        """ 
            action_priors: (action, prob)， 所有可以出的牌和其機率
            next_player: 下一個出牌的人
            根據以上參數擴展節點
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob, next_player, action)


    def select(self, c_puct):
        """
        選擇Q+U最大的節點
        return: (action, next_node)
        """
        if show_detail and 1:
            for action, child in self._children.items():
                print(show_card(action),
                    "n_visits:", child._n_visits,
                    "P:{:.2f}".format(float(child._P)),
                    "Q:{:.2f}".format(float(child._Q)),
                    "u:{:.3f}".format(c_puct * child._P * np.sqrt(child._parent._n_visits) / (1 + child._n_visits)) 
                    )
            
        return max(self._children.items(), key=lambda node: node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """
            計算節點的PUCT值
        """
        u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        
        return self._Q + u

    def back_update(self, leaf_value):
        """ 
            反向去更所有直系節點的Q值和訪問次數
            leaf_value: 從葉節點到當前節點的分數，leaf_value自己的分數，若當前節點為對手，則leaf_value為13-leaf_value   
        """
        curr_value = leaf_value
        self._n_visits += 1
        
        if self.curr_player == PLAYER_OPPO:
            curr_value = -1. * curr_value
            
        if show_detail:  
            if self.act is not None:
                print(show_card(self.act), end=": ")
            else:
                print("root", end=": ")
            print(p_map[self.curr_player], end=" ")
            print(f"Q:{float(self._Q):.2f} -> {float(self._Q+((curr_value - self._Q) / self._n_visits)):.2f}", end=" ")
        # 更新Q值
        self._Q += 1.0 * (curr_value - self._Q) / self._n_visits
       
        if self._parent:
            if show_detail: print(" =>", end=" ")
            self._parent.back_update(leaf_value)

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

# 蒙地卡羅搜索樹
class MCTS(object):

    def __init__(self, policy_value_model, c_puct=5, n_simulation=2000):
       
        self._root = None
        self.policy_value_model = policy_value_model
        self._c_puct = c_puct
        self._n_simulation = n_simulation # 每次搜索的次数

    def _simulation(self, state:VirtualState):
        """
        進行一次搜索
        """
        node = self._root
        while True:
            
            if node.is_leaf():
                break
            
            if show_detail:
                print(f"curr_player: {p_map[state.curr_player]} --", end=" ")
                if state.off_change == -1:
                    print("為先手")
                else:
                    print("為後手")
                
                print("child: ", end="")
                for act, child in node._children.items():
                    print(show_card(act), end=" ")
                print()  
            
            # 非葉節點，繼續用PUCT算法選擇子節點
            action, node = node.select(self._c_puct)
            if show_detail:
                print("select", show_card(action))
            # 進行選擇的動作
            state.do_move(action)
                
        # 判斷是否結束
        if show_detail:
            if node.act is None:
                print("node root is leaf")
            else:
                print(f"node {show_card(node.act)} is leaf")
        
        end, my_score = state.game_end()
        
        if not end:
            # 預測當前節點的動作機率和分數
            action_probs, leaf_value = self.get_model_policy(state)
            if show_detail:
                print("predict curr node value: {:.2f}".format(float(leaf_value)))
            node.expand(action_probs, state.curr_player)
        else:
            if show_detail:
                print("is end, my_score:", my_score)
            leaf_value = my_score
            
        # 完成一次搜索，反向更新分數和訪問次數
        if show_detail:print("Backpropagation:", end=" ")
        node.back_update(leaf_value)
    
        
    def get_model_policy(self, state:VirtualState):
        
        avliable_moves = state.get_avliable_moves()
        if show_detail:
            print("not end, expand avliable_moves:", show_card(avliable_moves).replace("\n", " "))
        
        model_inputs  = state.get_inputs_for_model() 
        
        probs, value = self.policy_value_model(model_inputs)
        probs = np.squeeze(probs, 0)
 
        act_probs = np.zeros(52)
        act_probs[avliable_moves] = probs[avliable_moves]
        act_probs /= np.sum(act_probs)
        
        avliable_act_probs = [(move, act_probs[move]) for move in avliable_moves]
        
        return avliable_act_probs, value.numpy()[0][0]
        
    def get_move_probs(self, state, temp=1e-3):
        """
        進行n_simulation次搜索，然後返回所有動作和其對應的機率
        state:當前的遊戲狀態
        temp:介於(0, 1]之間的溫度參數，控制探索的程度，temp越小，越趨向於選擇概率最大的動作，也就是探索程度越低
        """
        for n in range(self._n_simulation):
            
            if show_detail: print(f"\n########### simulation {n} ###########")
            
            self._simulation(copy.deepcopy(state))

        # 跟据根节点处的访问计数来计算移动概率
            
        act_visits= [(act, node._n_visits)
                     for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        
        act_probs = tf.nn.softmax(1.0 / temp * np.log(np.array(visits,dtype=np.float32) + 1e-10)).numpy()
        
        return acts, act_probs

    # def update_with_move(self, last_move):
    #     """
    #     在当前的树上向前一步，保持我们已经直到的关于子树的一切
    #     """
    #     if last_move in self._root._children:
    #         self._root = self._root._children[last_move]
    #         self._root._parent = None
    #     else:
    #         self._root = TreeNode(None, 1.0)
    def set_root(self):
        self._root = TreeNode(None, 1.0, PLAYER_SELF, None)

    def __str__(self):
        return 'MCTS'


class MCTSBot(Base_bot.BaseBot):
    
    def __init__(self, is_selfplay=False, policy_value_model=None, n_simulation=2000, n_tree=3, c_puct=5, name="None"):
        super().__init__(name=name)
        
        self.is_selfplay = is_selfplay
        self.mcts = MCTS(policy_value_model, c_puct, n_simulation)
        self.n_simulation = n_simulation
        self.unknown = [i for i in range(0, 52)]
        self.decraler = None
        self.tree_table = {}
        self.n_tree = n_tree
        self.tree_table = {2:6,
                             4:5,
                             6:5,
                             8:5,
                             10:5}
        
    def reset(self):
        super().reset()
        self.unknown = [i for i in range(0, 52)]
        self.decraler = None
        self.mcts._n_simulation = self.n_simulation
        
    def deal_init(self):
        
        self.my_hand.sort()
        
        for i in range(len(self.my_hand)):
            self.card_state[self.my_hand[i]] = ON_MYHAND # 將自己的手牌設為已知狀態
            self.unknown.remove(self.my_hand[i])         # 從未知牌中移除自己的手牌
            
    def change_card(self, revealed_card, oppo_card, return_probs=False):
        """
            revealed_card: 翻開的牌
            oppo_card: 對手換出的牌，如果是先手，oppo_card為-1
        """
        if self.decraler == None:
            self.decraler = PLAYER_SELF if oppo_card == -1 else PLAYER_OPPO
        
        self.my_hand.sort()
        self.opponent_hand.sort()
        
        self.mcts.set_root()
        my_change, probs = self.get_action(revealed_card, oppo_card)
     
        
        self.my_hand.remove(my_change) 
        self.card_state[my_change] = USED
        
        self.curr_revealed_card = revealed_card
       
        if return_probs:
            return my_change, probs
        else:
            return my_change
    
    def dealchange(self, myget, oppo_change):
        '''
            myget:       自己得到的牌
            oppo_change: 對手在這次換牌所出的牌
        '''
        self.my_hand.append(myget)
        self.my_hand.sort()
        self.card_state[myget] = ON_MYHAND # 設定得到的牌為自己手牌
        
        self.unknown.remove(self.curr_revealed_card)
        
        if myget != self.curr_revealed_card:# 如果得到的牌不是剛剛翻開的牌，就表示被對手拿走了
            
            self.unknown.remove(myget)
            
            self.opponent_hand.append(self.curr_revealed_card) 
            self.opponent_hand.sort() # 將對手的手牌排序
            self.card_state[self.curr_revealed_card] = ON_OPPONENTHAND
      
        if oppo_change in self.opponent_hand:
            self.opponent_hand.remove(oppo_change) # 從對手手牌中移除對手換出的牌
        else:
            self.unknown.remove(oppo_change)       # 如果對手換出的牌不在對手手牌中，就從未知牌中移除
      
        self.card_state[oppo_change] = USED # 設定對手換出的牌為已出
        self.n_remain -= 2 # 剩餘牌數減2 

    def get_action(self, revealed_card, oppo_card):
        
        # 用當前的狀態來建立一個虛擬狀態，用來進行MCTS搜索
        

        #print(f"{self.n_remain} {self.name} start searching")
        
        if self.n_remain <= 10:
            self.mcts._n_simulation = max(int(self.mcts._n_simulation*0.7), 20)
            
        # global show_detail
        # if self.n_remain == 10:
        #    show_detail = True
        # else:
        #    show_detail = False   
        n_states = self.n_tree + self.tree_table[self.n_remain] if self.n_remain <= 10 else self.n_tree
        states = self.allocate_card(revealed_card, oppo_card, n_states=n_states, bad_level=1)
        total_probs = None
        for i in range(n_states):
            
            state_for_mcts = VirtualState(states[i], self.decraler, revealed_card, oppo_card, self.trump) 
            acts, probs = self.mcts.get_move_probs(state_for_mcts,temp=0.001)
            self.mcts.set_root()
            total_probs = probs if total_probs is None else total_probs + probs   
                
        total_probs /= n_states
        
        if self.is_selfplay:
            # 加入Dirichlet Noise
            p = 0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones_like(total_probs, dtype=np.float32))
            move = np.random.choice(acts, p=p/p.sum())
        else:
            move = np.random.choice(acts, p=total_probs)
            
        move = int(move)
        
        move_probs = np.zeros(52)
        move_probs[list(acts)] = total_probs
        return move, move_probs
    
    def allocate_card(self, revealed_card, oppo_card, n_states=3, bad_level=2) -> list:
        """分配未知牌給對手，並返回n_states組不同的狀態
            n_states: 要分配多少組不同的狀態
            bad_level: 設的越大，越容易分配給對手較大的牌
        """
        allot_len = 13 if oppo_card == -1 else 12 

      
        unknown = list(self.unknown)    
        unknown.remove(revealed_card)
        
        if oppo_card in unknown:
            unknown.remove(oppo_card)
            
        oppo_hand = list(self.opponent_hand)
        if oppo_card in oppo_hand:
            oppo_hand.remove(oppo_card) 
        
        # 計算未知牌的分數
        unknown = np.array(unknown)
        unknown_score = unknown % 13 + 1
        unknown_score = np.where(unknown // 13 == self.trump, unknown_score+5, unknown_score)*bad_level 
        unknown_prob = unknown_score / unknown_score.sum()
        
        states = []
        for _ in range(n_states):
            # 選出要為對手補充的牌
            fill_cards = np.random.choice(unknown, allot_len-len(oppo_hand), replace=False, p=unknown_prob).tolist()
            
            # 更新對手的手牌和未知牌
            new_oppo_hand = oppo_hand + fill_cards
            new_unknown = list(set(unknown)-set(fill_cards))
            
            # 將三種牌組成一個狀態
            states.append([list(self.my_hand), new_oppo_hand, new_unknown])
            
        # if show_detail or 1:
        #     for i in range(n_states):
        #         print(f"state {i}:")
        #         print("oppo_hand:", show_card(new_oppo_hand[i]).replace("\n", " "))
        #         print("unknown:", show_card(new_unknown[i]).replace("\n", " "))          
        return states
    
    def play(self, oppo_play=-1):
       
        self.opponent_hand.sort()
        self.my_hand.sort()  
     
        #先手
        if oppo_play == -1:
            #concat_map 是雙向的字典，可以從index對應到card，也可以從card對應到index，但是index對應到card時key要加100
            #而index代表的是該牌在雙方所有手牌中由小到大的排序
            key, concat_map = mt.get_key_map(self.my_hand, self.opponent_hand, self.trump)
            if key not in self.tree_table.keys():
                mt.mk_tree(self.my_hand, self.opponent_hand, self.trump, self.tree_table)
            
            my_play = concat_map[self.tree_table[key][2]+100]
        #後手
        else:
            key, concat_map = mt.get_key_map(self.opponent_hand, self.my_hand, self.trump)
            if key not in self.tree_table.keys():
                mt.mk_tree(self.opponent_hand, self.my_hand, self.trump, self.tree_table)
            
            my_play = concat_map[self.tree_table[key][3][concat_map[oppo_play]]+100]
        self.my_hand.remove(my_play)
        return my_play