from AIGamePlatform import Othello
from othello.bots.Random import BOT
from othello.bots.mcts_pure import MCTS_BOT
from othello.bots.alpha_beta import Alpha_Beta_BOT

app=Othello() # 和平台建立WebSocket連線
# bot_alpha = Alpha_Beta_BOT(n=12) # 建立隨機bot
bot_mcts = MCTS_BOT(n_playout=1100, n=12, time_limit=3.05)

@app.competition(competition_id='test_12x12') # 競賽ID
def _callback_(board, color): # 當需要走步會收到盤面及我方棋種
    # print(board, color)
    # return bot_alpha.getAction(board, color) # bot回傳落子座標
    return bot_mcts.getAction(board, color) # bot回傳落子座標

