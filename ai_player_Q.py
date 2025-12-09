import copy
import socket
import time
import random
import argparse
import pickle
import numpy as np
import math
from classes import Action, Strategy, QTable, Player, get_card_info, get_action_name
from config import PORT, BET, INITIAL_MONEY, N_DECKS


# RETRY関連設定 (一部CLIで上書き可)
RETRY_MAX = 10              # 1ゲームあたりRETRY最大回数 (後で --max_retries_per_game で上書き)
RETRY_BUCKET_MAX = 5         # 状態に含めるRETRY回数バケットの上限
RETRY_PENALTY_SCALE = 0.3    # ペナルティエスカレーション係数 (後で --retry_penalty_scale で上書き)


### グローバル変数 ###

# ゲームごとのRETRY回数のカウンター
g_retry_counter = 0

# プレイヤークラスのインスタンスを作成
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)

# ディーラーとの通信用ソケット
soc = None
g_dealer_host = 'localhost'

# Q学習用のQテーブル
q_table = QTable(action_class=Action, default_value=0)

# Q学習の設定値（デフォルト）
EPS = 0.3 # ε-greedyにおけるε
LEARNING_RATE = 0.1 # 学習率
DISCOUNT_FACTOR = 0.9 # 割引率
EPS_START = EPS
EPS_END = 0.05
EPS_DECAY_EPISODES = 1000
EPS_DECAY_TYPE = 'linear'  # 'const'|'linear'|'exp'

### 関数 ###

# ゲームを開始する
def game_start(game_ID=0, verbose=True):
    global g_retry_counter, player, soc, g_dealer_host

    if verbose:
        print('Game {0} start.'.format(game_ID))
        print('  money: ', player.get_money(), '$')

    # RETRY回数カウンターの初期化
    g_retry_counter = 0

    # ディーラープログラムに接続する（ホスト優先順位 + リトライ強化 + タイムアウト）
    max_connect_attempts = 60
    base_hosts = [g_dealer_host, '127.0.0.1', 'localhost', socket.gethostname()]
    # dedupe while preserving order
    hosts_to_try = []
    for h in base_hosts:
        if h and h not in hosts_to_try:
            hosts_to_try.append(h)
    connected = False
    last_err = None
    for attempt in range(max_connect_attempts):
        random.shuffle(hosts_to_try)
        for host in hosts_to_try:
            try:
                s = socket.create_connection((host, PORT), timeout=2.5)
                try:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                except Exception:
                    pass
                s.settimeout(None)
                soc = s
                connected = True
                break
            except OSError as e:
                last_err = e
                continue
        if connected:
            break
        # stagger retries to reduce thundering herd
        time.sleep(0.15 + random.random() * 0.5)
    if not connected:
        raise OSError(f'Failed to connect to dealer after {max_connect_attempts} attempts: {last_err}')

    # ベット
    bet, money = player.set_bet()
    if verbose:
        print('Action: BET')
        print('  money: ', money, '$')
        print('  bet: ', bet, '$')

    # ディーラーから「カードシャッフルを行ったか否か」の情報を取得
    # シャッフルが行われた場合は True が, 行われなかった場合は False が，変数 cardset_shuffled にセットされる
    # なお，本サンプルコードではここで取得した情報は使用していない
    cardset_shuffled = player.receive_card_shuffle_status(soc)
    if cardset_shuffled and verbose:
        print('Dealer said: Card set has been shuffled before this game.')

    # ディーラーから初期カード情報を受信
    dc, pc1, pc2 = player.receive_init_cards(soc)
    if verbose:
        print('Dealer gave cards.')
        print('  dealer-card: ', get_card_info(dc))
        print('  player-card 1: ', get_card_info(pc1))
        print('  player-card 2: ', get_card_info(pc2))
        print('  current score: ', player.get_score())

# 現時点での手札情報（ディーラー手札は見えているもののみ）を取得
def get_current_hands():
    return copy.deepcopy(player.player_hand), copy.deepcopy(player.dealer_hand)

# HITを実行する
def hit(verbose=True):
    global player, soc

    if verbose:
        print('Action: HIT')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'hit')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    if verbose:
        print('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
        print('  current score: ', score)

    # バーストした場合はゲーム終了
    if status == 'bust':
        if verbose:
            for i in range(len(dc)):
                print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
            print("  dealer's score: ", player.get_dealer_score())
        soc.close() # ディーラーとの通信をカット
        reward = player.update_money(rate=rate) # 所持金額を更新
        if verbose:
            print('Game finished.')
            print('  result: bust')
            print('  money: ', player.get_money(), '$')
        return reward, True, status

    # バーストしなかった場合は続行
    else:
        return 0, False, status

# STANDを実行する
def stand(verbose=True):
    global player, soc

    if verbose:
        print('Action: STAND')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'stand')

    # ディーラーから情報を受信
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    if verbose:
        print('  current score: ', score)
        for i in range(len(dc)):
            print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
        print("  dealer's score: ", player.get_dealer_score())

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    if verbose:
        print('Game finished.')
        print('  result: ', status)
        print('  money: ', player.get_money(), '$')
    return reward, True, status

# DOUBLE_DOWNを実行する
def double_down(verbose=True):
    global player, soc

    if verbose:
        print('Action: DOUBLE DOWN')

    # 今回のみベットを倍にする
    bet, money = player.double_bet()
    if verbose:
        print('  money: ', money, '$')
        print('  bet: ', bet, '$')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'double_down')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    if verbose:
        print('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
        print('  current score: ', score)
        for i in range(len(dc)):
            print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
        print("  dealer's score: ", player.get_dealer_score())

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    if verbose:
        print('Game finished.')
        print('  result: ', status)
        print('  money: ', player.get_money(), '$')
    return reward, True, status

# SURRENDERを実行する
def surrender(verbose=True):
    global player, soc

    if verbose:
        print('Action: SURRENDER')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'surrender')

    # ディーラーから情報を受信
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    if verbose:
        print('  current score: ', score)
        for i in range(len(dc)):
            print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
        print("  dealer's score: ", player.get_dealer_score())

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    if verbose:
        print('Game finished.')
        print('  result: ', status)
        print('  money: ', player.get_money(), '$')
    return reward, True, status

# RETRYを実行する
def retry(verbose=True):
    global player, soc
    if verbose:
        print('Action: RETRY')
    # エスカレートするペナルティ: base=BET/4, (1 + scale * 現在までのRETRY回数)
    base = player.current_bet / 4.0
    penalty = int(base * (1.0 + RETRY_PENALTY_SCALE * g_retry_counter))
    player.consume_money(penalty)
    if verbose:
        print('  player-card {0} has been removed.'.format(player.get_num_player_cards()))
        print('  money: ', player.get_money(), '$')
    player.send_message(soc, 'retry')
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)
    if verbose:
        print('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
        print('  current score: ', score)
    if status == 'bust':
        if verbose:
            for i in range(len(dc)):
                print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
            print("  dealer's score: ", player.get_dealer_score())
        soc.close()
        reward = player.update_money(rate=rate)
        if verbose:
            print('Game finished.')
            print('  result: bust')
            print('  money: ', player.get_money(), '$')
        return reward - penalty, True, status
    else:
        return -penalty, False, status

# 行動の実行
def act(action: Action, verbose=True):
    if action == Action.HIT:
        return hit(verbose=verbose)
    elif action == Action.STAND:
        return stand(verbose=verbose)
    elif action == Action.DOUBLE_DOWN:
        return double_down(verbose=verbose)
    elif action == Action.SURRENDER:
        return surrender(verbose=verbose)
    elif action == Action.RETRY:
        return retry(verbose=verbose)
    else:
        exit()


### これ以降の関数が重要 ###

# 現在の状態の取得
def get_state():
    p_hand, _ = get_current_hands()
    score = p_hand.get_score()
    length = p_hand.length()
    retry_bucket = min(g_retry_counter, RETRY_BUCKET_MAX)
    return (score, length, retry_bucket)

# 行動戦略
def select_action(state, strategy: Strategy, epsilon: float = None):
    global q_table

    # Q値最大行動を選択する戦略
    if strategy == Strategy.QMAX:
        return q_table.get_best_action(state)

    # ε-greedy
    elif strategy == Strategy.E_GREEDY:
        eps = EPS if epsilon is None else epsilon
        if np.random.rand() < eps:
            return select_action(state, strategy=Strategy.RANDOM)
        else:
            return q_table.get_best_action(state)

    # ランダム戦略
    else:
        # ランダム戦略: RETRY上限到達時は除外
        actions = [Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER]
        if g_retry_counter < RETRY_MAX:
            actions.append(Action.RETRY)
        return random.choice(actions)


### ここから処理開始 ###

def main():
    global g_retry_counter, player, soc, q_table, RETRY_MAX, RETRY_PENALTY_SCALE

    parser = argparse.ArgumentParser(description='AI Black Jack Player (Q-learning)')
    parser.add_argument('--games', type=int, default=1, help='num. of games to play')
    parser.add_argument('--history', type=str, default='play_log.csv', help='filename where game history will be saved')
    parser.add_argument('--load', type=str, default='', help='filename of Q table to be loaded before learning')
    parser.add_argument('--save', type=str, default='', help='filename where Q table will be saved after learning')
    parser.add_argument('--testmode', help='this option runs the program without learning', action='store_true')
    parser.add_argument('--alpha', '--learning_rate', type=float, default=LEARNING_RATE, help='learning rate (alpha)')
    parser.add_argument('--gamma', '--discount_factor', type=float, default=DISCOUNT_FACTOR, help='discount factor (gamma)')
    parser.add_argument('--eps_start', type=float, default=EPS_START, help='epsilon start for e-greedy')
    parser.add_argument('--eps_end', type=float, default=EPS_END, help='epsilon end for decay')
    parser.add_argument('--eps_decay_episodes', type=int, default=EPS_DECAY_EPISODES, help='episodes over which to decay epsilon')
    parser.add_argument('--eps_decay_type', choices=['const', 'linear', 'exp'], default=EPS_DECAY_TYPE, help='epsilon decay type')
    parser.add_argument('--eps_log', type=str, default='', help='optional CSV filename to append per-episode records: game_id,eps,total_reward')
    parser.add_argument('--max_retries_per_game', type=int, default=RETRY_MAX, help='hard cap of RETRY actions per game')
    parser.add_argument('--retry_penalty_scale', type=float, default=RETRY_PENALTY_SCALE, help='scaling factor for escalating retry penalty')
    parser.add_argument('--quiet', action='store_true', help='suppress per-action verbose logs for faster long runs')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    parser.add_argument('--dealer_host', type=str, default='localhost', help='dealer host to connect (default: localhost)')
    args = parser.parse_args()

    n_games = args.games + 1
    learning_rate = args.alpha
    discount_factor = args.gamma
    eps_start = args.eps_start
    eps_end = args.eps_end
    eps_decay_episodes = max(1, args.eps_decay_episodes)
    eps_decay_type = args.eps_decay_type
    g_dealer_host = args.dealer_host

    # シード設定
    if args.seed is not None:
        try:
            random.seed(args.seed)
            np.random.seed(args.seed)
        except Exception:
            pass

    # Qテーブルをロード
    if args.load != '':
        # load can be either legacy (table dict) or new format {'meta':..., 'table':...}
        try:
            with open(args.load, 'rb') as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict) and 'table' in loaded:
                q_table.table = loaded['table']
                print(f'Loaded Q-table (with meta) from {args.load}')
            elif isinstance(loaded, dict):
                q_table.table = loaded
                print(f'Loaded legacy Q-table dict from {args.load}')
            else:
                # fallback to QTable.load (in case other formats)
                q_table.load(args.load)
                print(f'Loaded Q-table via QTable.load from {args.load}')
            # 既存キーが2次元 (score,length) の場合は (score,length,0) に変換
            converted = {}
            legacy_dim = False
            for k, v in q_table.table.items():
                if isinstance(k, tuple) and len(k) == 2:
                    converted[(k[0], k[1], 0)] = v
                    legacy_dim = True
                else:
                    converted[k] = v
            if legacy_dim:
                q_table.table = converted
                print('[INFO] Expanded legacy state keys to 3D (retry_bucket=0)')
        except Exception as e:
            print(f'Warning: failed to load Q-table from {args.load}: {e}')

    # ログファイルを開き、必ず閉じられるよう with 構文で処理
    with open(args.history, 'w', encoding='utf-8') as logfile:
        print('score,hand_length,retry_bucket,action,status,reward', file=logfile)

        # n_games回ゲームを実行
        # RETRY設定を反映
        RETRY_MAX = max(0, args.max_retries_per_game)
        RETRY_PENALTY_SCALE = max(0.0, args.retry_penalty_scale)

        for n in range(1, n_games):

            # nゲーム目を開始
            game_start(n, verbose=not args.quiet)

            # 「現在の状態」を取得
            state = get_state()

            # エピソードごとの epsilon を計算
            if eps_decay_type == 'const':
                current_eps = eps_start
            elif eps_decay_type == 'linear':
                fraction = min(1.0, (n - 1) / float(eps_decay_episodes))
                current_eps = eps_start + fraction * (eps_end - eps_start)
            elif eps_decay_type == 'exp':
                tau = max(1.0, eps_decay_episodes / 5.0)
                current_eps = eps_end + (eps_start - eps_end) * math.exp(-(n - 1) / tau)
            else:
                current_eps = eps_start

            while True:

                # 次に実行する行動を選択
                if args.testmode:
                    action = select_action(state, Strategy.QMAX)
                else:
                    action = select_action(state, Strategy.E_GREEDY, epsilon=current_eps)
                if g_retry_counter >= RETRY_MAX and action == Action.RETRY:
                    action = random.choice([Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER])
                action_name = get_action_name(action) # 行動名を表す文字列を取得

                # 選択した行動を実際に実行
                # 戻り値:
                #   - done: 終了フラグ．今回の行動によりゲームが終了したか否か（終了した場合はTrue, 続行中ならFalse）
                #   - reward: 獲得金額（ゲーム続行中の場合は 0 , ただし RETRY を実行した場合は1回につき -BET/4 ）
                #   - status: 行動実行後のプレイヤーステータス（バーストしたか否か，勝ちか負けか，などの状態を表す文字列）
                reward, done, status = act(action, verbose=not args.quiet)

                # 実行した行動がRETRYだった場合はRETRY回数カウンターを1増やす
                if action == Action.RETRY:
                    g_retry_counter += 1

                # 「現在の状態」を再取得
                prev_state = state # 行動前の状態を別変数に退避
                prev_score = prev_state[0] # 行動前のプレイヤー手札のスコア（prev_state の一つ目の要素）
                state = get_state()
                score = state[0] # 行動後のプレイヤー手札のスコア（state の一つ目の要素）

                # Qテーブルを更新
                if not args.testmode:
                    _, V = q_table.get_best_action(state, with_value=True)
                    Q = q_table.get_Q_value(prev_state, action) # 現在のQ値
                    Q = (1 - learning_rate) * Q + learning_rate * (reward + discount_factor * V) # 新しいQ値
                    q_table.set_Q_value(prev_state, action, Q) # 新しいQ値を登録

                # ログファイルに「行動前の状態」「行動の種類」「行動結果」「獲得金額」などの情報を記録
                print('{},{},{},{},{},{}'.format(prev_state[0], prev_state[1], prev_state[2], action_name, status, reward), file=logfile)

                # 終了フラグが立った場合はnゲーム目を終了
                if done == True:
                    break

            if not args.quiet:
                print('')

    # Qテーブルをセーブ (新仕様: 保存にメタ情報を付与する)
    if args.save != '':
        try:
            meta = {
                'alpha': learning_rate,
                'gamma': discount_factor,
                'eps_start': eps_start,
                'eps_end': eps_end,
                'eps_decay_episodes': eps_decay_episodes,
                'eps_decay_type': eps_decay_type,
                'games': args.games,
            }
            to_save = {'meta': meta, 'table': q_table.table}
            with open(args.save, 'wb') as f:
                pickle.dump(to_save, f)
            print(f'Saved Q-table with meta to {args.save}')
        except Exception as e:
            print(f'Warning: failed to save Q-table to {args.save}: {e}')


if __name__ == '__main__':
    main()
