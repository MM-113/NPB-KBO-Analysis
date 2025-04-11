# -*- coding: utf-8 -*-
# 檔案名稱：NPB_KBO_Complete_Analysis.py
import numpy as np
from scipy.stats import nbinom, poisson
import sys

# 常數設定
NPB_STD_DEV = 2.3  # 日本職棒標準差
NPB_NB_R = 9.0     # 日本職棒負二項分布參數
KBO_STD_DEV = 2.8  # 韓國職棒標準差
KBO_NB_R = 7.5     # 韓國職棒負二項分布參數
NUM_SIMULATIONS = 100000  # 模擬次數

def safe_input(prompt, input_type=float, default=None, min_val=None, max_val=None):
    """防呆輸入函數"""
    while True:
        try:
            user_input = input(prompt).strip()
            if default is not None and user_input == "":
                return default
            val = input_type(user_input)
            if min_val is not None and val < min_val:
                print(f"輸入值不能小於 {min_val}！")
                continue
            if max_val is not None and val > max_val:
                print(f"輸入值不能大於 {max_val}！")
                continue
            return val
        except ValueError:
            print("請輸入有效的數值！")
        except KeyboardInterrupt:
            print("\n輸入已取消")
            sys.exit(0)

def input_team_data(team_name, time_period, is_home=True):
    """輸入球隊數據"""
    while True:
        try:
            print(f"\n【{team_name} {'主' if is_home else '客'}隊數據】({time_period})")
            team = {
                'name': team_name,
                'time_period': time_period,
                'time_avg': safe_input(f"{time_period}場均得分: ", min_val=0),
                'base_avg': safe_input("整體場均得分: ", min_val=0),
                'allow': safe_input("場均失分: ", min_val=0),
                'over_rate': safe_input("大分過盤率(%): ", min_val=0, max_val=100) / 100,
                'team_batting': safe_input("團隊打擊率(0.000-1.000): ", min_val=0, max_val=1),
                'team_obp': safe_input("團隊上壘率(0.000-1.000): ", min_val=0, max_val=1),
                'pitcher': {
                    'era': safe_input("先發投手防禦率: ", min_val=0),
                    'baa': safe_input("先發投手被打擊率(0.000-1.000): ", min_val=0, max_val=1)
                }
            }
            
            # 數據合理性檢查
            if team['time_avg'] > team['base_avg'] * 1.5:
                print(f"警告: {time_period}場均得分({team['time_avg']})明顯高於整體場均({team['base_avg']})")
                if not input("是否確認無誤?(y/n): ").lower().startswith('y'):
                    continue
            return team
        except Exception as e:
            print(f"輸入錯誤: {str(e)}，請重新輸入")

def calculate_scores(home, away, league, include_effects=True):
    """計算得分（可選擇是否包含投手和團隊影響）"""
    try:
        league_factor = 0.95 if league == 'NPB' else 1.05
        
        if include_effects:
            # 完整模型計算（含投手和團隊影響）
            def weighted_score(data, opp_pitcher):
                # 投手影響計算
                era_impact = (4.5 / max(0.01, opp_pitcher['era'])) * 0.7
                baa_impact = (0.3 / max(0.001, opp_pitcher['baa'])) * 0.3
                pitcher_factor = min(2.0, era_impact + baa_impact)
                
                # 團隊影響計算
                batting_impact = (data['team_batting'] / max(0.001, 0.270)) * 0.5
                obp_impact = (data['team_obp'] / max(0.001, 0.340)) * 0.5
                
                # 基礎得分
                base_score = (data['time_avg'] * 0.5 + 
                            data['base_avg'] * 0.3 + 
                            data['allow'] * 0.2)
                
                return max(0.1, base_score * league_factor * 
                       (0.8 + 0.2 * data['over_rate']) * 
                       (1.0 / max(0.1, pitcher_factor)) * 
                       (0.6 + batting_impact + obp_impact))
            
            home_score = weighted_score(home, away['pitcher'])
            away_score = weighted_score(away, home['pitcher'])
        else:
            # 基準模型計算（僅基礎數據）
            home_score = (home['time_avg'] * 0.6 + home['base_avg'] * 0.4) * league_factor
            away_score = (away['time_avg'] * 0.6 + away['base_avg'] * 0.4) * league_factor
            
        return home_score, away_score
    except Exception as e:
        print(f"得分計算錯誤: {str(e)}")
        return None, None

def run_probability_models(home_score, away_score, target, league):
    """執行三種概率模型模擬"""
    try:
        std_dev = NPB_STD_DEV if league == 'NPB' else KBO_STD_DEV
        nb_r = NPB_NB_R if league == 'NPB' else KBO_NB_R
        
        # 蒙特卡洛模擬
        home_std = std_dev * (home_score / max(0.1, (home_score + away_score)))
        away_std = std_dev * (away_score / max(0.1, (home_score + away_score)))
        mc_scores = np.random.normal(home_score, home_std, NUM_SIMULATIONS) + \
                   np.random.normal(away_score, away_std, NUM_SIMULATIONS)
        mc_prob = np.mean(mc_scores > target) * 100
        
        # 負二項分布模擬
        def generate_nb(mean):
            p = nb_r / max(0.1, (nb_r + mean))
            return nbinom.rvs(nb_r, p, size=NUM_SIMULATIONS)
        nb_prob = np.mean((generate_nb(home_score) + generate_nb(away_score)) > target) * 100
        
        # 泊松分布模擬
        poisson_prob = np.mean((np.random.poisson(home_score, NUM_SIMULATIONS) + 
                              np.random.poisson(away_score, NUM_SIMULATIONS)) > target) * 100
        
        # 聯盟加權概率
        if league == 'NPB':
            final_prob = mc_prob * 0.5 + nb_prob * 0.35 + poisson_prob * 0.15
        else:
            final_prob = mc_prob * 0.6 + nb_prob * 0.25 + poisson_prob * 0.15
            
        return {
            'mc_prob': mc_prob,
            'nb_prob': nb_prob,
            'poisson_prob': poisson_prob,
            'final_prob': final_prob
        }
    except Exception as e:
        print(f"模擬過程錯誤: {str(e)}")
        return {
            'mc_prob': 50.0,
            'nb_prob': 50.0,
            'poisson_prob': 50.0,
            'final_prob': 50.0
        }

def get_star_recommendation(prob):
    """統一推薦邏輯：>50%推薦大分，<50%推薦小分"""
    try:
        if prob >= 50:  # 大分推薦
            strength = prob - 50  # 計算超出50%的幅度 (0-50)
            if strength >= 20: return 5.0, "🔥 極強力推薦大分 (概率:%.1f%%)" % prob
            elif strength >= 10: return 4.0, "★ 強力推薦大分 (概率:%.1f%%)" % prob
            elif strength >= 5: return 3.5, "↑ 看好大分 (概率:%.1f%%)" % prob
            else: return 3.0, "→ 稍推大分 (概率:%.1f%%)" % prob
        else:  # 小分推薦
            strength = 50 - prob  # 計算低於50%的幅度 (0-50)
            if strength >= 20: return 5.0, "🔥 極強力推薦小分 (概率:%.1f%%)" % (100-prob)
            elif strength >= 10: return 4.0, "★ 強力推薦小分 (概率:%.1f%%)" % (100-prob)
            elif strength >= 5: return 3.5, "↑ 看好小分 (概率:%.1f%%)" % (100-prob)
            else: return 3.0, "→ 稍推小分 (概率:%.1f%%)" % (100-prob)
    except:
        return 0.0, "無法生成推薦"

def display_results(model_name, home_score, away_score, target, results):
    """顯示分析結果"""
    print(f"\n【{model_name}模型】")
    print(f"預期得分: 主隊 {home_score:.2f} - 客隊 {away_score:.2f}")
    print(f"預期總分: {home_score + away_score:.2f} (盤口: {target})")
    print("三種概率模型:")
    print(f"  蒙特卡洛: {results['mc_prob']:.1f}%")
    print(f"  負二項分布: {results['nb_prob']:.1f}%")
    print(f"  泊松分布: {results['poisson_prob']:.1f}%")
    print(f"綜合概率: {results['final_prob']:.1f}%")
    
    rating, comment = get_star_recommendation(results['final_prob'])
    print(f"推薦: {'★' * int(rating)}{'½' * (1 if rating % 1 >= 0.5 else 0)}{'☆' * (5 - int(np.ceil(rating)))} {comment}")

def main():
    print("=== 日韓職棒深度分析系統 ===")
    print(f"★ 模擬次數: {NUM_SIMULATIONS:,}次 ★")
    print("★ 功能說明 ★")
    print("- 完整模型: 包含投手數據和團隊打擊數據")
    print("- 基準模型: 僅使用基本得分數據")
    
    while True:
        try:
            # 輸入基本資料
            league = 'NPB' if input("\n選擇聯盟 (1.NPB日棒/2.KBO韓棒): ").strip() == "1" else 'KBO'
            time_period = "日間" if input("比賽時段 (1.日間/2.晚間): ").strip() == "1" else "晚間"
            home_name = input("\n主隊名稱: ").strip()
            away_name = input("客隊名稱: ").strip()
            target = safe_input("盤口總分: ", min_val=0)
            
            # 輸入球隊資料
            home = input_team_data(home_name, time_period, True)
            away = input_team_data(away_name, time_period, False)
            
            # 完整模型分析
            full_home, full_away = calculate_scores(home, away, league, True)
            full_results = run_probability_models(full_home, full_away, target, league)
            
            # 基準模型分析
            base_home, base_away = calculate_scores(home, away, league, False)
            base_results = run_probability_models(base_home, base_away, target, league)
            
            # 顯示結果
            print(f"\n=== {league} 分析結果 ===")
            print(f"對戰組合: {home_name} vs {away_name}")
            print(f"盤口總分: {target}")
            
            display_results("完整（含投手+團隊）", full_home, full_away, target, full_results)
            display_results("基準（僅基礎數據）", base_home, base_away, target, base_results)
            
            # 影響分析
            diff = full_results['final_prob'] - base_results['final_prob']
            print(f"\n★ 投手+團隊影響: {'增加' if diff >= 0 else '減少'} {abs(diff):.1f}% 大分概率")
            
            if not input("\n是否繼續分析? (y/n): ").lower().startswith('y'):
                print("感謝使用分析系統！")
                break
                
        except KeyboardInterrupt:
            print("\n程式已終止")
            break
        except Exception as e:
            print(f"\n發生錯誤: {str(e)}")
            if not input("是否繼續? (y/n): ").lower().startswith('y'):
                break

if __name__ == "__main__":
    main()