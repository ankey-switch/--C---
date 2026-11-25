# -*- coding: utf-8 -*-
"""回测引擎的离线单元测试：不依赖网络，全部使用合成行情。

运行方式（在 quant/ 目录下）：
    python -m unittest test_engine -v
"""

import unittest

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine


def make_df(opens, closes, signals, start="2024-01-01"):
    """用 (开盘价, 收盘价, 信号) 三列等长序列构造引擎可用的行情表。"""
    n = len(opens)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n),
        "open": opens,
        "high": [o * 1.02 for o in opens],
        "low": [o * 0.98 for o in opens],
        "close": closes,
        "volume": [10_000] * n,
        "signal": signals,
    })


class TestExecutionTiming(unittest.TestCase):
    """T+1 执行时机与成交约束。"""

    def test_buy_on_signal_day_open(self):
        # 信号列已是策略平移后的执行信号：signal=1 当日按开盘价买入
        df = make_df([10.0, 10.0, 11.0], [10.0, 10.0, 11.0], [0, 1, 0])
        r = BacktestEngine().run(df)
        self.assertEqual(r.loc[0, "shares"], 0, "信号日之前必须空仓")
        self.assertEqual(r.loc[1, "trade"], "BUY")
        self.assertGreater(r.loc[1, "shares"], 0)
        # 成交价 = 开盘价 * (1 + 滑点)
        self.assertAlmostEqual(r.loc[1, "buy_price"], 10.0 * 1.001, places=9)

    def test_no_reentry_while_holding(self):
        # 持仓期间重复出现买入信号，不应加仓
        df = make_df([10.0] * 4, [10.0] * 4, [0, 1, 1, 1])
        r = BacktestEngine().run(df)
        self.assertEqual((r["trade"] == "BUY").sum(), 1)
        self.assertTrue((r["shares"].iloc[1:] == r.loc[1, "shares"]).all(),
                        "持仓期间股数不应变化")

    def test_sell_only_when_holding(self):
        # 空仓时出现卖出信号不应产生成交
        df = make_df([10.0] * 3, [10.0] * 3, [0, -1, -1])
        r = BacktestEngine().run(df)
        self.assertEqual((r["trade"] == "SELL").sum(), 0)
        self.assertEqual(r["shares"].sum(), 0)

    def test_suspended_day_skipped(self):
        # 开盘价缺失（停牌）当日不交易，次日信号仍可执行
        df = make_df([10.0, np.nan, 10.0], [10.0, 10.0, 10.0], [0, 1, 1])
        r = BacktestEngine(initial_capital=100_000).run(df)
        self.assertEqual(r.loc[1, "trade"] if r.loc[1, "trade"] == r.loc[1, "trade"] else None, None)
        self.assertEqual(r.loc[2, "trade"], "BUY")

    def test_cannot_afford_one_lot(self):
        # 资金不足一手时放弃开仓，保持空仓
        df = make_df([5_000.0] * 3, [5_000.0] * 3, [0, 1, 0])
        r = BacktestEngine(initial_capital=30_000).run(df)  # 一手约 5 万买不起
        self.assertEqual(r["shares"].sum(), 0)
        self.assertTrue(r["trade"].isna().all())


class TestCashAccounting(unittest.TestCase):
    """现金与费用核算。"""

    def test_buy_cash_conservation(self):
        cash0 = 200_000.0
        df = make_df([10.0, 10.0], [10.0, 10.0], [0, 1])
        eng = BacktestEngine(initial_capital=cash0)
        r = eng.run(df)
        day = r.iloc[1]
        price = 10.0 * 1.001
        expected_cost = day["shares"] * price
        expected_fee = max(expected_cost * 0.0003, 5.0)
        self.assertAlmostEqual(day["cash"], cash0 - expected_cost - expected_fee, places=6)
        self.assertGreaterEqual(day["cash"], 0, "现金不允许为负")

    def test_sell_round_trip_with_stamp(self):
        cash0 = 1_000_000.0
        df = make_df([10.0, 10.0, 10.0, 10.0], [10.0] * 4, [0, 1, 0, -1])
        eng = BacktestEngine(initial_capital=cash0)
        r = eng.run(df)
        t = eng.get_trades()
        self.assertEqual(len(t), 2)
        buy, sell = t.iloc[0], t.iloc[1]
        # 卖出印花税 = 卖出金额 * 万五
        self.assertAlmostEqual(sell["stamp_tax"], sell["amount"] * 0.0005, places=9)
        # 佣金 = 金额 * 万三 与 5 元取大
        self.assertAlmostEqual(buy["commission"], max(buy["amount"] * 0.0003, 5.0), places=9)
        # 期末现金与逐笔记录一致
        self.assertAlmostEqual(t.iloc[-1]["cash_after"], r.iloc[-1]["cash"], places=6)

    def test_min_commission_applies(self):
        # 极小资金只买得起 1 手，佣金应取最低 5 元
        df = make_df([10.0, 10.0], [10.0, 10.0], [0, 1])
        eng = BacktestEngine(initial_capital=1_500)
        eng.run(df)
        t = eng.get_trades()
        self.assertEqual(t.iloc[0]["commission"], 5.0)

    def test_trades_csv_fields(self):
        df = make_df([10.0, 10.0, 10.0], [10.0] * 3, [0, 1, -1])
        eng = BacktestEngine()
        eng.run(df)
        t = eng.get_trades()
        for col in ("date", "action", "price", "shares", "amount",
                    "commission", "stamp_tax", "cash_after"):
            self.assertIn(col, t.columns)


class TestInputValidation(unittest.TestCase):
    """输入校验与报错信息。"""

    def test_empty_df_raises(self):
        empty = pd.DataFrame(columns=["date", "open", "high", "low",
                                      "close", "volume", "signal"])
        with self.assertRaises(ValueError):
            BacktestEngine().run(empty)

    def test_missing_column_message(self):
        bad = pd.DataFrame({"date": [1], "close": [1.0], "signal": [0]})
        with self.assertRaises(ValueError) as ctx:
            BacktestEngine().run(bad)
        self.assertIn("缺少必需列", str(ctx.exception))

    def test_unsorted_dates_raise(self):
        df = make_df([10.0, 10.0], [10.0, 10.0], [0, 0])
        shuffled = df.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError):
            BacktestEngine().run(shuffled)

    def test_non_dataframe_raises(self):
        with self.assertRaises(TypeError):
            BacktestEngine().run([[10.0, 10.0, 0]])

    def test_nonpositive_close_raises(self):
        df = make_df([10.0, 10.0], [10.0, 0.0], [0, 0])
        with self.assertRaises(ValueError):
            BacktestEngine().run(df)


class TestMetrics(unittest.TestCase):
    """绩效指标计算与边界情况。"""

    def test_metrics_on_empty_result(self):
        m = BacktestEngine.compute_metrics(pd.DataFrame({"nav": []}))
        self.assertIsNone(m["cumulative_return"])
        self.assertEqual(m["n_trades"], 0)

    def test_metrics_on_benchmark_nav(self):
        # 基准净值表只有 nav 列，不应报错，且 n_trades=0
        m = BacktestEngine.compute_metrics(pd.DataFrame({"nav": [1.0, 1.1, 1.2]}))
        self.assertAlmostEqual(m["cumulative_return"], 0.2, places=9)
        self.assertEqual(m["n_trades"], 0)

    def test_max_drawdown_known_value(self):
        m = BacktestEngine.compute_metrics(pd.DataFrame({"nav": [1.0, 2.0, 0.5, 1.0]}))
        self.assertAlmostEqual(m["max_drawdown"], -0.75, places=9)

    def test_flat_nav_sharpe_zero(self):
        m = BacktestEngine.compute_metrics(pd.DataFrame({"nav": [1.0] * 10}))
        self.assertEqual(m["sharpe"], 0.0)
        self.assertEqual(m["max_drawdown"], 0.0)

    def test_metrics_consistent_with_result(self):
        df = make_df([10.0, 10.0, 12.0, 12.0], [10.0, 11.0, 12.0, 13.0],
                     [0, 1, 0, 0])
        r = BacktestEngine().run(df)
        m = BacktestEngine.compute_metrics(r)
        self.assertAlmostEqual(m["cumulative_return"], r["nav"].iloc[-1] - 1.0, places=9)
        self.assertAlmostEqual(m["final_total"], r["total"].iloc[-1], places=6)


if __name__ == "__main__":
    unittest.main()
