# -*- coding: utf-8 -*-
"""数据获取模块的离线单元测试：不访问真实网络。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import requests

from data import data_fetcher as fetcher


def stock_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": ["2024-01-03", "2024-01-02", "2024-01-03", "无效日期"],
            "开盘": [11.0, 10.0, 11.1, 99.0],
            "最高": [12.0, 11.0, 12.1, 99.0],
            "最低": [10.5, 9.5, 10.6, 99.0],
            "收盘": [11.5, 10.5, 11.6, 99.0],
            "成交量": [1200, 1000, 1300, 1],
        }
    )


class DataFetcherTests(unittest.TestCase):
    def test_fetch_data_cleans_sorts_and_uses_cache(self):
        api = Mock(return_value=stock_raw())
        fake_ak = SimpleNamespace(stock_zh_a_hist=api, index_zh_a_hist=Mock())

        with tempfile.TemporaryDirectory() as directory, patch.object(fetcher, "ak", fake_ak):
            first = fetcher.fetch_data(
                "000001", "20240101", "20240131", cache_dir=directory
            )
            second = fetcher.fetch_data(
                "000001", "20240101", "20240131", cache_dir=directory
            )

            self.assertEqual(list(first.columns), fetcher.STOCK_COLUMNS)
            self.assertEqual(len(first), 2)
            self.assertTrue(first["date"].is_monotonic_increasing)
            self.assertEqual(float(first.iloc[-1]["close"]), 11.6)
            self.assertEqual(second.attrs["source"], "cache")
            self.assertEqual(api.call_count, 1)
            self.assertEqual(len(list(Path(directory).glob("stock_qfq_*.csv"))), 1)

    def test_fetch_benchmark_has_stable_schema(self):
        raw = pd.DataFrame(
            {"日期": ["2024-01-02", "2024-01-03"], "收盘": [3300, 3320]}
        )
        api = Mock(return_value=raw)
        fake_ak = SimpleNamespace(stock_zh_a_hist=Mock(), index_zh_a_hist=api)

        with tempfile.TemporaryDirectory() as directory, patch.object(fetcher, "ak", fake_ak):
            result = fetcher.fetch_benchmark(
                "000300", "20240101", "20240131", cache_dir=directory
            )

        self.assertEqual(list(result.columns), fetcher.BENCHMARK_COLUMNS)
        self.assertEqual(result["close"].tolist(), [3300, 3320])

    def test_benchmark_switches_source_when_primary_is_unavailable(self):
        primary = Mock(side_effect=requests.exceptions.ConnectionError("东方财富限流"))
        fallback = Mock(
            return_value=pd.DataFrame(
                {"date": ["2024-01-02", "2024-01-03"], "close": [3300, 3320]}
            )
        )
        fake_ak = SimpleNamespace(
            index_zh_hist_em=primary,
            index_zh_daily_tx=fallback,
        )

        with tempfile.TemporaryDirectory() as directory, patch.object(fetcher, "ak", fake_ak):
            result = fetcher.fetch_benchmark(
                "000300", "20240101", "20240131", cache_dir=directory, retries=1
            )

        self.assertEqual(result["close"].tolist(), [3300, 3320])
        fallback.assert_called_once_with(
            symbol="sh000300", start_date="20240101", end_date="20240131"
        )

    def test_invalid_inputs_fail_before_network_request(self):
        with self.assertRaisesRegex(ValueError, "6 位数字"):
            fetcher.fetch_data("1", "20240101", "20240131")
        with self.assertRaisesRegex(ValueError, "有效日期"):
            fetcher.fetch_data("000001", "20240230", "20240301")
        with self.assertRaisesRegex(ValueError, "不能晚于"):
            fetcher.fetch_data("000001", "20240201", "20240101")

    def test_empty_or_changed_upstream_data_has_clear_error(self):
        fake_empty = SimpleNamespace(
            stock_zh_a_hist=Mock(return_value=pd.DataFrame()), index_zh_a_hist=Mock()
        )
        with patch.object(fetcher, "ak", fake_empty):
            with self.assertRaisesRegex(fetcher.DataFetchError, "没有返回数据"):
                fetcher.fetch_data(
                    "000001", "20240101", "20240131", use_cache=False, retries=1
                )

        fake_changed = SimpleNamespace(
            stock_zh_a_hist=Mock(return_value=pd.DataFrame({"日期": ["2024-01-02"]})),
            index_zh_a_hist=Mock(),
        )
        with patch.object(fetcher, "ak", fake_changed):
            with self.assertRaisesRegex(fetcher.DataFetchError, "缺少列"):
                fetcher.fetch_data(
                    "000001", "20240101", "20240131", use_cache=False, retries=1
                )

    def test_transient_network_error_is_retried(self):
        api = Mock(
            side_effect=[
                requests.exceptions.Timeout("第一次超时"),
                requests.exceptions.ConnectionError("第二次断开"),
                pd.DataFrame({"ok": [1]}),
            ]
        )
        result = fetcher._call_with_retry(api, retries=3, retry_wait=0)
        self.assertEqual(api.call_count, 3)
        self.assertEqual(result.iloc[0]["ok"], 1)


if __name__ == "__main__":
    unittest.main()
