# -*- coding: utf-8 -*-
"""A 股历史行情数据接入模块。

本模块通过 AKShare 获取个股和指数日频行情，并统一转换为回测系统使用的
DataFrame 格式。它同时提供输入校验、有限次数重试、CSV 本地缓存、字段检查
和清晰的异常信息，避免网络或上游接口变化造成难以定位的报错。

对外接口保持简单：

    stock = fetch_data("000001", "20200101", "20241231")
    benchmark = fetch_benchmark("000300", "20200101", "20241231")
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

import pandas as pd
import requests
import urllib3

try:
    import akshare as ak
except ImportError:  # 允许在未安装依赖时仍能给出清晰提示，也便于离线单元测试
    ak = None


STOCK_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
BENCHMARK_COLUMNS = ["date", "close"]
_DATE_PATTERN = re.compile(r"^\d{8}$")
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
PathLike = Union[os.PathLike, str]


class DataFetchError(RuntimeError):
    """数据下载、字段转换或质量检查失败。"""


def _validate_inputs(symbol: str, start_date: str, end_date: str) -> tuple[str, str, str]:
    """校验并规范化证券代码和日期参数。"""
    symbol = str(symbol).strip()
    start_date = str(start_date).strip()
    end_date = str(end_date).strip()

    if not re.fullmatch(r"\d{6}", symbol):
        raise ValueError(f"证券代码必须是 6 位数字，例如 '000001'；当前值为 {symbol!r}")

    for name, value in (("start_date", start_date), ("end_date", end_date)):
        if not _DATE_PATTERN.fullmatch(value):
            raise ValueError(f"{name} 必须为 YYYYMMDD 格式；当前值为 {value!r}")
        try:
            pd.to_datetime(value, format="%Y%m%d", errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 不是有效日期：{value!r}") from exc

    if start_date > end_date:
        raise ValueError("start_date 不能晚于 end_date")
    return symbol, start_date, end_date


def _require_akshare() -> None:
    if ak is None:
        raise DataFetchError(
            "未安装 AKShare。请先在项目根目录执行：pip install -r requirements.txt"
        )


def _ak_function(*candidate_names: str) -> Callable[..., pd.DataFrame]:
    """取得 AKShare 接口函数，同时兼容改名前后的版本。"""
    _require_akshare()
    for name in candidate_names:
        try:
            function = getattr(ak, name)
        except AttributeError:
            continue
        if callable(function):
            return function
    raise DataFetchError(
        "当前 AKShare 版本不包含可用的历史行情接口："
        + " / ".join(candidate_names)
        + "。请执行 pip install --upgrade akshare"
    )


@contextmanager
def _temporary_ssl_bypass() -> Iterator[None]:
    """临时关闭 requests 证书校验，并在请求结束后恢复原方法。

    该降级仅在用户显式允许且首次请求确认为 SSL 证书错误时使用，不会像永久
    monkey patch 那样污染后续网络请求。
    """
    original = requests.Session.request

    def patched(session, method, url, **kwargs):  # noqa: ANN001
        kwargs["verify"] = False
        return original(session, method, url, **kwargs)

    requests.Session.request = patched
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        yield
    finally:
        requests.Session.request = original


def _call_with_retry(
    fetch_fn: Callable[..., pd.DataFrame],
    *args,
    retries: int = 3,
    retry_wait: float = 1.0,
    allow_insecure_ssl: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """调用数据接口；对临时网络错误做有限次数重试。"""
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            return fetch_fn(*args, **kwargs)
        except requests.exceptions.SSLError as exc:
            last_error = exc
            if allow_insecure_ssl:
                try:
                    with _temporary_ssl_bypass():
                        return fetch_fn(*args, **kwargs)
                except Exception as insecure_exc:  # 上游库可能封装为其他异常类型
                    last_error = insecure_exc
            if attempt < retries:
                time.sleep(retry_wait * attempt)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_wait * attempt)
        except Exception as exc:
            # AKShare 有时会把连接问题包装成通用异常。保留原异常并有限重试，
            # 但绝不无限循环或静默返回空表。
            last_error = exc
            if attempt < retries:
                time.sleep(retry_wait * attempt)

    hint = "；如为校园网证书代理问题，可设置环境变量 QUANT_ALLOW_INSECURE_SSL=1 后重试"
    raise DataFetchError(f"AKShare 请求失败（已尝试 {retries} 次）：{last_error}{hint}") from last_error


def _cache_path(
    kind: str,
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[PathLike],
) -> Path:
    directory = Path(cache_dir) if cache_dir is not None else Path(
        os.getenv("QUANT_CACHE_DIR", str(_DEFAULT_CACHE_DIR))
    )
    return directory / f"{kind}_{symbol}_{start_date}_{end_date}.csv"


def _load_cache(path: Path, expected_columns: list[str]) -> Optional[pd.DataFrame]:
    """读取并检查缓存；缓存损坏时返回 None，由网络重新获取。"""
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, parse_dates=["date"])
        if list(frame.columns) != expected_columns or frame.empty:
            return None
        return frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return None


def _save_cache(frame: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    except OSError as exc:
        # 缓存失败不应让已成功下载的数据不可用，但需要给出可见提示。
        print(f"[数据] 警告：无法写入缓存 {path}：{exc}")


def _transform(
    raw: pd.DataFrame,
    column_map: dict[str, str],
    start_date: str,
    end_date: str,
    data_name: str,
) -> pd.DataFrame:
    """选择、重命名、清洗并检查 AKShare 返回的数据。"""
    if raw is None or raw.empty:
        raise DataFetchError(
            f"{data_name}在 {start_date}~{end_date} 没有返回数据；"
            "请检查代码、日期范围、网络连接以及证券是否已上市"
        )

    missing = [name for name in column_map if name not in raw.columns]
    if missing:
        raise DataFetchError(
            f"AKShare 返回字段发生变化，{data_name}缺少列 {missing}；"
            f"实际列为 {list(raw.columns)}"
        )

    frame = raw[list(column_map)].rename(columns=column_map).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    numeric_columns = [column for column in frame.columns if column != "date"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = (
        frame.dropna(subset=list(frame.columns))
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    start_ts = pd.to_datetime(start_date, format="%Y%m%d")
    end_ts = pd.to_datetime(end_date, format="%Y%m%d")
    frame = frame.loc[frame["date"].between(start_ts, end_ts)].reset_index(drop=True)

    if frame.empty:
        raise DataFetchError(f"{data_name}清洗后没有有效记录，请检查日期范围和上游数据")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise DataFetchError(f"{data_name}日期质量检查失败")
    if "volume" in frame and (frame["volume"] < 0).any():
        raise DataFetchError(f"{data_name}存在负成交量，数据异常")
    if "close" in frame and (frame["close"] <= 0).any():
        raise DataFetchError(f"{data_name}存在非正收盘价，数据异常")

    return frame


def _allow_insecure_ssl_from_env() -> bool:
    return os.getenv("QUANT_ALLOW_INSECURE_SSL", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _index_symbol_with_market(symbol: str) -> str:
    """把 6 位指数代码转换为新浪/腾讯接口使用的市场前缀格式。"""
    return ("sz" if symbol.startswith("399") else "sh") + symbol


def _fetch_benchmark_raw(
    symbol: str,
    start_date: str,
    end_date: str,
    retries: int,
) -> pd.DataFrame:
    """优先使用东方财富指数接口，限流时自动切换腾讯免费接口。"""
    allow_insecure_ssl = _allow_insecure_ssl_from_env()
    primary_error: Optional[Exception] = None
    try:
        return _call_with_retry(
            _ak_function("index_zh_hist_em", "index_zh_a_hist"),
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            retries=retries,
            allow_insecure_ssl=allow_insecure_ssl,
        )
    except DataFetchError as exc:
        primary_error = exc

    # 东方财富接口会临时限流；腾讯接口的数据口径足以用于日频基准比较。
    try:
        return _call_with_retry(
            _ak_function("index_zh_daily_tx"),
            symbol=_index_symbol_with_market(symbol),
            start_date=start_date,
            end_date=end_date,
            retries=retries,
            allow_insecure_ssl=allow_insecure_ssl,
        )
    except DataFetchError as fallback_error:
        raise DataFetchError(
            "基准指数的东方财富与腾讯数据源均不可用。"
            f"东方财富错误：{primary_error}；腾讯错误：{fallback_error}"
        ) from fallback_error


def fetch_data(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    adjust: str = "qfq",
    use_cache: bool = True,
    refresh: bool = False,
    cache_dir: Optional[PathLike] = None,
    retries: int = 3,
) -> pd.DataFrame:
    """获取 A 股个股历史日频行情。

    返回列固定为 ``date, open, high, low, close, volume``，日期升序且无重复。
    ``adjust='qfq'`` 表示前复权，适合计算跨除权日的均线；也可传 ``''`` 获取
    不复权行情。默认优先读取本地缓存，演示时不必重复依赖网络。
    """
    symbol, start_date, end_date = _validate_inputs(symbol, start_date, end_date)
    if adjust not in {"", "qfq", "hfq"}:
        raise ValueError("adjust 只能是 ''、'qfq' 或 'hfq'")
    if retries < 1:
        raise ValueError("retries 必须大于等于 1")

    cache_path = _cache_path(f"stock_{adjust or 'none'}", symbol, start_date, end_date, cache_dir)
    if use_cache and not refresh:
        cached = _load_cache(cache_path, STOCK_COLUMNS)
        if cached is not None:
            cached.attrs.update(source="cache", symbol=symbol, adjust=adjust)
            return cached

    raw = _call_with_retry(
        _ak_function("stock_zh_a_hist_em", "stock_zh_a_hist"),
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
        retries=retries,
        allow_insecure_ssl=_allow_insecure_ssl_from_env(),
    )
    frame = _transform(
        raw,
        {"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
         "收盘": "close", "成交量": "volume"},
        start_date,
        end_date,
        f"股票 {symbol} ",
    )[STOCK_COLUMNS]
    frame.attrs.update(source="akshare", symbol=symbol, adjust=adjust)
    if use_cache:
        _save_cache(frame, cache_path)
    return frame


def fetch_benchmark(
    symbol: str = "000300",
    start_date: str = "20200101",
    end_date: str = "20241231",
    *,
    use_cache: bool = True,
    refresh: bool = False,
    cache_dir: Optional[PathLike] = None,
    retries: int = 3,
) -> pd.DataFrame:
    """获取基准指数历史日频行情，返回 ``date, close`` 两列。"""
    symbol, start_date, end_date = _validate_inputs(symbol, start_date, end_date)
    if retries < 1:
        raise ValueError("retries 必须大于等于 1")

    cache_path = _cache_path("benchmark", symbol, start_date, end_date, cache_dir)
    if use_cache and not refresh:
        cached = _load_cache(cache_path, BENCHMARK_COLUMNS)
        if cached is not None:
            cached.attrs.update(source="cache", symbol=symbol)
            return cached

    raw = _fetch_benchmark_raw(symbol, start_date, end_date, retries)
    benchmark_column_map = (
        {"日期": "date", "收盘": "close"}
        if {"日期", "收盘"}.issubset(raw.columns)
        else {"date": "date", "close": "close"}
    )
    frame = _transform(
        raw,
        benchmark_column_map,
        start_date,
        end_date,
        f"基准指数 {symbol} ",
    )[BENCHMARK_COLUMNS]
    frame.attrs.update(source="akshare", symbol=symbol)
    if use_cache:
        _save_cache(frame, cache_path)
    return frame


__all__ = [
    "BENCHMARK_COLUMNS",
    "DataFetchError",
    "STOCK_COLUMNS",
    "fetch_benchmark",
    "fetch_data",
]
