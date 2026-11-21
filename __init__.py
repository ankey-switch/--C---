"""数据接入包。"""

from .data_fetcher import DataFetchError, fetch_benchmark, fetch_data

__all__ = ["DataFetchError", "fetch_benchmark", "fetch_data"]
