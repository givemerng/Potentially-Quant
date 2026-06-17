import pandas as pd

from src.data.downloader import YahooFinanceDownloader


def test_chunked_batches() -> None:
    logger = type("LoggerStub", (), {"info": lambda *args, **kwargs: None})()
    downloader = YahooFinanceDownloader(logger=logger, max_workers=2)
    batches = list(downloader._chunked(["A", "B", "C", "D", "E"], 2))
    assert batches == [["A", "B"], ["C", "D"], ["E"]]


def test_normalize_download_single_ticker() -> None:
    logger = type(
        "LoggerStub",
        (),
        {"info": lambda *args, **kwargs: None, "warning": lambda *args, **kwargs: None},
    )()
    downloader = YahooFinanceDownloader(logger=logger)
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Adj Close": [1.4],
            "Volume": [100],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    normalized = downloader._normalize_download(raw, ["AAPL"])
    assert list(normalized.index.names) == ["date", "ticker"]
    assert normalized.loc[(pd.Timestamp("2024-01-02"), "AAPL"), "adj_close"] == 1.4
