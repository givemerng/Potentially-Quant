from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from typing import Optional

import pandas as pd


class FamaFrenchDownloader:
    """Client to download daily Fama-French 5-factor returns."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def download_daily_factors(self) -> pd.DataFrame:
        """Download and parse F-F Research Data 5 Factors 2x3 Daily from Tuck website."""
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        self.logger.info("Downloading Fama-French 5-factor daily returns from %s", url)

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                zip_data = response.read()

            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                # Find the main data text file
                txt_files = [name for name in z.namelist() if name.lower().endswith(".txt") or name.lower().endswith(".csv")]
                if not txt_files:
                    raise FileNotFoundError("No text or CSV file found inside Fama-French ZIP file")
                
                with z.open(txt_files[0]) as f:
                    lines = [line.decode("utf-8") for line in f.readlines()]
        except Exception as exc:
            self.logger.exception("Failed to download or extract Fama-French data: %s", exc)
            return pd.DataFrame()

        # Parse Fama-French daily format
        data_lines = []
        header_found = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Header row looks like: " ,Mkt-RF,SMB,HML,RMW,CMA,RF"
            if "Mkt-RF" in stripped and not header_found:
                header_found = True
                continue

            if header_found:
                # Split columns by comma
                parts = [p.strip() for p in stripped.split(",") if p.strip()]
                if len(parts) >= 7:
                    # Check if the first part is an 8-digit date string like "20000103"
                    date_str = parts[0]
                    if date_str.isdigit() and len(date_str) == 8:
                        data_lines.append(parts[:7])
                else:
                    # The end of the daily data section is followed by annual factors or copyright lines
                    if data_lines:
                        break

        if not data_lines:
            self.logger.warning("Parsed Fama-French data is empty.")
            return pd.DataFrame()

        df = pd.DataFrame(data_lines, columns=["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
        try:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            for col in ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]:
                df[col] = pd.to_numeric(df[col]) / 100.0  # Convert percent to decimal
            df = df.sort_values("date").reset_index(drop=True)
            self.logger.info("Successfully parsed %d daily Fama-French factor rows", len(df))
            return df
        except Exception as exc:
            self.logger.exception("Error parsing Fama-French DataFrame columns: %s", exc)
            return pd.DataFrame()
