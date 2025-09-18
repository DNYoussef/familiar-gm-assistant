"""
Yahoo Finance Data Source
Free and reliable data source for stock market information
"""

import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import asyncio
from lib.shared.utilities import get_logger
logger = get_logger(__name__)
        self.session = None
        self.crumb = None
        self.cookies = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with proper headers"""
        if self.session is None or self.session.closed:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)

            # Get crumb for authenticated requests
            await self._get_crumb()

        return self.session

    async def _get_crumb(self):
        """Get Yahoo Finance crumb for authenticated requests"""
        try:
            if self.session is None:
                return

            # Get initial cookies
            async with self.session.get("https://finance.yahoo.com/") as response:
                self.cookies = response.cookies

            # Get crumb
            async with self.session.get("https://query1.finance.yahoo.com/v1/test/getcrumb") as response:
                if response.status == 200:
                    self.crumb = await response.text()
                    self.crumb = self.crumb.strip()

        except Exception as e:
            self.logger.warning(f"Failed to get Yahoo crumb: {e}")

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical stock data from Yahoo Finance

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date
            end_date: End date
            timeframe: Timeframe (1Min, 5Min, 15Min, 1H, 1D)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            session = await self._get_session()

            # Convert timestamps
            period1 = int(start_date.timestamp())
            period2 = int(end_date.timestamp())

            # Convert timeframe to Yahoo format
            interval = self._convert_timeframe(timeframe)

            # Build request URL
            url = f"{self.config.base_url}/v7/finance/download/{symbol}"
            params = {
                "period1": period1,
                "period2": period2,
                "interval": interval,
                "events": "history",
                "includeAdjustedClose": "true"
            }

            if self.crumb:
                params["crumb"] = self.crumb

            retry_count = 0
            while retry_count < self.config.retry_attempts:
                try:
                    async with session.get(url, params=params, cookies=self.cookies) as response:
                        if response.status == 200:
                            content = await response.text()

                            # Parse CSV data
                            from io import StringIO
                            df = pd.read_csv(StringIO(content))

                            if not df.empty and len(df) > 0:
                                # Clean column names
                                df.columns = df.columns.str.lower()
                                df.columns = df.columns.str.replace(' ', '_')

                                # Convert date column
                                if 'date' in df.columns:
                                    df['timestamp'] = pd.to_datetime(df['date'])
                                    df.set_index('timestamp', inplace=True)
                                    df.drop('date', axis=1, inplace=True)

                                # Rename columns to standard format
                                column_mapping = {
                                    'adj_close': 'adjusted_close'
                                }
                                df.rename(columns=column_mapping, inplace=True)

                                # Remove rows with null values in essential columns
                                essential_columns = ['open', 'high', 'low', 'close', 'volume']
                                df = df.dropna(subset=[col for col in essential_columns if col in df.columns])

                                # Convert numeric columns
                                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'adjusted_close']
                                for col in numeric_columns:
                                    if col in df.columns:
                                        df[col] = pd.to_numeric(df[col], errors='coerce')

                                return df

                            else:
                                self.logger.warning(f"Empty dataset returned for {symbol}")
                                break

                        elif response.status == 404:
                            self.logger.warning(f"Symbol {symbol} not found")
                            break

                        else:
                            error_text = await response.text()
                            self.logger.error(f"Yahoo API error for {symbol}: {response.status} - {error_text}")

                            if response.status == 429:  # Rate limited
                                await asyncio.sleep(60)  # Wait 1 minute

                            retry_count += 1
                            await asyncio.sleep(2 ** retry_count)

                except Exception as e:
                    self.logger.error(f"Request failed for {symbol}: {e}")
                    retry_count += 1
                    await asyncio.sleep(2 ** retry_count)

            return None

        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None

    async def get_real_time_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote for a symbol"""
        try:
            session = await self._get_session()
            url = f"{self.config.base_url}/v8/finance/chart/{symbol}"

            params = {
                "interval": "1m",
                "range": "1d",
                "includeAdjustedClose": "true"
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if "chart" in data and "result" in data["chart"]:
                        result = data["chart"]["result"][0]
                        meta = result["meta"]

                        quote = {
                            "symbol": symbol,
                            "price": float(meta.get("regularMarketPrice", 0)),
                            "change": float(meta.get("regularMarketDayRange", "0-0").split("-")[0]),
                            "change_percent": float(meta.get("regularMarketChangePercent", 0)),
                            "volume": int(meta.get("regularMarketVolume", 0)),
                            "market_cap": meta.get("marketCap"),
                            "timestamp": pd.to_datetime(meta.get("regularMarketTime"), unit="s")
                        }

                        return quote

                else:
                    error_text = await response.text()
                    self.logger.error(f"Quote request failed for {symbol}: {error_text}")

            return None

        except Exception as e:
            self.logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    async def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company information"""
        try:
            session = await self._get_session()
            url = f"{self.config.base_url}/v10/finance/quoteSummary/{symbol}"

            params = {
                "modules": "summaryProfile,financialData,defaultKeyStatistics"
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if "quoteSummary" in data and "result" in data["quoteSummary"]:
                        result = data["quoteSummary"]["result"][0]

                        info = {
                            "symbol": symbol,
                            "company_name": "",
                            "sector": "",
                            "industry": "",
                            "market_cap": None,
                            "pe_ratio": None,
                            "dividend_yield": None
                        }

                        # Summary profile
                        if "summaryProfile" in result:
                            profile = result["summaryProfile"]
                            info["company_name"] = profile.get("longName", "")
                            info["sector"] = profile.get("sector", "")
                            info["industry"] = profile.get("industry", "")

                        # Financial data
                        if "financialData" in result:
                            financial = result["financialData"]
                            info["market_cap"] = financial.get("marketCap", {}).get("raw")

                        # Key statistics
                        if "defaultKeyStatistics" in result:
                            stats = result["defaultKeyStatistics"]
                            pe_ratio = stats.get("trailingPE", {})
                            if isinstance(pe_ratio, dict) and "raw" in pe_ratio:
                                info["pe_ratio"] = pe_ratio["raw"]

                        return info

                else:
                    error_text = await response.text()
                    self.logger.error(f"Company info request failed for {symbol}: {error_text}")

            return None

        except Exception as e:
            self.logger.error(f"Failed to get company info for {symbol}: {e}")
            return None

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to Yahoo format"""
        mapping = {
            "1Min": "1m",
            "2Min": "2m",
            "5Min": "5m",
            "15Min": "15m",
            "30Min": "30m",
            "1H": "1h",
            "1D": "1d",
            "1W": "1wk",
            "1M": "1mo"
        }
        return mapping.get(timeframe, "1d")

    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.session.close())
                else:
                    loop.run_until_complete(self.session.close())
            except Exception:
                pass