"""
Polygon.io Data Source
Integration with Polygon.io API for comprehensive market data
"""

import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import asyncio
from lib.shared.utilities import get_logger
logger = get_logger(__name__)
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical stock data from Polygon

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

            # Convert timeframe to Polygon format
            multiplier, timespan = self._convert_timeframe(timeframe)

            # Format dates
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Build request URL
            url = f"{self.config.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}"
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apikey": self.config.api_key
            }

            retry_count = 0
            while retry_count < self.config.retry_attempts:
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()

                            if data.get("status") == "OK" and "results" in data and data["results"]:
                                results = data["results"]
                                df_data = []

                                for result in results:
                                    df_data.append({
                                        "timestamp": pd.to_datetime(result["t"], unit="ms"),
                                        "open": float(result["o"]),
                                        "high": float(result["h"]),
                                        "low": float(result["l"]),
                                        "close": float(result["c"]),
                                        "volume": int(result["v"])
                                    })

                                if df_data:
                                    df = pd.DataFrame(df_data)
                                    df.set_index("timestamp", inplace=True)
                                    return df

                            else:
                                self.logger.warning(f"No data or error status for {symbol}: {data.get('status')}")
                                break

                        elif response.status == 429:  # Rate limited
                            retry_after = int(response.headers.get("Retry-After", 60))
                            self.logger.warning(f"Rate limited for {symbol}, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            retry_count += 1

                        else:
                            error_text = await response.text()
                            self.logger.error(f"Polygon API error for {symbol}: {response.status} - {error_text}")
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
            url = f"{self.config.base_url}/v2/last/nbbo/{symbol}"
            params = {"apikey": self.config.api_key}

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "OK" and "results" in data:
                        result = data["results"]
                        return {
                            "symbol": symbol,
                            "bid": float(result["P"]),
                            "ask": float(result["p"]),
                            "bid_size": int(result["S"]),
                            "ask_size": int(result["s"]),
                            "timestamp": pd.to_datetime(result["t"], unit="ns")
                        }
                else:
                    error_text = await response.text()
                    self.logger.error(f"Quote request failed for {symbol}: {error_text}")

            return None

        except Exception as e:
            self.logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    async def get_options_chain(
        self,
        underlying_symbol: str,
        expiration_date: Optional[datetime] = None,
        contract_type: str = "both"  # "call", "put", "both"
    ) -> Optional[pd.DataFrame]:
        """Get options chain data"""
        try:
            session = await self._get_session()
            url = f"{self.config.base_url}/v3/reference/options/contracts"

            params = {
                "underlying_ticker": underlying_symbol,
                "limit": 1000,
                "apikey": self.config.api_key
            }

            if expiration_date:
                params["expiration_date"] = expiration_date.strftime("%Y-%m-%d")

            if contract_type != "both":
                params["contract_type"] = contract_type

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("status") == "OK" and "results" in data:
                        results = data["results"]
                        df_data = []

                        for result in results:
                            df_data.append({
                                "contract_symbol": result["ticker"],
                                "underlying_symbol": underlying_symbol,
                                "contract_type": result["contract_type"],
                                "strike_price": float(result["strike_price"]),
                                "expiration_date": pd.to_datetime(result["expiration_date"]),
                                "shares_per_contract": int(result.get("shares_per_contract", 100))
                            })

                        if df_data:
                            return pd.DataFrame(df_data)

                else:
                    error_text = await response.text()
                    self.logger.error(f"Options chain request failed: {error_text}")

            return None

        except Exception as e:
            self.logger.error(f"Failed to get options chain for {underlying_symbol}: {e}")
            return None

    async def get_news(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        published_utc_gte: Optional[datetime] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get news articles"""
        try:
            session = await self._get_session()
            url = f"{self.config.base_url}/v2/reference/news"

            params = {
                "limit": limit,
                "apikey": self.config.api_key
            }

            if symbol:
                params["ticker"] = symbol

            if published_utc_gte:
                params["published_utc.gte"] = published_utc_gte.isoformat()

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("status") == "OK" and "results" in data:
                        articles = []
                        for result in data["results"]:
                            articles.append({
                                "id": result["id"],
                                "title": result["title"],
                                "description": result.get("description", ""),
                                "url": result["article_url"],
                                "published_utc": pd.to_datetime(result["published_utc"]),
                                "publisher": result["publisher"]["name"],
                                "tickers": result.get("tickers", [])
                            })

                        return articles

                else:
                    error_text = await response.text()
                    self.logger.error(f"News request failed: {error_text}")

            return None

        except Exception as e:
            self.logger.error(f"Failed to get news: {e}")
            return None

    def _convert_timeframe(self, timeframe: str) -> tuple[int, str]:
        """Convert standard timeframe to Polygon format"""
        mapping = {
            "1Min": (1, "minute"),
            "5Min": (5, "minute"),
            "15Min": (15, "minute"),
            "30Min": (30, "minute"),
            "1H": (1, "hour"),
            "1D": (1, "day")
        }
        return mapping.get(timeframe, (1, "day"))

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