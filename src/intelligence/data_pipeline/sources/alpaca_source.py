"""
Alpaca Data Source
Integration with Alpaca Markets API for stock and crypto data
"""

import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import asyncio
from lib.shared.utilities import get_logger
logger = get_logger(__name__)
        self.session = None
        self.base_headers = {
            "APCA-API-KEY-ID": config.api_key,
            "APCA-API-SECRET-KEY": self._get_secret_key(),
            "Content-Type": "application/json"
        }

    def _get_secret_key(self) -> str:
        """Get Alpaca secret key from environment"""
        import os
        return os.getenv("ALPACA_SECRET_KEY", "")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                headers=self.base_headers,
                timeout=timeout
            )
        return self.session

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical stock data from Alpaca

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

            # Convert timeframe to Alpaca format
            alpaca_timeframe = self._convert_timeframe(timeframe)

            # Build request URL
            url = f"{self.config.base_url}/v2/stocks/{symbol}/bars"
            params = {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "timeframe": alpaca_timeframe,
                "limit": 10000,  # Max records per request
                "page_token": None
            }

            all_data = []
            retry_count = 0

            while retry_count < self.config.retry_attempts:
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()

                            if "bars" in data and data["bars"]:
                                bars = data["bars"]
                                df_data = []

                                for bar in bars:
                                    df_data.append({
                                        "timestamp": pd.to_datetime(bar["t"]),
                                        "open": float(bar["o"]),
                                        "high": float(bar["h"]),
                                        "low": float(bar["l"]),
                                        "close": float(bar["c"]),
                                        "volume": int(bar["v"])
                                    })

                                if df_data:
                                    df = pd.DataFrame(df_data)
                                    df.set_index("timestamp", inplace=True)
                                    all_data.append(df)

                                # Check for pagination
                                if "next_page_token" in data and data["next_page_token"]:
                                    params["page_token"] = data["next_page_token"]
                                    continue
                                else:
                                    break

                            else:
                                self.logger.warning(f"No data returned for {symbol}")
                                break

                        elif response.status == 429:  # Rate limited
                            retry_after = int(response.headers.get("Retry-After", 60))
                            self.logger.warning(f"Rate limited for {symbol}, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            retry_count += 1

                        else:
                            error_text = await response.text()
                            self.logger.error(f"Alpaca API error for {symbol}: {response.status} - {error_text}")
                            retry_count += 1
                            await asyncio.sleep(2 ** retry_count)  # Exponential backoff

                except Exception as e:
                    self.logger.error(f"Request failed for {symbol}: {e}")
                    retry_count += 1
                    await asyncio.sleep(2 ** retry_count)

            # Combine all data
            if all_data:
                combined_df = pd.concat(all_data)
                combined_df = combined_df.sort_index()
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                return combined_df

            return None

        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None

    async def get_real_time_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote for a symbol"""
        try:
            session = await self._get_session()
            url = f"{self.config.base_url}/v2/stocks/{symbol}/quotes/latest"

            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if "quote" in data:
                        quote = data["quote"]
                        return {
                            "symbol": symbol,
                            "bid": float(quote["bp"]),
                            "ask": float(quote["ap"]),
                            "bid_size": int(quote["bs"]),
                            "ask_size": int(quote["as"]),
                            "timestamp": pd.to_datetime(quote["t"])
                        }
                else:
                    error_text = await response.text()
                    self.logger.error(f"Quote request failed for {symbol}: {error_text}")

            return None

        except Exception as e:
            self.logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    async def get_trades(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000
    ) -> Optional[pd.DataFrame]:
        """Get trade data for a symbol"""
        try:
            session = await self._get_session()
            url = f"{self.config.base_url}/v2/stocks/{symbol}/trades"

            params = {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "limit": limit
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if "trades" in data and data["trades"]:
                        trades = data["trades"]
                        df_data = []

                        for trade in trades:
                            df_data.append({
                                "timestamp": pd.to_datetime(trade["t"]),
                                "price": float(trade["p"]),
                                "size": int(trade["s"]),
                                "conditions": trade.get("c", [])
                            })

                        if df_data:
                            df = pd.DataFrame(df_data)
                            df.set_index("timestamp", inplace=True)
                            return df

                else:
                    error_text = await response.text()
                    self.logger.error(f"Trades request failed for {symbol}: {error_text}")

            return None

        except Exception as e:
            self.logger.error(f"Failed to get trades for {symbol}: {e}")
            return None

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to Alpaca format"""
        mapping = {
            "1Min": "1Min",
            "5Min": "5Min",
            "15Min": "15Min",
            "30Min": "30Min",
            "1H": "1Hour",
            "1D": "1Day"
        }
        return mapping.get(timeframe, "1Day")

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