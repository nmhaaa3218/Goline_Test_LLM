import os
import pandas as pd
from typing import Type, Optional, List
from pydantic import BaseModel, Field
from vnstock import Company
from vnstock import Quote
import pandas_ta_classic as ta
from langchain.tools import BaseTool
from datetime import datetime
from datetime import timezone, timedelta

# Get today's date
def get_today():
    gmt_plus_7 = timezone(timedelta(hours=7))
    return datetime.now(gmt_plus_7).strftime("%Y-%m-%d")

def get_past_date(time_delta: int = 30):
    gmt_plus_7 = timezone(timedelta(hours=7))
    return (datetime.now(gmt_plus_7) - timedelta(days=time_delta)).strftime("%Y-%m-%d")

#============Shareholders============
class ViewShareholdersInput(BaseModel):
    """Input for viewing shareholders of Vietnamese companies."""
    symbols: List[str] = Field(description="List of stock symbols of Vietnamese companies (e.g., ['VIC', 'VCB', 'FPT'])")


class ViewShareholdersTool(BaseTool):
    """Tool to view shareholders information of Vietnamese companies using vnstock."""
    
    name: str = "view_shareholders"
    description: str = "Get shareholders information for Vietnamese stock symbols. Use this when you need to find information about major shareholders, ownership structure of Vietnamese companies."
    args_schema: Type[BaseModel] = ViewShareholdersInput
    return_direct: bool = False
    
    def _run(self, symbols: List[str]) -> dict:
        """Execute the tool to get shareholders information."""
        results = {}
        
        for symbol in symbols:
            try:
                company = Company(symbol=symbol, source='TCBS')
                shareholders_data = company.shareholders()
                
                if shareholders_data is None or shareholders_data.empty:
                    results[symbol] = f"Không tìm thấy thông tin cổ đông cho mã {symbol}"
                else:
                    results[symbol] = shareholders_data.to_json(orient='records', force_ascii=False)
                    
            except Exception as e:
                results[symbol] = f"Lỗi khi lấy thông tin cổ đông cho mã {symbol}: {str(e)}"
        
        return results
    
    async def _arun(self, symbols: List[str]) -> dict:
        """Async version of the tool."""
        return self._run(symbols)

# Create an instance of the tool and invoke it
# result = ViewShareholdersTool().invoke({'symbols': ['VCB', 'VIC']})
# print(result)

#============Management============
class ViewManagementInput(BaseModel):
    """Input for viewing management of Vietnamese companies."""
    symbols: List[str] = Field(description="List of stock symbols of Vietnamese companies (e.g., ['VIC', 'VCB', 'FPT'])")


class ViewManagementTool(BaseTool):
    """Tool to view management information of Vietnamese companies using vnstock."""
    
    name: str = "view_management"
    description: str = "Get management information for Vietnamese stock symbols. Use this when you need to find information about company officers, management team, executives of Vietnamese companies."
    args_schema: Type[BaseModel] = ViewManagementInput
    return_direct: bool = False
    
    def _run(self, symbols: List[str]) -> dict:
        """Execute the tool to get management information."""
        results = {}
        
        for symbol in symbols:
            try:
                company = Company(symbol=symbol, source='TCBS')
                management_data = company.officers(filter_by='working')
                
                if management_data is None or management_data.empty:
                    results[symbol] = f"Không tìm thấy thông tin ban lãnh đạo cho mã {symbol}"
                else:
                    results[symbol] = management_data.to_json(orient='records', force_ascii=False)
                    
            except Exception as e:
                results[symbol] = f"Lỗi khi lấy thông tin ban lãnh đạo cho mã {symbol}: {str(e)}"
        
        return results
    
    async def _arun(self, symbols: List[str]) -> dict:
        """Async version of the tool."""
        return self._run(symbols)

# Create an instance of the tool and invoke it
# result = ViewManagementTool().invoke({'symbols': ['TCB', 'VIC']})
# print(result)

# ============Subsidaries============
class ViewSubsidiariesInput(BaseModel):
    """Input for viewing subsidiaries of Vietnamese companies."""
    symbols: List[str] = Field(description="List of stock symbols of Vietnamese companies (e.g., ['VIC', 'VCB', 'FPT'])")


class ViewSubsidiariesTool(BaseTool):
    """Tool to view subsidiaries information of Vietnamese companies using vnstock."""
    
    name: str = "view_subsidiaries"
    description: str = "Get subsidiaries information for Vietnamese stock symbols. Use this when you need to find information about subsidiary companies, affiliated companies of Vietnamese companies."
    args_schema: Type[BaseModel] = ViewSubsidiariesInput
    return_direct: bool = False
    
    def _run(self, symbols: List[str]) -> dict:
        """Execute the tool to get subsidiaries information."""
        results = {}
        
        for symbol in symbols:
            try:
                company = Company(symbol=symbol, source='TCBS')
                subsidiaries_data = company.subsidiaries()
                
                if subsidiaries_data is None or subsidiaries_data.empty:
                    results[symbol] = f"Không tìm thấy thông tin công ty con cho mã {symbol}"
                else:
                    results[symbol] = subsidiaries_data.to_json(orient='records', force_ascii=False)
                    
            except Exception as e:
                results[symbol] = f"Lỗi khi lấy thông tin công ty con cho mã {symbol}: {str(e)}"
        
        return results
    
    async def _arun(self, symbols: List[str]) -> dict:
        """Async version of the tool."""
        return self._run(symbols)

# Create an instance of the tool and invoke it
# result = ViewSubsidiariesTool().invoke({'symbols': ['TCB', 'VIC']})
# print(result)

# ============OHLCV============
class ViewOHLCVInput(BaseModel):
    """Input for viewing OHLCV data of Vietnamese companies."""
    symbols: List[str] = Field(description="List of stock symbols of Vietnamese companies (e.g., ['VIC', 'VCB', 'FPT'])")
    start: str = Field(description="Start date for OHLCV data in YYYY-mm-dd format", default_factory=get_past_date)
    end: str = Field(description="End date for OHLCV data in YYYY-mm-dd format", default_factory=get_today)
    interval: str = Field(description="Timeframe for OHLCV data. Available options: '1m' (1 minute), '5m' (5 minutes), '15m' (15 minutes), '30m' (30 minutes), '1H' (1 hour), '1D' (1 day), '1W' (1 week), '1M' (1 month)", default="1D")
    columns: Optional[List[str]] = Field(description="List of columns to return. Available options: ['time', 'open', 'high', 'low', 'close', 'volume']. If not specified, all columns will be returned.", default=None)


class ViewOHLCVTool(BaseTool):
    """Tool to view OHLCV (Open, High, Low, Close, Volume) data of Vietnamese companies using vnstock."""
    
    name: str = "view_ohlcv"
    description: str = "Get OHLCV (Open, High, Low, Close, Volume) data for Vietnamese stock symbols with specified timeframe and date range. You can select specific columns like 'open', 'close', 'volume', etc. Use this when you need historical price data, trading volume, or technical analysis data."
    args_schema: Type[BaseModel] = ViewOHLCVInput
    return_direct: bool = False
    
    def _run(self, symbols: List[str], start: Optional[str] = None, end: Optional[str] = None, interval: str = "1D", columns: Optional[list] = None) -> dict:
        """Execute the tool to get OHLCV data."""
        if start is None:
            start = get_past_date()  # Uses your 30-day default
        if end is None:
            end = get_today()
        if interval is None:
            interval = "1D"  # Your default interval
            
        results = {}
        
        # Validate interval
        valid_intervals = ['1m', '5m', '15m', '30m', '1H', '1D', '1W', '1M']
        if interval not in valid_intervals:
            return f"Khung thời gian không hợp lệ. Các khung thời gian có sẵn: {', '.join(valid_intervals)}"
        
        # Validate columns if provided
        available_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        if columns:
            invalid_columns = [col for col in columns if col not in available_columns]
            if invalid_columns:
                return f"Cột không hợp lệ: {', '.join(invalid_columns)}. Các cột có sẵn: {', '.join(available_columns)}"
        
        for symbol in symbols:
            try:
                quote = Quote(symbol=symbol, source='VCI')
                ohlcv_data = quote.history(start=start, end=end, interval=interval)
                
                if ohlcv_data is None or ohlcv_data.empty:
                    results[symbol] = f"Không tìm thấy dữ liệu OHLCV cho mã {symbol} từ {start} đến {end} với khung thời gian {interval}"
                    continue
                
                # Filter columns if specified
                if columns:
                    # Ensure 'time' is always included if other columns are selected
                    if 'time' not in columns and len(columns) > 0:
                        columns = ['time'] + columns
                    
                    # Filter the dataframe to only include requested columns
                    available_cols_in_data = [col for col in columns if col in ohlcv_data.columns]
                    if available_cols_in_data:
                        ohlcv_data = ohlcv_data[available_cols_in_data]
                    else:
                        results[symbol] = f"Không tìm thấy cột nào trong dữ liệu cho mã {symbol}"
                        continue
                
                results[symbol] = ohlcv_data.to_json(orient='records', force_ascii=False)
                
            except Exception as e:
                results[symbol] = f"Lỗi khi lấy dữ liệu OHLCV cho mã {symbol}: {str(e)}"
        
        return results
    
    async def _arun(self, symbols: List[str], start: Optional[str] = None, end: Optional[str] = None, interval: str = "1D", columns: Optional[list] = None) -> dict:
        """Async version of the tool."""
        return self._run(symbols, start, end, interval, columns)


# result = ViewOHLCVTool().invoke({"symbols": ["TCB"], "start": "2025-11-04", "end": "2025-11-13", "interval": "1D", "columns": ["open", "close", "volume"]})
# print(result)

#============Volume Profile============
class CalculateTotalVolumeInput(BaseModel):
    """Input schema for Calculate Total Volume tool."""
    symbols: List[str] = Field(description="List of stock symbols (e.g., ['VCB', 'VIC', 'FPT'])")
    start: str = Field(default_factory=get_past_date, description="Ngày bắt đầu (YYYY-MM-DD)")
    end: str = Field(default_factory=get_today, description="Ngày kết thúc (YYYY-MM-DD)")
    interval: str = Field(default="1D", description="Khung thời gian (1m, 5m, 15m, 30m, 1H, 1D, 1W, 1M)")

class CalculateTotalVolumeTool(BaseTool):
    """Tool to calculate total volume for Vietnamese stocks in a specified timeframe."""
    
    name: str = "calculate_total_volume"
    description: str = "Tính tổng khối lượng giao dịch của các cổ phiếu Việt Nam trong khoảng thời gian chỉ định. Trả về tổng khối lượng cho từng mã."
    args_schema: Type[BaseModel] = CalculateTotalVolumeInput
    return_direct: bool = False
    
    def _run(self, symbols: List[str], start: Optional[str] = None, end: Optional[str] = None, interval: str = "1D") -> dict:
        """Execute the tool to calculate total volume."""
        if start is None:
            start = get_past_date()  # Uses your 30-day default
        if end is None:
            end = get_today()
        if interval is None:
            interval = "1D"  # Your default interval
            
        results = {}
        
        # Validate interval
        valid_intervals = ['1m', '5m', '15m', '30m', '1H', '1D', '1W', '1M']
        if interval not in valid_intervals:
            return f"Khung thời gian không hợp lệ. Các khung thời gian có sẵn: {', '.join(valid_intervals)}"
        
        for symbol in symbols:
            try:
                quote = Quote(symbol=symbol, source='VCI')
                ohlcv_data = quote.history(start=start, end=end, interval=interval)
                
                if ohlcv_data is None or ohlcv_data.empty:
                    results[symbol] = f"Không tìm thấy dữ liệu OHLCV cho mã {symbol} từ {start} đến {end} với khung thời gian {interval}"
                    continue
                
                # Check if volume column exists
                if 'volume' not in ohlcv_data.columns:
                    results[symbol] = f"Không tìm thấy cột 'volume' trong dữ liệu cho mã {symbol}"
                    continue
                
                # Calculate total volume
                total_volume = ohlcv_data['volume'].sum()
                results[symbol] = int(total_volume)
                
            except Exception as e:
                results[symbol] = f"Lỗi khi tính tổng khối lượng cho mã {symbol}: {str(e)}"
        
        return results
    
    async def _arun(self, symbols: List[str], start: Optional[str] = None, end: Optional[str] = None, interval: str = "1D") -> dict:
        """Async version of the tool."""
        return self._run(symbols, start, end, interval)

# result = CalculateTotalVolumeTool().invoke({"symbols": ["VCB", "VIC"], "start": "2024-10-01", "end": "2024-11-13", "interval": "1D"})
# print(result)


#============Technical Indicators============
class CalculateSMAInput(BaseModel):
    """Input schema for Calculate SMA tool."""
    symbols: List[str] = Field(description="List of stock symbols (e.g., ['VCB', 'VIC', 'FPT'])")
    start: str = Field(default_factory=get_past_date, description="Ngày bắt đầu (YYYY-MM-DD)")
    end: str = Field(default_factory=get_today, description="Ngày kết thúc (YYYY-MM-DD)")
    interval: str = Field(default="1D", description="Khung thời gian (1m, 5m, 15m, 30m, 1H, 1D, 1W, 1M)")
    period: List[int] = Field(default=[20], description="Chu kỳ tính SMA (ví dụ: 20 cho SMA 20 ngày hoặc [9, 20] cho SMA 9 và 20 ngày)")

class CalculateSMATool(BaseTool):
    """Tool to calculate Simple Moving Average (SMA) for Vietnamese stocks."""
    
    name: str = "calculate_sma"
    description: str = "Tính toán đường trung bình động đơn giản (SMA) cho các cổ phiếu Việt Nam. Trả về dữ liệu OHLCV kèm theo cột SMA cho từng mã. Có thể tính nhiều chu kỳ SMA cùng lúc."
    args_schema: Type[BaseModel] = CalculateSMAInput
    return_direct: bool = False
   
    def _run(self, symbols: List[str], start: Optional[str] = None, end: Optional[str] = None, interval: str = "1D", period: List[int] = [20]) -> dict:
        """Execute the tool to calculate SMA."""
        if start is None:
            start = get_past_date()  # Uses your 30-day default
        if end is None:
            end = get_today()
        if interval is None:
            interval = "1D"  # Your default interval
            
        results = {}
        
        # Validate interval
        valid_intervals = ['1m', '5m', '15m', '30m', '1H', '1D', '1W', '1M']
        if interval not in valid_intervals:
            return f"Khung thời gian không hợp lệ. Các khung thời gian có sẵn: {', '.join(valid_intervals)}"
        
        # Convert period to list if it's a single integer
        if isinstance(period, int):
            periods = [period]
        else:
            periods = period
        
        # Validate periods
        for p in periods:
            if p <= 0:
                return f"Chu kỳ SMA phải là số dương. Giá trị nhận được: {p}"
        
        for symbol in symbols:
            try:
                quote = Quote(symbol=symbol, source='VCI')
                ohlcv_data = quote.history(start=start, end=end, interval=interval)
                
                if ohlcv_data is None or ohlcv_data.empty:
                    results[symbol] = f"Không tìm thấy dữ liệu OHLCV cho mã {symbol} từ {start} đến {end} với khung thời gian {interval}"
                    continue
                
                # Calculate SMA using pandas_ta
                if 'close' not in ohlcv_data.columns:
                    results[symbol] = f"Không tìm thấy cột 'close' trong dữ liệu cho mã {symbol}"
                    continue
                
                # Calculate Simple Moving Average for each period using pandas_ta
                for p in periods:
                    ohlcv_data[f'SMA_{p}'] = ta.sma(ohlcv_data['close'], length=p)
                    # Round SMA values to 2 decimal places for better readability
                    ohlcv_data[f'SMA_{p}'] = ohlcv_data[f'SMA_{p}'].round(2)
                
                results[symbol] = ohlcv_data.to_json(orient='records', force_ascii=False)
                
            except Exception as e:
                results[symbol] = f"Lỗi khi tính toán SMA cho mã {symbol}: {str(e)}"
        
        return results
    
    async def _arun(self, symbols: List[str], start: Optional[str] = None, end: Optional[str] = None, interval: str = "1D", period: List[int] = [20]) -> dict:
        """Async version of the tool."""
        return self._run(symbols, start, end, interval, period)

# result = CalculateSMATool().invoke({"symbols": ["VCB", "VIC"], "start": "2024-10-01", "end": "2024-11-13", "interval": "1D", "period": [9, 20]})
# print(result)

#============RSI============
class CalculateRSIInput(BaseModel):
    """Input schema for Calculate RSI tool."""
    symbols: List[str] = Field(description="List of stock symbols (e.g., ['VCB', 'VIC', 'FPT'])")
    start: str = Field(default_factory=get_past_date, description="Ngày bắt đầu (YYYY-MM-DD)")
    end: str = Field(default_factory=get_today, description="Ngày kết thúc (YYYY-MM-DD)")
    interval: str = Field(default="1D", description="Khung thời gian (1m, 5m, 15m, 30m, 1H, 1D, 1W, 1M)")
    period: int = Field(default=14, description="Chu kỳ tính RSI (ví dụ: 14 cho RSI 14 ngày)")

class CalculateRSITool(BaseTool):
    """Tool to calculate Relative Strength Index (RSI) for Vietnamese stocks."""
    
    name: str = "calculate_rsi"
    description: str = "Tính toán chỉ số sức mạnh tương đối (RSI) cho các cổ phiếu Việt Nam. Trả về dữ liệu OHLCV kèm theo cột RSI cho từng mã."
    args_schema: Type[BaseModel] = CalculateRSIInput
    return_direct: bool = False
    
    def _run(self, symbols: List[str], start: Optional[str] = None, end: Optional[str] = None, interval: str = "1D", period: int = 14) -> dict:
        """Execute the tool to calculate RSI."""
        if start is None:
            start = get_past_date()  # Uses your 30-day default
        if end is None:
            end = get_today()
        if interval is None:
            interval = "1D"  # Your default interval
            
        results = {}
        
        # Validate interval
        valid_intervals = ['1m', '5m', '15m', '30m', '1H', '1D', '1W', '1M']
        if interval not in valid_intervals:
            return f"Khung thời gian không hợp lệ. Các khung thời gian có sẵn: {', '.join(valid_intervals)}"
        
        # Validate period
        if period <= 0:
            return f"Chu kỳ RSI phải là số dương. Giá trị nhận được: {period}"
        
        for symbol in symbols:
            try:
                quote = Quote(symbol=symbol, source='VCI')
                ohlcv_data = quote.history(start=start, end=end, interval=interval)
                
                if ohlcv_data is None or ohlcv_data.empty:
                    results[symbol] = f"Không tìm thấy dữ liệu OHLCV cho mã {symbol} từ {start} đến {end} với khung thời gian {interval}"
                    continue
                
                # Calculate RSI using pandas_ta
                if 'close' not in ohlcv_data.columns:
                    results[symbol] = f"Không tìm thấy cột 'close' trong dữ liệu cho mã {symbol}"
                    continue
                
                # Calculate RSI using pandas_ta
                ohlcv_data[f'RSI_{period}'] = ta.rsi(ohlcv_data['close'], length=period)
                
                # Round RSI values to 2 decimal places for better readability
                ohlcv_data[f'RSI_{period}'] = ohlcv_data[f'RSI_{period}'].round(2)
                
                results[symbol] = ohlcv_data.to_json(orient='records', force_ascii=False)
                
            except Exception as e:
                results[symbol] = f"Lỗi khi tính toán RSI cho mã {symbol}: {str(e)}"
        
        return results
    
    async def _arun(self, symbols: List[str], start: Optional[str] = None, end: Optional[str] = None, interval: str = "1D", period: int = 14) -> dict:
        """Async version of the tool."""
        return self._run(symbols, start, end, interval, period)

# result = CalculateRSITool().invoke({"symbols": ["VCB", "VIC"], "start": "2024-10-01", "end": "2024-11-13", "interval": "1D", "period": 28})
# print(result)

#============Get all tools============
def get_all_tools():
    return [
        ViewOHLCVTool(),
        ViewManagementTool(),
        ViewShareholdersTool(),
        ViewSubsidiariesTool(),
        CalculateTotalVolumeTool(),
        CalculateSMATool(),
        CalculateRSITool()
    ]