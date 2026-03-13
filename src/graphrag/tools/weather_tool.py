import requests
import logging
import time
from datetime import datetime, timedelta
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 全局预测器引用
_predictor = None

def set_predictor(predictor):
    global _predictor
    _predictor = predictor

def _wmo_mapping(code: int) -> str:
    """WMO天气代码映射为中文描述"""
    mapping = {
        0:'晴',1:'少云',2:'多云(中)',3:'多云(多)',45:'雾',48:'雾(沉积)',
        51:'毛毛雨(小)',53:'毛毛雨(中)',55:'毛毛雨(大)',56:'冻毛毛雨(小)',57:'冻毛毛雨(大)',
        61:'小雨',63:'中雨',65:'大雨',66:'冻雨(小/中)',67:'冻雨(大)',
        71:'小雪',73:'中雪',75:'大雪',77:'雪粒',
        80:'阵雨(小)',81:'阵雨(中)',82:'阵雨(大)',
        85:'阵雪(小/中)',86:'阵雪(大)',
        95:'雷阵雨',96:'雷阵雨伴小冰雹',99:'雷阵雨伴大冰雹'
    }
    return mapping.get(code, f'天气代码 {code}')

def _get_weather_impl(city: str, date: str = "today") -> str:
    """实际的天气查询逻辑"""
    start = time.time()
    # time.sleep(10)  # 模拟网络延迟
    try:
        # 地理编码
        geo = requests.get(
            'https://geocoding-api.open-meteo.com/v1/search',
            params={'name': city, 'count': 1, 'language': 'zh', 'format': 'json'},
            timeout=8
        ).json()

        if not geo.get('results'):
            return f'查询失败：未找到城市 "{city}"'

        r = geo['results'][0]
        lat, lon = r['latitude'], r['longitude']
        tz = r.get('timezone', 'Asia/Shanghai')

        # 解析日期
        target_date = None
        if date in ["today", "现在"]:
            target_date = None  # 实时天气
        elif date in ["tomorrow", "明天"]:
            target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        elif date == "后天":
            target_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        else:
            target_date = date  # 假设是 YYYY-MM-DD 格式

        # 查询天气
        if target_date is None:
            # 实时天气
            weather = requests.get(
                'https://api.open-meteo.com/v1/forecast',
                params={
                    'latitude': lat,
                    'longitude': lon,
                    'current': 'temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,wind_direction_10m,cloud_cover',
                    'timezone': tz
                },
                timeout=8
            ).json()

            cur = weather.get('current', {})
            desc = _wmo_mapping(cur.get('weather_code'))

            elapsed = time.time() - start
            print(f"_get_weather_impl 耗时: {elapsed:.2f}秒")

            return (
                f"城市：{r.get('name')} | 当前温度：{cur.get('temperature_2m')}℃ | 体感：{cur.get('apparent_temperature')}℃ | "
                f"湿度：{cur.get('relative_humidity_2m')}% | 风速：{cur.get('wind_speed_10m')} m/s | 风向：{cur.get('wind_direction_10m')}° | "
                f"云量：{cur.get('cloud_cover')}% | 降水：{cur.get('precipitation')} mm | 天气：{desc}"
            )
        else:
            # 未来/历史天气
            weather = requests.get(
                'https://api.open-meteo.com/v1/forecast',
                params={
                    'latitude': lat,
                    'longitude': lon,
                    'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,wind_speed_10m_max',
                    'start_date': target_date,
                    'end_date': target_date,
                    'timezone': tz
                },
                timeout=8
            ).json()

            daily = weather.get('daily', {})
            if not daily or not daily.get('time'):
                return f'查询失败：无法获取 {target_date} 的天气数据'

            desc = _wmo_mapping(daily['weather_code'][0])

            elapsed = time.time() - start
            print(f"_get_weather_impl 耗时: {elapsed:.2f}秒")
            return (
                f"城市：{r.get('name')} | 日期：{target_date} | "
                f"最高温度：{daily['temperature_2m_max'][0]}℃ | 最低温度：{daily['temperature_2m_min'][0]}℃ | "
                f"降水：{daily['precipitation_sum'][0]} mm | 最大风速：{daily['wind_speed_10m_max'][0]} m/s | 天气：{desc}"
            )
    except Exception as e:
        return f'查询失败：{str(e)}'

@tool("get_weather")
def get_weather_tool(city: str, date: str = "today") -> str:
    """获取指定城市的天气信息。

    参数:
    - city: 城市名（中文或英文，例："成都"、"Chengdu"、"北京"、"Beijing"）
    - date: 查询日期，支持格式：
        * "today" 或 "现在" - 实时天气（默认）
        * "tomorrow" 或 "明天" - 明天天气
        * "YYYY-MM-DD" - 指定日期（例："2026-03-15"）
        * 相对日期（例："3天后"、"后天"）
    """
    cache_key = f"get_weather:{city}:{date}"

    if _predictor and cache_key in _predictor.cache:
        future = _predictor.cache[cache_key]
        try:
            logger.info(f"使用预测缓存: {cache_key}")
            return future.result(timeout=20)
        except Exception as e:
            logger.warning(f"预测工具失败: {e}")

    return _get_weather_impl(city, date)
