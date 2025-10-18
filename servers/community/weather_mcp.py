#!/usr/bin/env python3
"""
Weather MCP Server for KissanDial
Provides weather information through Model Context Protocol
This is a demo server with mock data - replace with real weather API
"""

import asyncio
import json
import random
import sys
import os
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from pydantic import BaseModel

# Load environment variables
load_dotenv()

class WeatherQuery(BaseModel):
    location: str
    days: int = 1

class WeatherMCPServer:
    def __init__(self):
        self.server = Server("weather-server")
        self.api_key = os.getenv('WEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.setup_handlers()

    async def get_real_weather_data(self, location: str, endpoint: str = "weather") -> Dict:
        """Fetch real weather data from OpenWeatherMap API"""
        if not self.api_key:
            print("No weather API key found, using mock data", file=sys.stderr)
            return None
            
        try:
            url = f"{self.base_url}/{endpoint}"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"Weather API error: {response.status}", file=sys.stderr)
                        return None
        except Exception as e:
            print(f"Error fetching weather data: {e}", file=sys.stderr)
            return None

    def generate_mock_weather_data(self, location: str) -> Dict:
        """Generate mock weather data as fallback"""
        return {
            'main': {
                'temp': random.randint(20, 35),
                'humidity': random.randint(60, 90),
                'pressure': random.randint(1000, 1020)
            },
            'weather': [{'main': random.choice(['Clear', 'Clouds', 'Rain']), 'description': 'mock data'}],
            'wind': {'speed': random.randint(5, 20)},
            'name': location
        }

    def setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="get_current_weather",
                    description="Get current weather conditions for a location",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location name (city, district, or coordinates)"
                            }
                        },
                        "required": ["location"]
                    }
                ),
                Tool(
                    name="get_weather_forecast",
                    description="Get weather forecast for a location",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location name (city, district, or coordinates)"
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of days for forecast (1-7)",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 7
                            }
                        },
                        "required": ["location"]
                    }
                ),
                Tool(
                    name="get_agricultural_weather_alert",
                    description="Get weather alerts relevant to agriculture",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location name"
                            },
                            "crop_type": {
                                "type": "string",
                                "description": "Type of crop (optional)",
                                "default": "general"
                            }
                        },
                        "required": ["location"]
                    }
                ),
                Tool(
                    name="get_rainfall_data",
                    description="Get rainfall data and predictions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location name"
                            },
                            "period": {
                                "type": "string",
                                "description": "Time period (weekly, monthly)",
                                "default": "weekly"
                            }
                        },
                        "required": ["location"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
            """Handle tool calls"""
            if arguments is None:
                arguments = {}
                
            try:
                if name == "get_current_weather":
                    return await self.get_current_weather(arguments)
                elif name == "get_weather_forecast":
                    return await self.get_weather_forecast(arguments)
                elif name == "get_agricultural_weather_alert":
                    return await self.get_agricultural_weather_alert(arguments)
                elif name == "get_rainfall_data":
                    return await self.get_rainfall_data(arguments)
                else:
                    return [TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error executing tool {name}: {str(e)}"
                )]

    async def get_current_weather(self, arguments: dict) -> list[TextContent]:
        """Get current weather conditions using real API"""
        location = arguments.get("location", "Unknown")
        
        # Try to get real weather data first
        weather_data = await self.get_real_weather_data(location)
        
        if weather_data:
            # Parse real API response
            temp = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            pressure = weather_data['main']['pressure']
            condition = weather_data['weather'][0]['main']
            description = weather_data['weather'][0]['description']
            wind_speed = weather_data['wind']['speed']
            location_name = weather_data['name']
            
            result = f"Current Weather in {location_name}:\\n\\n"
            result += f"Temperature: {temp}°C\\n"
            result += f"Condition: {condition} ({description})\\n"
            result += f"Humidity: {humidity}%\\n"
            result += f"Pressure: {pressure} hPa\\n"
            result += f"Wind Speed: {wind_speed} m/s\\n"
            result += f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
            
            # Add agricultural advice based on real data
            if temp > 30:
                result += "Agricultural Advisory: High temperature. Ensure adequate irrigation for crops.\\n"
            if humidity > 80:
                result += "Agricultural Advisory: High humidity. Monitor crops for fungal diseases.\\n"
            if condition in ["Rain", "Thunderstorm"]:
                result += "Agricultural Advisory: Wet conditions. Protect crops and delay spraying operations.\\n"
            if wind_speed > 10:
                result += "Agricultural Advisory: Strong winds. Secure loose farm equipment and support tall crops.\\n"
            
            result += "\\n*Data from OpenWeatherMap API"
        else:
            # Fallback to mock data
            mock_data = self.generate_mock_weather_data(location)
            temp = mock_data['main']['temp']
            humidity = mock_data['main']['humidity']
            condition = mock_data['weather'][0]['main']
            wind_speed = mock_data['wind']['speed']
            
            result = f"Current Weather in {location} (Mock Data):\\n\\n"
            result += f"Temperature: {temp}°C\\n"
            result += f"Condition: {condition}\\n"
            result += f"Humidity: {humidity}%\\n"
            result += f"Wind Speed: {wind_speed} km/h\\n"
            result += f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
            result += "Agricultural Advisory: Check real weather conditions before farm operations.\\n"
            result += "\\n*Note: This is mock weather data for demonstration purposes."
        
        return [TextContent(type="text", text=result)]

    async def get_weather_forecast(self, arguments: dict) -> list[TextContent]:
        """Get weather forecast using real API"""
        location = arguments.get("location", "Unknown")
        days = min(arguments.get("days", 3), 5)  # OpenWeatherMap free tier gives 5 days
        
        # Try to get real forecast data
        forecast_data = await self.get_real_weather_data(location, "forecast")
        
        result = f"{days}-Day Weather Forecast for {location}:\\n\\n"
        
        if forecast_data and 'list' in forecast_data:
            # Parse real forecast data
            forecasts = forecast_data['list'][:days * 8]  # 8 forecasts per day (3-hour intervals)
            
            daily_forecasts = {}
            for forecast in forecasts:
                date = datetime.fromtimestamp(forecast['dt']).date()
                if date not in daily_forecasts:
                    daily_forecasts[date] = {
                        'temps': [],
                        'conditions': [],
                        'humidity': [],
                        'rain': 0
                    }
                
                daily_forecasts[date]['temps'].append(forecast['main']['temp'])
                daily_forecasts[date]['conditions'].append(forecast['weather'][0]['main'])
                daily_forecasts[date]['humidity'].append(forecast['main']['humidity'])
                
                if 'rain' in forecast:
                    daily_forecasts[date]['rain'] += forecast['rain'].get('3h', 0)
            
            for date, data in list(daily_forecasts.items())[:days]:
                min_temp = min(data['temps'])
                max_temp = max(data['temps'])
                avg_humidity = sum(data['humidity']) / len(data['humidity'])
                most_common_condition = max(set(data['conditions']), key=data['conditions'].count)
                
                day_name = date.strftime('%A')
                result += f"{day_name} ({date.strftime('%m-%d')}):\\n"
                result += f"  Temperature: {min_temp:.1f}°C - {max_temp:.1f}°C\\n"
                result += f"  Condition: {most_common_condition}\\n"
                result += f"  Humidity: {avg_humidity:.0f}%\\n"
                if data['rain'] > 0:
                    result += f"  Rainfall: {data['rain']:.1f}mm\\n"
                result += "\\n"
                
            result += "\\n*Data from OpenWeatherMap API"
        else:
            # Fallback to mock forecast
            conditions = ["Clear", "Partly Cloudy", "Cloudy", "Light Rain", "Heavy Rain"]
            
            for day in range(days):
                date = datetime.now() + timedelta(days=day)
                min_temp = random.randint(18, 25)
                max_temp = random.randint(26, 35)
                condition = random.choice(conditions)
                rain_chance = random.randint(0, 100)
                
                day_name = date.strftime('%A')
                result += f"{day_name} ({date.strftime('%m-%d')}):\\n"
                result += f"  Temperature: {min_temp}°C - {max_temp}°C\\n"
                result += f"  Condition: {condition}\\n"
                result += f"  Rain Chance: {rain_chance}%\\n\\n"
            
            result += "\\n*Note: This is mock weather data for demonstration purposes."
        
        # Add agricultural advice
        result += "\\nAgricultural Advisory:\\n"
        result += "- Monitor soil moisture levels daily\\n"
        result += "- Plan irrigation based on weather predictions\\n"
        result += "- Check weather before applying chemicals\\n"
        
        return [TextContent(type="text", text=result)]

    async def get_agricultural_weather_alert(self, arguments: dict) -> list[TextContent]:
        """Get agricultural weather alerts"""
        location = arguments.get("location", "Unknown")
        crop_type = arguments.get("crop_type", "general")
        
        # Generate mock alerts
        alerts = []
        
        # Random chance of different types of alerts
        if random.random() < 0.3:
            alerts.append("HEAT WAVE WARNING: Temperatures may exceed 35°C for next 3 days")
        
        if random.random() < 0.4:
            alerts.append("HEAVY RAINFALL ALERT: 50-100mm rainfall expected in next 48 hours")
        
        if random.random() < 0.2:
            alerts.append("WIND ADVISORY: Strong winds (>25 km/h) expected tomorrow")
        
        if random.random() < 0.3:
            alerts.append("HUMIDITY ALERT: High humidity (>85%) may increase disease risk")
        
        result = f"Agricultural Weather Alerts for {location}:\\n"
        if crop_type != "general":
            result += f"Crop-specific alerts for {crop_type}:\\n"
        result += "\\n"
        
        if alerts:
            for i, alert in enumerate(alerts, 1):
                result += f"{i}. {alert}\\n\\n"
                
                # Add specific advice based on alert type
                if "HEAT WAVE" in alert:
                    result += "   Recommended Actions:\\n"
                    result += "   - Increase irrigation frequency\\n"
                    result += "   - Provide shade for sensitive crops\\n"
                    result += "   - Avoid mid-day field operations\\n\\n"
                elif "RAINFALL" in alert:
                    result += "   Recommended Actions:\\n"
                    result += "   - Ensure proper drainage\\n"
                    result += "   - Postpone spraying operations\\n"
                    result += "   - Protect harvested crops\\n\\n"
                elif "WIND" in alert:
                    result += "   Recommended Actions:\\n"
                    result += "   - Provide support to tall crops\\n"
                    result += "   - Avoid spraying operations\\n"
                    result += "   - Secure farm equipment\\n\\n"
        else:
            result += "No active weather alerts for your area.\\n\\n"
            result += "General recommendations:\\n"
            result += "- Continue regular monitoring of crops\\n"
            result += "- Maintain optimal irrigation schedule\\n"
            result += "- Check weather updates daily\\n\\n"
        
        result += f"Alert issued: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n"
        result += "\\n*Note: These are mock weather alerts for demonstration purposes."
        
        return [TextContent(type="text", text=result)]

    async def get_rainfall_data(self, arguments: dict) -> list[TextContent]:
        """Get rainfall data and predictions"""
        location = arguments.get("location", "Unknown")
        period = arguments.get("period", "weekly")
        
        result = f"Rainfall Data for {location} ({period}):\\n\\n"
        
        if period == "weekly":
            total_rainfall = 0
            result += "7-Day Rainfall Forecast:\\n"
            
            for day in range(7):
                date = datetime.now() + timedelta(days=day)
                rainfall = random.randint(0, 25) if random.random() < 0.6 else 0
                total_rainfall += rainfall
                
                day_name = date.strftime('%A')
                result += f"{day_name}: {rainfall}mm\\n"
            
            result += f"\\nTotal Expected: {total_rainfall}mm\\n"
            
            if total_rainfall < 10:
                result += "Status: LOW - Irrigation may be needed\\n"
            elif total_rainfall < 50:
                result += "Status: MODERATE - Monitor soil moisture\\n"
            else:
                result += "Status: HIGH - Ensure proper drainage\\n"
                
        elif period == "monthly":
            # Generate monthly data
            weeks_data = []
            total_monthly = 0
            
            result += "Monthly Rainfall Outlook:\\n"
            
            for week in range(4):
                week_rainfall = random.randint(15, 80)
                weeks_data.append(week_rainfall)
                total_monthly += week_rainfall
                
                result += f"Week {week + 1}: {week_rainfall}mm\\n"
            
            result += f"\\nMonthly Total Expected: {total_monthly}mm\\n"
            
            avg_monthly = 120  # Mock average
            if total_monthly < avg_monthly * 0.7:
                result += f"Status: BELOW NORMAL (Normal: {avg_monthly}mm)\\n"
            elif total_monthly > avg_monthly * 1.3:
                result += f"Status: ABOVE NORMAL (Normal: {avg_monthly}mm)\\n"
            else:
                result += f"Status: NORMAL (Normal: {avg_monthly}mm)\\n"
        
        result += "\\nAgricultural Recommendations:\\n"
        if total_rainfall < 30:
            result += "- Plan for supplemental irrigation\\n"
            result += "- Consider drought-resistant crop varieties\\n"
        elif total_rainfall > 100:
            result += "- Ensure adequate field drainage\\n"
            result += "- Monitor for water-logging\\n"
            result += "- Delay field operations during heavy rain\\n"
        else:
            result += "- Maintain regular crop monitoring\\n"
            result += "- Optimize irrigation schedule\\n"
        
        result += f"\\nData updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n"
        result += "\\n*Note: This is mock rainfall data for demonstration purposes."
        
        return [TextContent(type="text", text=result)]

async def main():
    """Main function to run the MCP server"""
    server_instance = WeatherMCPServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            server_instance.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
