#!/usr/bin/env python3
"""
Agricultural Market Price MCP Server for KissanDial
Provides market price information through Model Context Protocol
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

class PriceQuery(BaseModel):
    crop: str
    district: str = "All"
    state: str = "Karnataka"

class PriceMCPServer:
    def __init__(self):
        self.server = Server("price-server")
        self.market_api_key = os.getenv('MARKET_API_KEY')
        self.agro_api_key = os.getenv('AGRO_API_KEY')
        self.mock_prices = self.initialize_mock_data()
        self.setup_handlers()

    async def get_real_market_data(self, crop: str, district: str = "All") -> Dict:
        """Fetch real market data from agricultural APIs"""
        if not self.market_api_key:
            print("No market API key found, using mock data", file=sys.stderr)
            return None
            
        try:
            # This is a placeholder for real market API integration
            # Replace with actual agricultural market API endpoints
            
            # Example using Indian government's data.gov.in API
            base_url = "https://api.data.gov.in/resource"
            
            # Common Indian crop names mapping
            crop_mapping = {
                'rice': 'Rice',
                'wheat': 'Wheat', 
                'tomato': 'Tomato',
                'onion': 'Onion',
                'potato': 'Potato'
            }
            
            crop_standard = crop_mapping.get(crop.lower(), crop)
            
            params = {
                'api-key': self.market_api_key,
                'format': 'json',
                'filters[commodity]': crop_standard
            }
            
            if district != "All":
                params['filters[district]'] = district
            
            async with aiohttp.ClientSession() as session:
                # Note: This is a mock endpoint - replace with real agricultural API
                url = f"{base_url}/some-market-endpoint"
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        print(f"Market API error: {response.status}", file=sys.stderr)
                        return None
                        
        except Exception as e:
            print(f"Error fetching market data: {e}", file=sys.stderr)
            return None

    async def get_agro_insights(self, crop: str, location: str) -> Dict:
        """Get agricultural insights and recommendations"""
        if not self.agro_api_key:
            return None
            
        try:
            # Placeholder for agricultural insights API
            # This could integrate with services like:
            # - FarmLogs API
            # - Climate FieldView API  
            # - John Deere Operations Center API
            # - Agricultural Research APIs
            
            insights = {
                'recommendations': [
                    f"Optimal planting season for {crop} in {location}",
                    f"Current market trends for {crop}",
                    f"Disease prevention tips for {crop}"
                ],
                'market_outlook': 'positive',
                'seasonal_advice': f"Good time to cultivate {crop}"
            }
            
            return insights
            
        except Exception as e:
            print(f"Error fetching agro insights: {e}", file=sys.stderr)
            return None

    def initialize_mock_data(self) -> Dict[str, Dict[str, float]]:
        """Initialize mock price data for different crops and districts"""
        crops = [
            "Rice", "Wheat", "Jowar", "Bajra", "Maize", "Ragi",
            "Tur/Arhar", "Moong", "Urad", "Masoor", "Gram",
            "Groundnut", "Sunflower", "Soybean", "Sesamum",
            "Cotton", "Sugarcane", "Turmeric", "Coriander",
            "Onion", "Potato", "Tomato", "Chilli", "Garlic",
            "Ginger", "Coconut", "Banana", "Mango", "Orange"
        ]
        
        districts = [
            "Bangalore Urban", "Bangalore Rural", "Mysuru", "Mandya", 
            "Hassan", "Tumkur", "Kolar", "Chikkaballapur", "Ramanagara",
            "Chitradurga", "Davanagere", "Bellary", "Bidar", "Gulbarga",
            "Raichur", "Koppal", "Gadag", "Haveri", "Dharwad", "Belgaum",
            "Bagalkot", "Vijayapura", "Bijapur", "Shimoga", "Chikkamagaluru",
            "Udupi", "Dakshina Kannada", "Uttara Kannada", "Kodagu"
        ]
        
        mock_data = {}
        
        for crop in crops:
            mock_data[crop] = {}
            base_price = random.uniform(2000, 8000)  # Base price in INR per quintal
            
            for district in districts:
                # Add some variation based on district (±20%)
                variation = random.uniform(0.8, 1.2)
                mock_data[crop][district] = round(base_price * variation, 2)
                
        return mock_data

    def setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="get_mandi_price",
                    description="Get current market price for a crop in a specific district",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "crop": {
                                "type": "string",
                                "description": "Name of the crop (e.g., 'Rice', 'Wheat', 'Tomato', 'Onion')"
                            },
                            "district": {
                                "type": "string",
                                "description": "District name (e.g., 'Mysuru', 'Bangalore Urban')",
                                "default": "All"
                            },
                            "state": {
                                "type": "string",
                                "description": "State name",
                                "default": "Karnataka"
                            }
                        },
                        "required": ["crop"]
                    }
                ),
                Tool(
                    name="get_price_trends",
                    description="Get price trends for a crop over the last 30 days",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "crop": {
                                "type": "string",
                                "description": "Name of the crop"
                            },
                            "district": {
                                "type": "string",
                                "description": "District name",
                                "default": "Mysuru"
                            }
                        },
                        "required": ["crop"]
                    }
                ),
                Tool(
                    name="compare_crop_prices",
                    description="Compare prices of multiple crops in a district",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "crops": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of crop names to compare"
                            },
                            "district": {
                                "type": "string",
                                "description": "District name",
                                "default": "Mysuru"
                            }
                        },
                        "required": ["crops"]
                    }
                ),
                Tool(
                    name="get_available_crops",
                    description="Get list of crops with available price data",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="get_available_districts",
                    description="Get list of districts with available price data",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
            """Handle tool calls"""
            if arguments is None:
                arguments = {}
                
            try:
                if name == "get_mandi_price":
                    return await self.get_mandi_price(arguments)
                elif name == "get_price_trends":
                    return await self.get_price_trends(arguments)
                elif name == "compare_crop_prices":
                    return await self.compare_crop_prices(arguments)
                elif name == "get_available_crops":
                    return await self.get_available_crops()
                elif name == "get_available_districts":
                    return await self.get_available_districts()
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

    async def get_mandi_price(self, arguments: dict) -> list[TextContent]:
        """Get current market price for a crop with real API integration"""
        crop = arguments.get("crop", "").title()
        district = arguments.get("district", "All")
        state = arguments.get("state", "Karnataka")
        
        # Try to get real market data first
        market_data = await self.get_real_market_data(crop, district)
        
        if market_data and 'records' in market_data:
            # Parse real API response
            records = market_data['records']
            if records:
                record = records[0]  # Get first record
                price = record.get('modal_price', 0)
                market_name = record.get('market', district)
                date = record.get('arrival_date', datetime.now().strftime('%Y-%m-%d'))
                
                result = f"Current Market Price for {crop}:\n\n"
                result += f"Market: {market_name}, {state}\n"
                result += f"Price: ₹{price}/quintal\n"
                result += f"Date: {date}\n\n"
                
                # Get agricultural insights
                insights = await self.get_agro_insights(crop, district)
                if insights:
                    result += "Agricultural Insights:\n"
                    for rec in insights['recommendations'][:3]:
                        result += f"• {rec}\n"
                    result += f"\nMarket Outlook: {insights['market_outlook']}\n"
                
                result += "\n*Data from real market APIs"
                
            else:
                result = f"No current market data found for {crop} in {district}"
        else:
            # Fallback to enhanced mock data
            if crop not in self.mock_prices:
                available_crops = list(self.mock_prices.keys())[:10]
                return [TextContent(
                    type="text",
                    text=f"Crop '{crop}' not found. Available crops include: {', '.join(available_crops)}..."
                )]
            
            if district == "All":
                # Return average price across all districts
                prices = list(self.mock_prices[crop].values())
                avg_price = sum(prices) / len(prices)
                
                result = f"Current Market Prices for {crop} in {state} (Mock Data):\n\n"
                result += f"Average Price: ₹{avg_price:.2f} per quintal\n\n"
                result += "District-wise Prices:\n"
                
                # Show top 10 districts
                sorted_districts = sorted(self.mock_prices[crop].items(), key=lambda x: x[1], reverse=True)
                for district_name, price in sorted_districts[:10]:
                    result += f"• {district_name}: ₹{price:.2f}/quintal\n"
                    
                result += f"\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                result += "\n\n*Note: This is mock data. Enable market API for real prices."
                
            else:
                if district not in self.mock_prices[crop]:
                    available_districts = list(self.mock_prices[crop].keys())[:10]
                    return [TextContent(
                        type="text",
                        text=f"District '{district}' not found. Available districts include: {', '.join(available_districts)}..."
                    )]
                
                price = self.mock_prices[crop][district]
                
                # Add some random daily variation (±5%)
                daily_variation = random.uniform(0.95, 1.05)
                current_price = round(price * daily_variation, 2)
                
                result = f"Current Market Price for {crop} (Mock Data):\n\n"
                result += f"District: {district}, {state}\n"
                result += f"Price: ₹{current_price:.2f} per quintal\n"
                result += f"Previous Day: ₹{price:.2f} per quintal\n"
                
                change = current_price - price
                change_pct = (change / price) * 100
                
                if change > 0:
                    result += f"Change: +₹{change:.2f} (+{change_pct:.1f}%)\n"
                elif change < 0:
                    result += f"Change: ₹{change:.2f} ({change_pct:.1f}%)\n"
                else:
                    result += "Change: No change\n"
                    
                result += f"\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                result += "\n\n*Note: This is mock data. Enable market API for real prices."
        
        return [TextContent(type="text", text=result)]

    async def get_price_trends(self, arguments: dict) -> list[TextContent]:
        """Get price trends for a crop over the last 30 days"""
        crop = arguments.get("crop", "").title()
        district = arguments.get("district", "Mysuru")
        
        if crop not in self.mock_prices:
            return [TextContent(
                type="text",
                text=f"Crop '{crop}' not found in price database."
            )]
        
        if district not in self.mock_prices[crop]:
            district = list(self.mock_prices[crop].keys())[0]  # Use first available district
        
        base_price = self.mock_prices[crop][district]
        
        result = f"30-Day Price Trend for {crop} in {district}:\n\n"
        
        # Generate mock 30-day trend
        prices = []
        current_date = datetime.now()
        
        for i in range(30, 0, -1):
            date = current_date - timedelta(days=i)
            # Generate realistic price variation
            trend_factor = 1 + (random.random() - 0.5) * 0.1  # ±5% variation
            daily_price = round(base_price * trend_factor, 2)
            prices.append((date, daily_price))
        
        # Show weekly averages
        weeks = []
        for week in range(4):
            week_start = week * 7
            week_end = min((week + 1) * 7, 30)
            week_prices = [price for _, price in prices[week_start:week_end]]
            week_avg = sum(week_prices) / len(week_prices)
            weeks.append(week_avg)
            
            week_date = current_date - timedelta(days=30 - week * 7)
            result += f"Week of {week_date.strftime('%Y-%m-%d')}: ₹{week_avg:.2f}/quintal\n"
        
        # Calculate overall trend
        if len(weeks) >= 2:
            trend_change = weeks[-1] - weeks[0]
            trend_pct = (trend_change / weeks[0]) * 100
            
            if trend_pct > 2:
                trend_direction = "Increasing ↗️"
            elif trend_pct < -2:
                trend_direction = "Decreasing ↘️"
            else:
                trend_direction = "Stable ➡️"
            
            result += f"\nOverall Trend: {trend_direction} ({trend_pct:+.1f}%)\n"
        
        result += f"Current Price: ₹{weeks[-1]:.2f}/quintal\n"
        result += "\n*Note: These are simulated prices for demonstration purposes."
        
        return [TextContent(type="text", text=result)]

    async def compare_crop_prices(self, arguments: dict) -> list[TextContent]:
        """Compare prices of multiple crops"""
        crops = arguments.get("crops", [])
        district = arguments.get("district", "Mysuru")
        
        if not crops:
            return [TextContent(
                type="text",
                text="Please provide a list of crops to compare."
            )]
        
        result = f"Price Comparison in {district}:\n\n"
        
        crop_prices = []
        for crop in crops:
            crop_title = crop.title()
            if crop_title in self.mock_prices:
                if district in self.mock_prices[crop_title]:
                    price = self.mock_prices[crop_title][district]
                    # Add daily variation
                    current_price = round(price * random.uniform(0.95, 1.05), 2)
                    crop_prices.append((crop_title, current_price))
                else:
                    result += f"• {crop_title}: Price data not available for {district}\n"
            else:
                result += f"• {crop_title}: Crop not found in database\n"
        
        if crop_prices:
            # Sort by price (descending)
            crop_prices.sort(key=lambda x: x[1], reverse=True)
            
            result += "Prices (per quintal):\n"
            for i, (crop, price) in enumerate(crop_prices, 1):
                result += f"{i}. {crop}: ₹{price:.2f}\n"
            
            if len(crop_prices) >= 2:
                highest = crop_prices[0]
                lowest = crop_prices[-1]
                difference = highest[1] - lowest[1]
                result += f"\nHighest: {highest[0]} (₹{highest[1]:.2f})\n"
                result += f"Lowest: {lowest[0]} (₹{lowest[1]:.2f})\n"
                result += f"Difference: ₹{difference:.2f}\n"
        
        result += f"\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        result += "\n\n*Note: These are simulated prices for demonstration purposes."
        
        return [TextContent(type="text", text=result)]

    async def get_available_crops(self) -> list[TextContent]:
        """Get list of available crops"""
        crops = sorted(self.mock_prices.keys())
        
        result = "Available Crops with Price Data:\n\n"
        
        # Group crops by category
        cereals = ["Rice", "Wheat", "Jowar", "Bajra", "Maize", "Ragi"]
        pulses = ["Tur/Arhar", "Moong", "Urad", "Masoor", "Gram"]
        oilseeds = ["Groundnut", "Sunflower", "Soybean", "Sesamum"]
        cash_crops = ["Cotton", "Sugarcane", "Turmeric", "Coriander"]
        vegetables = ["Onion", "Potato", "Tomato", "Chilli", "Garlic", "Ginger"]
        fruits = ["Coconut", "Banana", "Mango", "Orange"]
        
        categories = [
            ("Cereals", cereals),
            ("Pulses", pulses),
            ("Oilseeds", oilseeds),
            ("Cash Crops", cash_crops),
            ("Vegetables", vegetables),
            ("Fruits", fruits)
        ]
        
        for category, crop_list in categories:
            available_in_category = [crop for crop in crop_list if crop in crops]
            if available_in_category:
                result += f"{category}:\n"
                for crop in available_in_category:
                    result += f"  • {crop}\n"
                result += "\n"
        
        result += f"Total: {len(crops)} crops available"
        
        return [TextContent(type="text", text=result)]

    async def get_available_districts(self) -> list[TextContent]:
        """Get list of available districts"""
        # Get districts from the first crop's data
        first_crop = list(self.mock_prices.keys())[0]
        districts = sorted(self.mock_prices[first_crop].keys())
        
        result = "Available Districts in Karnataka:\n\n"
        
        # Group districts by region
        north_karnataka = ["Bidar", "Gulbarga", "Raichur", "Koppal", "Gadag", "Haveri", "Dharwad", "Belgaum", "Bagalkot", "Vijayapura", "Bijapur"]
        central_karnataka = ["Bangalore Urban", "Bangalore Rural", "Tumkur", "Kolar", "Chikkaballapur", "Ramanagara", "Chitradurga", "Davanagere", "Bellary"]
        south_karnataka = ["Mysuru", "Mandya", "Hassan", "Shimoga", "Chikkamagaluru", "Kodagu"]
        coastal_karnataka = ["Udupi", "Dakshina Kannada", "Uttara Kannada"]
        
        regions = [
            ("North Karnataka", north_karnataka),
            ("Central Karnataka", central_karnataka),
            ("South Karnataka", south_karnataka),
            ("Coastal Karnataka", coastal_karnataka)
        ]
        
        for region, district_list in regions:
            available_in_region = [district for district in district_list if district in districts]
            if available_in_region:
                result += f"{region}:\n"
                for district in available_in_region:
                    result += f"  • {district}\n"
                result += "\n"
        
        result += f"Total: {len(districts)} districts available"
        
        return [TextContent(type="text", text=result)]

async def main():
    """Main function to run the MCP server"""
    server_instance = PriceMCPServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            server_instance.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
