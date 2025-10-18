# KissanDial MCP API Reference

## Overview

This document provides a comprehensive API reference for all MCP servers in KissanDial, including available tools, parameters, and response formats.

## Core MCP Servers

### 1. Subsidy MCP Server (`servers/subsidy_mcp.py`)

**Purpose**: Provides access to government agricultural subsidies and schemes.

#### Available Tools

##### `search_subsidies`
Search for subsidies based on query parameters.

**Parameters:**
```json
{
  "query": {
    "type": "string",
    "description": "Search query for subsidies",
    "required": true
  },
  "state": {
    "type": "string", 
    "description": "State to filter subsidies",
    "required": false
  },
  "category": {
    "type": "string",
    "description": "Subsidy category filter",
    "required": false
  }
}
```

**Response Format:**
```json
{
  "results": [
    {
      "scheme_name": "PM-KISAN Samman Nidhi",
      "description": "Direct income support to farmers",
      "amount": "₹6,000 per year",
      "eligibility": "Small and marginal farmers",
      "state": "Karnataka",
      "category": "Direct Benefit Transfer",
      "application_process": "Online through PM-KISAN portal",
      "documents_required": ["Aadhaar", "Bank Account", "Land Records"]
    }
  ],
  "total_count": 15,
  "search_query": "organic farming",
  "filters_applied": {
    "state": "Karnataka",
    "category": "Organic Farming"
  }
}
```

##### `get_subsidy_categories`
Get available subsidy categories for a state.

**Parameters:**
```json
{
  "state": {
    "type": "string",
    "description": "State name",
    "required": true
  }
}
```

**Response Format:**
```json
{
  "categories": [
    "Direct Benefit Transfer",
    "Crop Insurance", 
    "Organic Farming",
    "Equipment Subsidy",
    "Irrigation Support"
  ],
  "state": "Karnataka",
  "total_categories": 5
}
```

### 2. Price MCP Server (`servers/price_mcp.py`)

**Purpose**: Provides current and historical agricultural market prices.

#### Available Tools

##### `get_current_prices`
Get current market prices for crops.

**Parameters:**
```json
{
  "crop": {
    "type": "string",
    "description": "Crop name",
    "required": true
  },
  "district": {
    "type": "string",
    "description": "District name",
    "required": false
  },
  "market": {
    "type": "string", 
    "description": "Market/mandi name",
    "required": false
  }
}
```

**Response Format:**
```json
{
  "crop": "Tomato",
  "prices": [
    {
      "market": "Bangalore (Binny Mills)",
      "district": "Bangalore Urban",
      "state": "Karnataka",
      "price_per_quintal": 2500,
      "currency": "INR",
      "date": "2024-01-15",
      "price_trend": "increasing",
      "quality": "Grade A"
    }
  ],
  "average_price": 2500,
  "price_range": {
    "min": 2000,
    "max": 3000
  },
  "last_updated": "2024-01-15T10:30:00Z"
}
```

##### `get_price_trends`
Get price trends for crops over time.

**Parameters:**
```json
{
  "crop": {
    "type": "string",
    "description": "Crop name", 
    "required": true
  },
  "days": {
    "type": "integer",
    "description": "Number of days for trend analysis",
    "default": 30,
    "required": false
  },
  "district": {
    "type": "string",
    "description": "District name",
    "required": false
  }
}
```

**Response Format:**
```json
{
  "crop": "Tomato",
  "district": "Bangalore Urban",
  "trend_period": "30 days",
  "trend_direction": "increasing",
  "percentage_change": 15.5,
  "price_history": [
    {
      "date": "2024-01-01",
      "price": 2000
    },
    {
      "date": "2024-01-15", 
      "price": 2500
    }
  ],
  "statistics": {
    "average": 2250,
    "highest": 2800,
    "lowest": 1800,
    "volatility": "medium"
  }
}
```

### 3. Weather MCP Server (`servers/community/weather_mcp.py`)

**Purpose**: Provides weather information relevant to agriculture.

#### Available Tools

##### `get_current_weather`
Get current weather conditions.

**Parameters:**
```json
{
  "location": {
    "type": "string",
    "description": "Location (city, district, or coordinates)",
    "required": true
  }
}
```

**Response Format:**
```json
{
  "location": "Bangalore, Karnataka",
  "current_conditions": {
    "temperature": 25.5,
    "humidity": 68,
    "rainfall": 0,
    "wind_speed": 12,
    "wind_direction": "SW",
    "cloud_cover": 40,
    "uv_index": 6
  },
  "agricultural_advisory": {
    "suitable_for_spraying": true,
    "irrigation_needed": false,
    "pest_risk": "low",
    "disease_risk": "medium"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

##### `get_weather_forecast`
Get weather forecast for agricultural planning.

**Parameters:**
```json
{
  "location": {
    "type": "string",
    "description": "Location name",
    "required": true
  },
  "days": {
    "type": "integer",
    "description": "Number of days for forecast",
    "default": 7,
    "maximum": 14,
    "required": false
  }
}
```

**Response Format:**
```json
{
  "location": "Bangalore, Karnataka",
  "forecast": [
    {
      "date": "2024-01-16",
      "temperature": {
        "max": 28,
        "min": 18
      },
      "humidity": 65,
      "rainfall_probability": 20,
      "expected_rainfall": 0,
      "wind_speed": 15,
      "agricultural_suitability": {
        "field_work": "suitable",
        "spraying": "suitable", 
        "harvesting": "suitable"
      }
    }
  ],
  "weekly_summary": {
    "total_rainfall": 5,
    "average_temperature": 24,
    "suitable_working_days": 6
  }
}
```

## Integration Tools

### SMS Tool (`app/agent_mcp.py`)

**Purpose**: Send agricultural information via SMS.

#### Available Functions

##### `send_sms_with_subsidy_info`
Send SMS with agricultural information.

**Parameters:**
```json
{
  "query": {
    "type": "string",
    "description": "Query/information to send via SMS",
    "required": true
  }
}
```

**Response Format:**
```json
{
  "status": "success",
  "message_sid": "SM1234567890abcdef",
  "recipient": "+919108006252",
  "content": "Agricultural information for your query: organic farming\n\nFor detailed information, please call back or visit our website."
}
```

## Error Handling

All MCP servers follow consistent error response format:

### Standard Error Response
```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Invalid crop name provided",
    "details": {
      "parameter": "crop",
      "provided_value": "invalidcrop",
      "valid_values": ["rice", "wheat", "tomato", "onion"]
    }
  },
  "request_id": "req_1234567890"
}
```

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `INVALID_PARAMETER` | Invalid parameter value | Check parameter format and valid values |
| `MISSING_REQUIRED_PARAMETER` | Required parameter not provided | Include all required parameters |
| `DATA_NOT_FOUND` | No data found for query | Try different search terms or location |
| `EXTERNAL_API_ERROR` | External data source unavailable | Retry after some time |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait before making more requests |

## Data Sources

### Subsidy Data Sources
- Government subsidy database (CSV files)
- Official scheme announcements
- State agricultural department data
- Central government portal information

### Price Data Sources  
- Agricultural Marketing Division (AGMARKNET)
- Mandi price boards
- Commodity exchanges
- Local market surveys

### Weather Data Sources
- India Meteorological Department (IMD)
- Agricultural weather stations
- Satellite weather data
- Regional meteorological centers

## Rate Limits

| Server | Requests per minute | Daily limit |
|--------|-------------------|-------------|
| Subsidy MCP | 60 | 1000 |
| Price MCP | 100 | 2000 |
| Weather MCP | 120 | 3000 |

## Authentication

Currently, KissanDial MCP servers use:
- No authentication for local servers
- API keys for external weather services
- Environment variable configuration

Future implementations may include:
- JWT tokens for secure access
- API key management
- User-based rate limiting

## Caching

### Cache Duration
- Subsidy data: 24 hours
- Price data: 1 hour  
- Weather data: 30 minutes

### Cache Keys Format
```
kissan:{server}:{tool}:{params_hash}
```

Examples:
```
kissan:subsidy:search:md5(query=organic&state=karnataka)
kissan:price:current:md5(crop=tomato&district=bangalore)
kissan:weather:current:md5(location=bangalore)
```

## Testing & Development

### Testing Tools
```bash
# Test specific server
python servers/subsidy_mcp.py

# Test all tools
python tools/llm_test.py

# Test agent integration
python -c "
import asyncio
from app.agent_mcp import agent
asyncio.run(agent.run('Test query'))
"
```

### Mock Data
All servers include mock data for development:
- Subsidy: Sample government schemes
- Price: Historical price patterns
- Weather: Seasonal weather data

### Extending Servers

To add new tools to existing servers:

1. **Add tool definition** in `list_tools()` handler
2. **Implement tool logic** in `call_tool()` handler  
3. **Update bridge function** in `tools/mcp_bridge_simple.py`
4. **Add tests** for new functionality

## Production Considerations

### Monitoring
- Log all API calls and responses
- Track response times and error rates
- Monitor cache hit rates
- Set up alerts for service failures

### Security
- Input validation and sanitization
- Rate limiting implementation
- Secure API key storage
- Request/response logging (excluding sensitive data)

### Performance
- Async operation for all I/O
- Connection pooling for external APIs
- Efficient caching strategies
- Database query optimization

### Reliability
- Graceful error handling
- Fallback data sources
- Circuit breaker pattern for external APIs
- Health check endpoints

---

## Summary

The KissanDial MCP API provides comprehensive access to agricultural data through standardized protocols. All servers follow consistent patterns for:

- **Parameter validation**
- **Response formatting** 
- **Error handling**
- **Caching strategies**
- **Rate limiting**

This enables easy integration of new data sources and maintains system reliability for farmers accessing critical agricultural information.
