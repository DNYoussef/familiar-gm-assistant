# Pathfinder 2e API and Data Source Research

## Archives of Nethys Overview
- Official URL: https://2e.aonprd.com/
- Status: Official Paizo-licensed SRD reference
- Backend: Elasticsearch (elasticsearch.aonprd.com)
- Data Coverage: Complete Pathfinder 2e SRD content

## API Access Options

### 1. Community API (GitHub: SargntSprinkles/Pathfinder-2e-API)
- **Status**: Unofficial community project
- **Data Source**: Scraped from Archives of Nethys
- **Standardization**: "Standardized set of Pathfinder Second Edition data for community use"
- **Risk**: Dependent on unofficial maintenance

### 2. Direct Elasticsearch Access
- **Method**: POST calls to elasticsearch.aonprd.com
- **Data**: Raw Archives of Nethys data
- **Structure**: Elasticsearch JSON format
- **Risk**: Unofficial access method, may change

### 3. Web Scraping
- **Method**: Direct HTML parsing of 2e.aonprd.com
- **Coverage**: All visible content
- **Risk**: Against terms of service, rate limiting

## Legal Compliance Analysis

### Paizo Community Use Policy
- Archives of Nethys content is **NOT** available under Community Use License
- Uses Paizo's Product Identity under commercial license
- Direct commercial use requires separate licensing

### Terms of Service Considerations
- Scraping may violate website terms of service
- Check robots.txt and terms before automated access
- Rate limiting and respectful access essential

### License Page Reference
- Full details at: https://2e.aonprd.com/Licenses.aspx
- Must review for current commercial use restrictions
- Product Identity vs Open Game License distinctions

## Data Structure Assessment

### Available Content Types
- Rules text and mechanics
- Monster stat blocks and abilities
- Spell descriptions and effects
- Equipment and item statistics
- Class features and abilities

### Integration Strategy
1. **Phase 1**: Use community API for basic functionality
2. **Phase 2**: Implement caching layer for performance
3. **Phase 3**: Evaluate direct partnership opportunities

## Risk Mitigation
- **Legal**: Obtain proper licensing for commercial use
- **Technical**: Implement aggressive caching to reduce API calls
- **Reliability**: Mirror critical data locally for offline fallback
- **Performance**: Rate limiting and request optimization

## Recommendations
1. Start with community API for prototype
2. Implement robust caching strategy
3. Pursue official Paizo partnership for commercial version
4. Plan for offline fallback with local data mirror