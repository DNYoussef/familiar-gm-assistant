# Foundry VTT Module Development Research

## API Documentation Sources
- Official API: https://foundryvtt.com/api/
- Community Wiki: https://foundryvtt.wiki/
- Current Version: v13 (v11+ compatibility required)

## Key Technical Findings

### Canvas Overlay System
- **Overlay Canvas Group**: Rendered above other groups, contains elements not bound to stage transform
- **HeadsUpDisplay (HUD)**: Container for HTML rendering on top of Canvas
- **Hook System**: Extensive event system for module integration

### Critical API Hooks for Raven Familiar
1. **canvasInit**: Fired when Canvas is initialized - configure textures/visibility
2. **canvasReady**: Fired when Canvas is ready - safe for UI overlay creation
3. **getSceneControlButtons**: Modify left-side controls
4. **dropCanvasData**: Handle data dropped onto canvas

### Module Development Best Practices
- Modify CONFIG object during init hook
- Use appropriate canvas lifecycle hooks
- Register hooks with Hooks.on() or Hooks.once()
- Layer-specific hooks fire for CanvasLayer drawing

### UI Integration Strategy
- Use HeadsUpDisplay container for HTML overlays
- Overlay group for WebGL-safe rendering
- Canvas document classes have hudClass properties for HUD customization
- Scene controls can be modified via getSceneControlButtons hook

## Architecture Recommendations
1. **Raven Familiar**: Create as HUD overlay using HeadsUpDisplay container
2. **Chat Window**: Implement as modal overlay with HTML/CSS
3. **Performance**: Use appropriate hooks to minimize canvas impact
4. **Compatibility**: Target v11+ API for stability

## Risk Assessment
- **Low Risk**: HUD overlay system is well-established
- **Medium Risk**: Canvas API changes between versions
- **Mitigation**: Use stable v11+ API patterns, test across versions