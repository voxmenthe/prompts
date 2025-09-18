# MorphSVGPlugin.defaultUpdateTarget

### MorphSVGPlugin.defaultUpdateTarget : Boolean

Sets the default `updateTarget` value for all MorphSVG animations; if `true`, the original tween target (typically an SVG `<path>` element) itself gets updated during the tween.

### Details[​](#details "Direct link to Details")

Sets the default `updateTarget` value for all MorphSVG animations; if `true`, the original tween target (typically an SVG `<path>` element) itself gets updated during the tween. If you've got a render function set up to draw to `<canvas>`, for example, then it may be wasteful to update the original target as well (duplicate efforts). Ultimately this is a performance optimization. Setting it to `false` allows MorphSVG to skip that step.
