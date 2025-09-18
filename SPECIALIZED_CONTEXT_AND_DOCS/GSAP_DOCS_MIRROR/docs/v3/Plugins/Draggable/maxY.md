# maxY

### maxY : Number

When bounds are applied, `maxY` refers to the maximum "legal" vertical property.

### Details[​](#details "Direct link to Details")

*Number* - When bounds are `applied`, `maxY` refers to the maximum "legal" value of the horizontal property (either `"y"` or `"top"`, depending on which type the Draggable is). This makes it easier to run your own custom logic inside the snap or callback function(s) if you so choose. So for a Draggable of `type: "x,y"`, `maxY` would correlate with `y` transform translation, as in the CSS `transform: translateY(...)`. For `type: "top,left"`, the Draggable's `maxY` would correlate with the CSS `top` value that's applied. This is not the global coordinate - it is the inline CSS-related value applied to the element.
