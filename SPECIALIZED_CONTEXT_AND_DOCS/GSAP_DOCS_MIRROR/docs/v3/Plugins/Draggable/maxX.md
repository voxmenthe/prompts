# maxX

### maxX : Number

When bounds are applied, `maxX` refers to the maximum "legal" horizontal property.

### Details[​](#details "Direct link to Details")

*Number* - When bounds are `applied`, `maxX` refers to the maximum "legal" value of the horizontal property (either `"x"` or `"left"`, depending on which type the Draggable is). This makes it easier to run your own custom logic inside the snap or callback function(s) if you so choose. So for a Draggable of `type: "x,y"`, `maxX` would correlate with `x` transform translation, as in the CSS `transform: translateX(...)`. For `type: "top,left"`, the Draggable's `maxX` would correlate with the CSS `left` value that's applied. This is not the global coordinate - it is the inline CSS-related value applied to the element.
