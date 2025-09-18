# endX

### endX : Number

\[read-only] The ending x (horizontal) position of the Draggable instance which is calculated as soon as the mouse/touch is released after a drag, meaning you can use it to predict precisely where it'll land after an `inertia` flick.

### Details[​](#details "Direct link to Details")

*Number* - The ending `x` (horizontal) position of the Draggable instance. `endX` gets populated immediately when the mouse (or touch) is released after dragging, even before tweening has completed. This makes it easy to predict exactly where the element will land (useful for `inertia: true` Draggables where momentum gets applied). For a Draggable of `type: "x,y"`, `endX` would pertain to the `x` transform translation, as in the CSS `transform: translateX(...)`. For `type: "top,left"`, the Draggable's `x` would refer to the CSS `left` value that's applied. This is not the global coordinate - it is the inline CSS-related value applied to the element.
