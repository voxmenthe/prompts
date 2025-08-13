# endY

### endY : Number

\[read-only] The ending y (vertical) position of the Draggable instance which is calculated as soon as the mouse/touch is released after a drag, meaning you can use it to predict precisely where it'll land after a `inertia` flick.

### Details[​](#details "Direct link to Details")

*Number* - The ending `y` (vertical) position of the Draggable instance. `endY` gets populated immediately when the mouse (or touch) is released after dragging, even before tweening has completed. This makes it easy to predict exactly where the element will land (useful for `inertia: true` Draggables where momentum gets applied). For a Draggable of `type: "x,y"`, `endY` would pertain to the `y` transform translation, as in the CSS `transform: translateY(...)`. For `type: "top,left"`, the Draggable's `y` would refer to the CSS `top` value that's applied. This is not the global coordinate - it is the inline CSS-related value applied to the element.
