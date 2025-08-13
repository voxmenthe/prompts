# endRotation

### endRotation : Number

\[read-only] \[only applies to type:"rotation"] The ending rotation of the Draggable instance which is calculated as soon as the mouse/touch is released after a drag, meaning you can use it to predict precisely where it'll land after a `inertia` flick.

### Details[​](#details "Direct link to Details")

*Number* - \[only applies to `type: "rotation"` Draggable objects] The ending rotation of the Draggable instance. `endRotation` gets populated immediately when the mouse (or touch) is released after dragging, even before tweening has completed. This makes it easy to predict exactly what angle the element will land at (useful for `inertia: true` Draggables where momentum gets applied and you want to predict where it'll land).
