# deltaY

### deltaY : Number

The amount of change (in pixels) vertically since the last time a callback was fired on that axis. For example, `onChangeY` or `onDown`

### Details[​](#details "Direct link to Details")

The amount of change (in pixels) vertically since the last time a callback was fired on that axis. For example, `onChangeY` or `onDown`

This is only affected by the event types that the Observer is watching. So, for example, `type: "wheel,touch"` would track the delta based on wheel and touch events (not pointer or scroll). By default, touch and pointer events are only tracked **while pressing/dragging** but if you define an `onMove` (which is mapped to "pointermove"/"mousemove" events), it'll be tracked during any movement while hovering over the target.
