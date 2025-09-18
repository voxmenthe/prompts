# event

### event : Event

The most recent Event object (could be a TouchEvent, PointerEvent, MouseEvent, WheelEvent, or ScrollEvent based on whatever `type` you define)

### Details[​](#details "Direct link to Details")

The most recent Event object (could be a TouchEvent, PointerEvent, MouseEvent, WheelEvent, or ScrollEvent based on whatever `type` you define). For example, if your Observer has `type: "touch,pointer"` and you press down and drag with your mouse, the `event` would be a PointerEvent or MouseEvent based on your browser/device. This allows you to get whatever information you need from that event like pageX, pageY, etc. For touch and pointer events, the event.clientX and event.clientY are automatically saved to the "x" and "y" properties of the Observer for convenience.
