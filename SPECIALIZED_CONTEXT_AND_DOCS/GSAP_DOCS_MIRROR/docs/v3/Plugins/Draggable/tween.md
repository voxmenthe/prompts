# tween

### tween : Tween

\[read-only] The Tween instance that gets created as soon as the mouse (or touch) is released (when `inertia` is `true`). This allows you to check its `duration`, `.pause()` or `.resume()` it, change its `timeScale`, or whatever you want.

### Details[​](#details "Direct link to Details")

*Tween* - The tween instance that gets created as soon as the mouse (or touch) is released (when `inertia` is `true`) - this allows you to check its `duration`, `pause()` it, `resume()` it, change its `timeScale`, or whatever you want. Keep in mind that a new tween is created each time the element is "thrown". You can easily get it when the user releases the mouse (or touch) by referencing `this.tween` inside the `onDragEnd` callback.
