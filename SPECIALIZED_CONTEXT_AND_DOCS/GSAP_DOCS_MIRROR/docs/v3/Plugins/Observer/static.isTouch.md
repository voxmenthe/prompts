# Observer.isTouch

### Observer.isTouch : Number

A way to discern the touch capabilities of the current device - `0` is mouse/pointer only (no touch), `1` is touch-only, `2` accommodates both.

### Details[​](#details "Direct link to Details")

A way to discern the touch capabilities of the device:

* `0` - **no touch** (pointer/mouse only)
* `1` - **touch-only** device (like a phone)
* `2` - device can accept **touch** input and**mouse/pointer** (like Windows tablets)

```
if (Observer.isTouch) {
  // any touch-capable device...
}

// or get more specific:
if (Observer.isTouch === 1) {
  // touch-only device
}
```
