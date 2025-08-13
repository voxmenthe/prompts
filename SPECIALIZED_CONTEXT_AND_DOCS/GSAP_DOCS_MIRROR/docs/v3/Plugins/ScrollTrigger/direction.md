# .direction

### .direction : Number

\[read-only] Reflects the moment-by-moment direction of scrolling where `1` is forward and `-1` is backward.

### Details[​](#details "Direct link to Details")

\[read-only] Reflects the moment-by-moment direction of scrolling where `1` is forward and `-1` is backward.

## Example[​](#example "Direct link to Example")

```
ScrollTrigger.create({
  trigger: ".trigger",
  start: "top center",
  end: "+=500",
  onUpdate: (self) => console.log("direction:", self.direction),
});
```
