# .isActive

### .isActive : Boolean

\[read-only] Only `true` if the scroll position is between the start and end positions of the ScrollTrigger instance.

### Details[​](#details "Direct link to Details")

\[read-only] Only `true` if the scroll position is between the start and end positions of the ScrollTrigger instance.

## Example[​](#example "Direct link to Example")

```
ScrollTrigger.create({
  trigger: ".trigger",
  start: "top center",
  end: "+=500",
  onToggle: (self) => console.log("toggled. active?", self.isActive),
});
```
