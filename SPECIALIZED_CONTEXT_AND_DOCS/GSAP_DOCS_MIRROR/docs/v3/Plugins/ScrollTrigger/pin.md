# .pin

### .pin : Element | undefined

\[read-only] The pin element (if one was defined). If selector text was used, like ".pin", the `pin` will be the element itself (not selector text)

### Details[​](#details "Direct link to Details")

\[read-only] The pin element (if one was defined). If selector text was used, like ".pin", the `pin` will be the element itself (not selector text)

## Example[​](#example "Direct link to Example")

```
let st = ScrollTrigger.create({
  trigger: ".trigger",
  pin: ".pin",
  start: "top center",
  end: "+=500",
});

console.log(st.pin); // pin element (not selector text)
```
