# .vars

### .vars : Object

\[read-only] The vars configuration object used to create the ScrollTrigger instance

### Details[​](#details "Direct link to Details")

\[read-only] The vars configuration object used to create the ScrollTrigger instance.

## Example[​](#example "Direct link to Example")

```
let st = ScrollTrigger.create({
  trigger: ".trigger",
  start: "top center",
  end: "+=500",
});

console.log(st.vars); // {trigger: ".trigger", start: "top center", end: "+=500"}
```

You can store arbitrary data in the `vars` object if you want; ScrollTrigger will just ignore the properties it doesn't recognize. So, for example, you could add a "group" property so that you could group your ScrollTriggers and then later to kill() all the ScrollTrigger instances from a particular group, you could do:

```
// helper function (reusable):
let getGroup = (group) =>
  ScrollTrigger.getAll().filter((t) => t.vars.group === group);

// then, to kill() anything with a group of "my-group":
getGroup("my-group").forEach((t) => t.kill());
```
