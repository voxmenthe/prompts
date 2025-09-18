# .progress

### .progress : Number

The progress value of the overall page scroll where 0 is at the very top and 1 is at the very bottom and 0.5 is halfway scrolled. This value will animate during the smooth scrolling and end when the `onStop` fires.

### Details[​](#details "Direct link to Details")

The progress value of the overall page scroll where 0 is at the very top and 1 is at the very bottom and 0.5 is halfway scrolled. This value will animate during the smooth scrolling and end when the `onStop` fires.

```
ScrollSmoother.create({
  smooth: 1,
  onUpdate: (self) => console.log("progress", self.progress),
});
```
