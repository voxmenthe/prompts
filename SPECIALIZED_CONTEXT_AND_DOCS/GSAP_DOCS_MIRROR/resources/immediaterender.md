# immediateRender

The `immediateRender` property of [from()](/docs/v3/GSAP/Timeline/from\(\).md) and [fromTo()](/docs/v3/GSAP/Timeline/fromTo\(\).md) tweens is one of those things you only find out about when it gives you unexpected results.

[from()](/docs/v3/GSAP/Timeline/from\(\).md) and [fromTo()](/docs/v3/GSAP/Timeline/fromTo\(\).md) tweens are special as they set `immediateRender` to `true` as soon as they are created. This helps especially when creating staggered builds where you don't want elements to appear until scheduled. The one case where it is important to change the default behavior is when you have **multiple from()/fromTo() tweens on the same property of the same object**. The video below explains in detail.

Video Walkthrough

<br />

Want more great tips and tricks like this? Our friend Carl over at the [Creative Coding Club](https://www.creativecodingclub.com/bundles/creative-coding-club?ref=44f484) has you covered. Learn the tips and tricks the pros use to make their animations shine. [Join Today](https://www.creativecodingclub.com/bundles/creative-coding-club?ref=44f484).

## **Demos**[​](#demos "Direct link to demos")

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/ZELyRYX?default-tab=result\&theme-id=41164)

the default of `immediateRender: true` however can cause problems when creating multiple from() tweens on the same properties of the same element.

Notice in the demo below that you don't see green animate a second time.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/vYgJjgd?default-tab=result\&theme-id=41164)
