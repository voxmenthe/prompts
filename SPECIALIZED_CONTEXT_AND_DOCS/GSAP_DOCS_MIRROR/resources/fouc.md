# Avoiding FOUC

Have you ever noticed an annoying *"flash of unstyled content"* (FOUC) when a web page first loads? That happens because browsers render things as quickly as possible, often **BEFORE** your JavaScript executes the first time. So what if some of your initial styles are set via JavaScript...like with GSAP?

**Solution**: apply `visibility: hidden;` to your elements in CSS and then use GSAP's `autoAlpha` property to show it (or animate it in) when the page loads. `autoAlpha` affects `opacity` *and* `visibility`, changing it to `visible` when the `opacity` is greater than 0. Pretty convenient!

Video Walkthrough

Check out this video from the ["GSAP 3 Express" course](https://courses.snorkl.tv/courses/gsap-3-express?ref=44f484) by Snorkl.tv - one of the best ways to learn the basics of GSAP 3:

warning

To make sure it works in browsers that don't have JavaScript enabled, you can undo the hiding inside of [`<noscript>` tags](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/noscript).
