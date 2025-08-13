# Webflow

[Webflow Interactions is here!](https://webflow.com/interactions-animations) It's powered by GSAP and it's available for free to all Webflow customers!

Now you have the power to **animate your way.** Build visually with an intuitive horizontal timeline, or drop into custom code for more power and control.

Interested in building interactions right in the Webflow Designer? You’ll get:

* A horizontal timeline with precise, granular controls
* Visual feedback so you can scrub, preview, and perfect
* Built-in support for SplitText, ScrollTrigger, staggers, and more
* The power to reuse animations across your site for faster workflows

Prefer writing code? Get access to the entire GSAP ecosystem, including all the plugins, in your project settings and start coding custom animations right away.

## Choose your adventure...[​](#choose-your-adventure "Direct link to Choose your adventure...")

* I'd like to build visually
* I'd like to write code

<br />

Webflow Interactions is here!

<br />

For product-related issues, technical problems, account questions, and application errors, please go to [support.webflow.com](https://support.webflow.com/) and submit a support request.

For design setup, UI/UX practices, website customization, or custom code please reach out via [forum.webflow.com.](https://forum.webflow.com/)

For more information, see the [guide](https://help.webflow.com/hc/en-us/articles/42832301823635-Intro-to-interactions-with-GSAP) and [glossary](https://help.webflow.com/hc/en-us/articles/42861691922963-Webflow-interactions-with-GSAP-glossary) on Webflow's site.

<br />

Installing GSAP for use with custom code.

<br />

All GSAP files can now be included directly from Webflow's settings, making it nice and easy to get started and make something amazing.

### Installing GSAP

1. Go to your site in Webflow

2. Go to the **settings** panel

3. Click **GSAP integration**

4. Toggle **on** to include the core GSAP library & use the checkboxes to enable plugins.

   ![](/assets/images/webflow-ui-a283b6d437385174b0eb5f82c731c917.png)

### Adding Custom Code

1. Open **Page settings** for the page where you’d like to add your code

2. In the UI find the **Before** `</body>` tag section under **Custom code**

   ![](/assets/images/bodytag-e7c8b2fd0e121058fd4baf0a0c3325fd.png)

3. Ensure that you wrap your GSAP code in a script tag, and a `DOMContentLoaded` event listener.

   ```
   <script>
   addEventListener("DOMContentLoaded", (e) => {

     gsap.to('.my-element', {
      rotation: 360,
      duration: 2,
      ease: 'bounce.out'
     })

   });
   </script>
   ```

4. If you're adding plugins, remember to register them!

   ```
     gsap.registerPlugin(SplitText) 

     let split = SplitText.create(".text", {type: "chars, words"});

     gsap.from(split.chars, {
     	duration: 1, 
     	y: 100, 
     	autoAlpha: 0, 
     	stagger: 0.05
       });
   ```

Per page script loading

Currently the Webflow GSAP integration loads GSAP and the plugins on a site-wide level.

If you would rather load GSAP or a specific plugin on a per page basis we recommend using a script tag along with your JS in the page settings.

## Resources[​](#resources "Direct link to Resources")

### From the webflow community[​](#from-the-webflow-community "Direct link to From the webflow community")

[![](/img/courses/osmo.jpg)](https://www.osmo.supply/)

[Osmo - Effects & Tutorials](https://www.osmo.supply/)

[![](/img/powerup.jpeg)](https://www.youtube.com/watch?v=dbje0UK39g)

[Advanced GSAP x Webflow setup](https://www.youtube.com/watch?v=dbje0UK39g)

[![](/img/custom-ide.jpeg)](https://www.loom.com/share/2d59330cc99c4935abe03815df76641d)

[custom IDE in Webflow](https://www.loom.com/share/2d59330cc99c4935abe03815df76641d)

### Web Bae[​](#web-bae "Direct link to Web Bae")

[![](/img/four.jpeg)](https://www.youtube.com/watch?v=5vQ23CIO6hQ)

[1 animation - 4 ways](https://www.youtube.com/watch?v=5vQ23CIO6hQ)

[![](/img/101.jpeg)](https://www.youtube.com/watch?v=NA-YNultNTo\&list=PLxl8HNCwRbBSD-rBUj-AZeXcGlMZDVXB_\&index=3)

[GSAP 101](https://www.youtube.com/watch?v=NA-YNultNTo\&list=PLxl8HNCwRbBSD-rBUj-AZeXcGlMZDVXB_\&index=3)

[![](/img/stagger.jpeg)](https://www.youtube.com/watch?v=-WALBRG6jp4)

[Stagger from bottom](https://www.youtube.com/watch?v=-WALBRG6jp4)

[Visit Web Bae's channel](https://www.youtube.com/@webbae)

### Timothy Ricks[​](#timothy-ricks "Direct link to Timothy Ricks")

[![](/img/thumbnails/t-ricks-1.jpg)](https://www.youtube.com/watch?v=m4YKslkob9Q)

[ScrollTrigger Crash Course](https://www.youtube.com/watch?v=m4YKslkob9Q)

[![](/img/thumbnails/t-ricks-2.jpg)](https://www.youtube.com/watch?v=rWmQf2KFR4k)

[GSAP + Container Queries 🤯](https://www.youtube.com/watch?v=rWmQf2KFR4k)

[![](/img/thumbnails/t-ricks-3.jpg)](https://www.youtube.com/watch?v=GTS72OESDvw)

[GPT for GSAP & Webflow](https://www.youtube.com/watch?v=GTS72OESDvw)

[More great learning content on his site](https://www.timothyricks.com/)

## **Demos**[​](#demos "Direct link to demos")

If you're looking for inspo, take a look at [all of these amazing Webflow + GSAP sites](https://gsap.com/showcase/?tags=Webflow) or check out these official [GSAP x Webflow Templates](https://webflow.com/templates/search/gsap).

Webflow Demos

Search..

\[x]All

Play Demo videos\[ ]

