# Wordpress

Video guide

## Installation[​](#installation "Direct link to Installation")

1. First up we need to load the GSAP files by enqueueing the scripts in our **functions.php** file. If you're using a GSAP plugin like [ScrollTrigger](/docs/v3/Plugins/ScrollTrigger/.md) you'll add it in here too.<br /><!-- -->Plugins *and* the file you're writing your own animation code in should both be passed ***gsap-js*** as a dependency.

   functions.php

   ```
   <?php
   // The proper way to enqueue GSAP script in WordPress

   // wp_enqueue_script( $handle, $src, $deps, $ver, $in_footer );
   function theme_gsap_script(){
       // The core GSAP library
       wp_enqueue_script( 'gsap-js', 'https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/gsap.min.js', array(), false, true );
       // ScrollTrigger - with gsap.js passed as a dependency
       wp_enqueue_script( 'gsap-st', 'https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/ScrollTrigger.min.js', array('gsap-js'), false, true );
       // Your animation code file - with gsap.js passed as a dependency
       wp_enqueue_script( 'gsap-js2', get_template_directory_uri() . 'js/app.js', array('gsap-js'), false, true );
   }

   add_action( 'wp_enqueue_scripts', 'theme_gsap_script' );
   ?>
   ```

   warning

   Note that if you're using a premium theme or something off the shelf, like [Divi](https://www.elegantthemes.com/gallery/divi/) or one of the [core themes](https://en-gb.wordpress.org/themes/twentytwentyone/), then **editing the \`functions.php\` file directly isn't advisable**: themes like this are updated regularly, and any edits you make - including enqueuing GSAP - will get overwritten.

   Creating a child theme is out of the scope of this article, but the [WordPress Developer Resources for Child Themes](https://developer.wordpress.org/themes/advanced-topics/child-themes/) have plenty of information.

2. In order to animate elements we need access to them. This script below uses an event listener to check that the DOM content is all loaded before running the animation code! If you're having issues - make sure to check your console to make sure the logs are firing.

   App.js

   ```
   // wait until DOM is ready
   document.addEventListener("DOMContentLoaded", function(event){

    console.log("DOM loaded");

    //wait until images, links, fonts, stylesheets, and js is loaded
    window.addEventListener("load", function(e){

     //custom GSAP code goes here
     // This tween will rotate an element with a class of .my-element
      gsap.to('.my-element', {
       rotation: 360,
       duration: 2,
       ease: 'bounce.out'
      })

      console.log("window loaded");
     }, false);

   });
   ```

## Video Guides[​](#video-guides "Direct link to Video Guides")

These tutorials cover the most common ways to use GSAP in wordpress. The first video covers Enqueuing in more detail. The second uses a [scripts organizer plugin](https://dplugins.com/products/scripts-organizer/) and the third has a more visual approach via a GUI with [motion.page](https://motion.page/).

guidance from the wordpress community

[![](/img/enqueue.jpg)Enqueue Scripts in wordpress](https://www.youtube.com/watch?v=26SLt4W1pcY)[![](/img/oxygen.jpg)Oxygen builder and scripts organiser](https://www.youtube.com/watch?v=xbDRQHvSbTs)[![](/img/motion.jpg)Animate using a GUI with motion.page](https://www.youtube.com/watch?v=ghkmOVcdumE)
