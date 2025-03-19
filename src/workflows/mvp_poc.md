One interesting design guideline I have when figuring out AI coding workflows: don't reject "outlandish" ideas. 

Instead, reframe them as "is this possible now that we have LLMs? is this maybe 90% possible? what would it take..."

Too often we dismiss creative solutions before exploring their potential.

Imagine you want an MVP beta for validating a product idea. 

Can you build it in an afternoon? 
Can you have the AI hackslash your codebase, hardcode all the things, replace the database with a CSV file, replace some endpoints with an LLM call altogether? 

Sounds extreme but the ROI can be incredible.

I would never do that to a proper codebase, because the effort (over a few days) wouldn't pay off. Plus redoing it would be just as painful in 3 weeks. 

But if it takes an afternoon, why not?

 I'll have the AI write a nice summary and spec of what we learned, as well as a guide on how to do it again so it one shots it next time. 

I might even have the AI build proper DI/hacking hooks to make this a "clean" thing, configurable with a config file.

Real example: 

We had been struggling for weeks getting our features demoable (complex RAG + indexing pipeline, along with fancy UI features). 

It was demo day -1 and we were about to push the date back when I said "wait, in 2h we can replace the indexing pipeline with a brutal indexing loop, hardcode most UX flows since it's a demo, and add placeholders for things we haven't even started building"

And thus the demo was stellar. 

Really opened my eyes. 

Why had we been struggling and pushing so hard and taking antiquated "shortcuts" to get things ready, when the solution was so close at hand? AI makes these "outlandish shortcuts" not just possible but practical.