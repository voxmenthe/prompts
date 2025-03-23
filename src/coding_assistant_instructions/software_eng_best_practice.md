This helps humans *and* LLMs:

- consistent naming
- APIs that make it hard to do the wrong thing
- clear code guidelines
- using well known design patterns
- logging and monitoring
- additional tooling: linting, unit tests, etc...
- documentation (tutorials, overview documents, reference documents)
- RFC/ADRs explaining decisions made
- clean git commit logs

vibe coding on big codebases is perfectly fine and not all that difficult. Use common practices around big software architecture:

- RFCs/ADRs
- modularization
- clear boundaries
- logging and monitoring

You do need a solid knowledge of software engineering to find the right patterns. This is ever more crucial.

1. event driven architecture and DDD

Software Architecture - the hard parts - https://oreilly.com/library/view/software-architecture-the/9781492086888/ 

Learning Domain Driven Design - https://oreilly.com/library/view/learning-domain-driven-design/9781098100124/

Designing Event Driven Systems - https://oreilly.com/library/view/designing-event-driven-systems/9781492038252/

Event sourcing and DDD (domain driven design) allow you to align architectural boundaries around domain language. The LLMs knows both, and is thus able to align your software with your goals.

2. Design patterns and software architecture

Martin Fowler - Refactoring - https://martinfowler.com/books/refactoring.html

Martin Fowler - Patterns of Enterprise Application Architecture - https://martinfowler.com/eaaCatalog/

Enterprise Integration Patterns - https://enterpriseintegrationpatterns.com

This stuff seems super dry and boring and from a previous era, but they are a GOLDMINE. However, try to use more domain specific names to avoid LLM slop and confusion, as ManagerFactoryMessageController is just overloaded in the training corpus.

3. Language construction and compiler / interpreter knowledge

Being able to manipulate code, telling LLMs how to manipulate code, writing custom linters and code generators is a super power in the LLM age. It is much easier to vibe code a deterministic code manipulator than asking the LLM to recreate code everytime.

Paradigms of artificial intelligence programming - https://norvig.github.io/paip-lisp/#/

This is the best programming book I know. It is not about AI programming. It is about writing crystal clear code, using symbolic data structures, going from a simple pattern matcher to an optimizing prolog compiler.

This is *THE* book I recommend for every programmer. It is absolutely wonderful, fun and chock full of concrete pragmatic advice.

4. Learn software best practices

Especially if you are new to vibe coding and programming, this book will change your life:

The pragmatic programmer - https://pragprog.com/titles/tpp20/the-pragmatic-programmer-20th-anniversary-edition/

It will teach you about manipulating code in a professional environment. And when you vibe code, you are immediately catapulted in a "professional" environment. You have to interface with multiple colleagues (the LLM agents) working on the same codebase at the same time. Version control, requirements, pitfalls, patterns, unit testing. It's all there. It's all necessary.

5. Learn how to deploy and monitor your applications

Because you are able to put stuff into production so quickly, you can't escape learning best practices, security, infrastructure.

I don't have the best resources here, but can recommend a lot of the pragmatic programmer, manning, oreilly literature. Especially newer books are of impressive quality compared to what I had growing up.

6. I have more to add and sure I forget a ton. But as a vibe coder, you are a software architect. Software architecture is all about connecting people. While some of these people are now LLM agents, you are ultimately still the interface between humans and machine.

Traditional software architect literature is directly applicable.

12 Essential Skills for Software Architects - https://amazon.com/12-Essential-Skills-Software-Architects/dp/0321717295

12 More Essential Skills for Software Architects - https://amazon.com/More-Essential-Skills-Software-Architects/dp/032190947X/ref=sr_1_1

These are not about technical software architecture, but about the role of software architect within a company/team/product workflow.

