# Rules for vibe coding

## Code structure & organization

- **Keep code DRY (Don't Repeat Yourself)**
  - Extract repeated logic into reusable functions
  - Create utility functions for common operations (validation, formatting, etc.)
  - Use shared components for UI patterns that appear multiple times

- **Break down large files**
  - Split files larger than 300-400 lines into smaller modules
  - Separate concerns: data fetching, business logic, UI rendering
  - Create focused components that do one thing well

- **Use logical file organization**
  - Group related files by feature or domain
  - Create separate directories for components, utilities, services, etc.
  - Follow consistent naming conventions across the project

## Security practices

- **Input validation and sanitization**
  - Validate all user inputs on both client and server sides
  - Use parameterized queries for database operations
  - Sanitize any data before rendering it to prevent XSS attacks

- **Authentication & authorization**
  - Protect sensitive routes with authentication middleware
  - Implement proper authorization checks for data access
  - Use role-based permissions for different user types

- **API security**
  - Implement rate limiting on authentication endpoints
  - Set secure HTTP headers (CORS, Content-Security-Policy)
  - Use HTTPS for all connections

- **Secrets management**
  - Never hardcode secrets or credentials in source code
  - Store sensitive values in environment variables
  - Use secret management services for production environments

## Error handling

- **Implement comprehensive error handling**
  - Catch and handle specific error types differently
  - Log errors with sufficient context for debugging
  - Present user-friendly error messages in the UI

- **Handle async operations properly**
  - Use try/catch blocks with async/await
  - Handle network failures gracefully
  - Implement loading states for better user experience

## Performance optimization

- **Minimize expensive operations**
  - Cache results of costly calculations
  - Use memoization for pure functions
  - Implement pagination for large data sets

- **Prevent memory leaks**
  - Clean up event listeners and subscriptions
  - Cancel pending requests when components unmount
  - Clear intervals and timeouts when no longer needed

- **Optimize rendering**
  - Avoid unnecessary re-renders
  - Use virtualization for long lists
  - Implement code splitting and lazy loading

## Database best practices

- **Use transactions for related operations**
  - Wrap related database operations in transactions
  - Ensure data consistency across multiple operations
  - Implement proper rollback mechanisms

- **Optimize queries**
  - Create indexes for frequently queried fields
  - Select only the fields you need
  - Use query pagination when fetching large datasets

- **Handle database connections properly**
  - Use connection pools
  - Close connections when operations complete
  - Implement retry mechanisms for transient failures

## API design

- **Follow RESTful principles**
  - Use appropriate HTTP methods (GET, POST, PUT, DELETE)
  - Return consistent response formats
  - Use meaningful HTTP status codes

- **Design clear endpoints**
  - Organize endpoints by resource
  - Version your API
  - Document all endpoints with examples

- **Implement proper error responses**
  - Return structured error objects
  - Include error codes and helpful messages
  - Maintain detailed logs of API errors

## Maintainability

- **Use clear naming**
  - Choose descriptive variable, function, and class names
  - Avoid abbreviations and cryptic naming
  - Use consistent naming patterns throughout the codebase

- **Add documentation**
  - Document complex functions with clear descriptions
  - Explain the "why" not just the "what"
  - Keep documentation up-to-date when code changes

- **Write tests**
  - Cover critical business logic with unit tests
  - Write integration tests for important flows
  - Implement end-to-end tests for critical user journeys

## Frontend specific

- **Implement form validation**
  - Validate input as users type
  - Provide clear error messages
  - Handle form submission errors gracefully

- **Use proper state management**
  - Choose appropriate state management for your app's complexity
  - Avoid prop drilling through many component levels
  - Keep state as close as possible to where it's needed

- **Ensure accessibility**
  - Use semantic HTML elements
  - Add proper ARIA attributes for complex elements
  - Ensure keyboard navigability
  - Maintain sufficient color contrast

## Security vulnerabilities to prevent

- **SQL/NoSQL injection**
  - Never concatenate user input directly into queries
  - Use parameterized queries or ORM methods

- **Cross-site scripting (XSS)**
  - Sanitize user input before displaying it
  - Use frameworks' built-in protection mechanisms

- **Cross-site request forgery (CSRF)**
  - Implement anti-CSRF tokens
  - Validate request origins

- **Broken authentication**
  - Implement proper session management
  - Use secure password hashing
  - Enforce strong password policies



Jason ✨👾SaaStr.Ai✨ Lemkin
@jasonlk
## Start with a throwaway hack. 

* Spend 60 minutes max telling a vibe coding app your wildest product dreams without any planning. See what emerges. 
* But commit upfront to throwing it away—this isn't your real product, it's your education. 
* That first hour will teach you more about platform capabilities and limitations than any tutorial.

## Before writing any code, spend a full week studying 20 production apps built on vibe coding platforms. 

* Not casual browsing—actually use apps that are live, taking payments, serving real customers. 
* You're looking for what's genuinely possible at scale and where limitations bite hardest. 
* This reconnaissance saves weeks of frustration later.

## Define your production requirements before you start building. 

Ask: 
1. How secure does this need to be? 
2. Who will maintain it after launch? 
3. Do you need it to scale to 100 users or 100,000? 
4. Did you find another vibe-coded app in production, with paying customers, at your complexity level? 

If you don't have solid answers, stop building and start researching.

## Write the most detailed specification you can manage. 

* Map every page, workflow, permission level. 
* Define email systems, dashboards, user management flows explicitly. 
* Yes, this seems counterintuitive for natural language prompts, but it forces you to think through edge cases and becomes your north star when AI suggests unwanted features.

## Some features look simple in demos but become really big engineering challenges.

Examples today at least (and this is constantly changing):

▶️ reliable email delivery
▶️ OAuth/identity management
▶️ media generation
▶️ native mobile apps
▶️ custom design beyond templates
▶️ enterprise security. 

* These consistently cause pain across platforms. Plan extra time or consider if they're actually necessary for MVP.
* Don't assume your static demo that seems to do these thing well really does them well.
* Find a seasoned engineer that has built on your platform and ASK them.  ASK them.

## AI systems fabricate data when they fail. 

* Everyone that has worked on ANY vibe coding platform, including Claude Code, knows this.
* It is a bug but also a feature. Without this, they can't solve problems.  
* An AI on ANY platform when it hits roadblocks will generate fictional data.
* This isn't a bug—they're trained to provide output rather than admit failure. After multiple failed attempts, they'll create convincing fake data instead of saying "I can't do this."
* You need to understand this, accept it, and work around it.  This will take time.

##  Spend your first full day learning every platform feature, not building. 

* These platforms pack tremendous functionality into their interfaces. 
* Every icon, menu option, feature exists for a reason. 
* You can't leverage capabilities you don't know exist. 
* This isn't optional research—it's essential knowledge for commercial-grade apps.
* There isn't a solution to every challenge.  But the platforms have more solutions that you will think at first.
* And they are kind of nerdy.  In a good way, but nerdy.  Deep down they were built for developers, no matter what the marketing says.
* Accept that and get to know EVERY feature before you start.  If you don't understand a feature, an icon, an acronym, then STOP.
* Go research it.  Now.  Not later.

## Master rollback systems on day one, before you need them desperately. 

* Most platforms offer elegant version control much like video game save points. 
* Practice rolling back intentionally while stakes are low. 
* Understand exactly how it works, what gets preserved, what gets lost. This becomes your most valuable debugging tool.

##  AI will make changes you didn't request.   It just will.

* It'll modify settled features, add unwanted functionality, break working code while "improving" something else. 
* Defense: Add "NO CHANGES WITHOUT ASKING" to every prompt. When discussing changes, state "NO CHANGES. NO CODE. JUST DISCUSSION." Reduces unwanted modifications ~80%.  But it doesn't stop them.
* This is true of every platform. In the end, they all run on Claude -- mostly.  They all have varying levels of the same issues from that.  
* They will >all< make changes you didn't request.  It's just the more prosumer apps will go further, since the developer-focused coding apps are more isolated in terms of the changes they make.

## Learn to fork your application when it reaches stable complexity. 

* Early on, rollbacks handle most issues. But as your app grows complex, you might not know which version to roll back to. 
* Fork at stable states to create safe experimentation branches while preserving known-good versions. Think insurance policies.

## Budget 150 hours across a full month to reach commercial quality.   Maybe more.

▶️That 20-minute prototype is 5% of your actual work. 
▶️More than half your time will be testing, debugging, refinement. 

* The initial build is easy—making it reliable, secure, user-friendly requires the majority of effort. 
* Don't let demo speed fool you.

## Accept your new role as QA engineer. 

Once you're days into serious development, expect daily routine of: 

▶️ taking bug screenshots
▶️ writing detailed reports for AI
▶️ testing partial fixes
▶️ retesting edge cases
▶️ documenting new issues
▶️ running unit tests on your fork

* This isn't a vibe coding limitation—it's software development reality. Platforms handle coding; QA remains human work.  
* The platforms do do ... some.  But only some.  You can't rely on them to do your QA alone.

## Plan your exit strategy from day one. 

* Most commercial apps eventually outgrow prosumer vibe coding platforms due to scale, customization, or security needs. 

Options: 
1. platform code export
2. hybrid approach
3. complete rebuild, or ... 
4. staying and scaling. 

* The truth is, on the prosumer apps today, most leave.   
* Not all.  But most that are building true commercial-grade apps.  For now.
* This doesn't mean you have to.  But have >options< when you start.  Have ... an exit plan if you need it.
* Document business logic, maintain specs, evaluate regularly. 
* If your app gets complex, in the end, you may find it easier to leave than work around accumulating constraints.
