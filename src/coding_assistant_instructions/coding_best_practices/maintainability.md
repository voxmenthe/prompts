Here are the key practices, structures, and patterns that lead to maintainable, debuggable, and extensible code in Python and TypeScript/JavaScript:

## Core Design Principles

**SOLID Principles**
- **Single Responsibility**: Each module/class/function should do one thing well
- **Open/Closed**: Code should be open for extension but closed for modification
- **Liskov Substitution**: Derived classes must be substitutable for their base classes
- **Interface Segregation**: Many specific interfaces are better than one general-purpose interface
- **Dependency Inversion**: Depend on abstractions, not concretions

**DRY (Don't Repeat Yourself)**
- Extract common functionality into reusable functions/modules
- Use configuration files for constants and settings
- Create shared utilities for cross-cutting concerns

## Code Organization

**Module Structure**
```python
# Python
project/
├── src/
│   ├── __init__.py
│   ├── models/
│   ├── services/
│   ├── utils/
│   └── config/
├── tests/
├── docs/
└── requirements.txt
```

```typescript
// TypeScript
project/
├── src/
│   ├── components/
│   ├── services/
│   ├── utils/
│   ├── types/
│   └── config/
├── tests/
├── docs/
└── package.json
```

**Principles demonstrated**: This structure follows the **Single Responsibility Principle** by organizing code into focused directories. Each folder has a clear purpose (models for data structures, services for business logic, utils for helpers). This organization also supports the **Open/Closed Principle** by making it easy to add new features without modifying existing structure.

**Alternative patterns for different coding styles**:
- **Domain-Driven Design**: Organize by business domains (`user/`, `order/`, `payment/`) instead of technical layers
- **Feature-based**: Group all related code by feature (`features/authentication/`, `features/checkout/`)
- **Hexagonal Architecture**: Use `core/`, `adapters/`, `ports/` for stronger separation between business logic and infrastructure

**Clear Separation of Concerns**
- Separate business logic from presentation
- Keep data access isolated in repository/service layers
- Use dependency injection for loose coupling

## Type Safety and Contracts

**TypeScript**
```typescript
// Define clear interfaces
interface UserService {
  getUser(id: string): Promise<User>;
  updateUser(id: string, data: Partial<User>): Promise<User>;
}

// Use discriminated unions for error handling
type Result<T> = 
  | { success: true; data: T }
  | { success: false; error: Error };

// Leverage type guards
function isValidUser(obj: unknown): obj is User {
  return typeof obj === 'object' && obj !== null && 'id' in obj;
}
```

**Principles demonstrated**: 
- **Interface Segregation**: The `UserService` interface defines only what's needed for user operations
- **Dependency Inversion**: Code depends on the interface abstraction, not concrete implementations
- **Type Safety**: Discriminated unions prevent runtime errors by making invalid states unrepresentable

**Alternative patterns**:
- **Branded Types**: Use branded/opaque types for stronger typing (e.g., `UserId` instead of `string`)
- **Effect System**: Libraries like `fp-ts` or `Effect` for functional error handling
- **Zod Schemas**: Runtime validation that generates TypeScript types automatically

**Python**
```python
# Use type hints (Python 3.5+)
from typing import Optional, List, Protocol

class UserRepository(Protocol):
    def get_user(self, user_id: str) -> Optional[User]:
        ...
    
    def update_user(self, user_id: str, data: dict) -> User:
        ...

# Use dataclasses for clear data structures
from dataclasses import dataclass

@dataclass
class User:
    id: str
    name: str
    email: str
```

**Principles demonstrated**:
- **Duck Typing with Safety**: Protocols provide interface-like behavior while maintaining Python's flexibility
- **Data Integrity**: Dataclasses ensure consistent data structures with less boilerplate
- **Self-Documenting Code**: Type hints serve as inline documentation

**Alternative patterns**:
- **Pydantic Models**: For runtime validation and serialization (`class User(BaseModel):`)
- **TypedDict**: For dictionary structures with known keys
- **ABC (Abstract Base Classes)**: More traditional inheritance-based interfaces
- **attrs**: More powerful alternative to dataclasses with validators

## Error Handling and Debugging

**Explicit Error Handling**
```python
# Python - Custom exceptions
class ValidationError(Exception):
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

# Use context managers for resource management
from contextlib import contextmanager

@contextmanager
def database_transaction():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

**Principles demonstrated**:
- **Fail Fast**: Custom exceptions make errors explicit and meaningful
- **Resource Safety**: Context managers guarantee cleanup even on failure
- **Single Responsibility**: Transaction logic is isolated from business logic

**Alternative patterns**:
- **Monadic Error Handling**: Use `Result` types (like Rust) via libraries like `returns`
- **Decorator-based**: `@retry`, `@timeout` decorators for cross-cutting concerns
- **Aspect-Oriented**: Use aspects for logging/monitoring without cluttering business logic

```typescript
// TypeScript - Result pattern
class Result<T, E = Error> {
  private constructor(
    private readonly value: T | null,
    private readonly error: E | null
  ) {}

  static ok<T>(value: T): Result<T> {
    return new Result(value, null);
  }

  static err<E>(error: E): Result<never, E> {
    return new Result(null, error);
  }

  isOk(): boolean {
    return this.error === null;
  }
}
```

**Principles demonstrated**:
- **Make Invalid States Unrepresentable**: Result can only be success OR failure, never both
- **Explicit Error Handling**: Forces callers to handle errors
- **Functional Paradigm**: Encourages composition over try-catch pyramids

**Alternative patterns**:
- **Async/Await with Try-Catch**: Traditional but less composable
- **Neverthrow Library**: Provides Result type with more utilities
- **RxJS Observables**: For complex async flows with built-in error handling

**Logging and Observability**
```python
# Python
import logging
import functools

def log_execution(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {e}", exc_info=True)
            raise
    return wrapper
```

**Principles demonstrated**:
- **Cross-Cutting Concerns**: Logging separated from business logic via decorators
- **Observability**: Automatic tracing of function calls and errors
- **DRY**: Reusable logging logic across all functions

**Alternative patterns**:
- **Structured Logging**: Use JSON logs with `structlog` for better parsing
- **OpenTelemetry**: For distributed tracing across services
- **Context Managers**: For timing and resource tracking
- **Middleware Pattern**: In web frameworks for request/response logging

## Testing Strategies

**Test Structure**
```python
# Python - pytest
class TestUserService:
    @pytest.fixture
    def mock_repository(self):
        return Mock(spec=UserRepository)
    
    @pytest.fixture
    def service(self, mock_repository):
        return UserService(mock_repository)
    
    def test_get_user_success(self, service, mock_repository):
        # Arrange
        expected_user = User(id="123", name="Test")
        mock_repository.get_user.return_value = expected_user
        
        # Act
        result = service.get_user("123")
        
        # Assert
        assert result == expected_user
        mock_repository.get_user.assert_called_once_with("123")
```

**Principles demonstrated**:
- **Arrange-Act-Assert**: Clear test structure for readability
- **Dependency Injection**: Easy mocking through constructor injection
- **Test Isolation**: Each test is independent with fresh fixtures

**Alternative patterns**:
- **Property-Based Testing**: Use `hypothesis` for generating test cases
- **BDD Style**: Use `behave` or `pytest-bdd` for behavior-driven tests
- **Integration Tests**: Test real dependencies with `testcontainers`
- **Snapshot Testing**: For complex outputs, use `pytest-snapshot`

```typescript
// TypeScript - Jest
describe('UserService', () => {
  let service: UserService;
  let mockRepository: jest.Mocked<UserRepository>;

  beforeEach(() => {
    mockRepository = createMockRepository();
    service = new UserService(mockRepository);
  });

  it('should get user successfully', async () => {
    // Arrange
    const expectedUser = { id: '123', name: 'Test' };
    mockRepository.getUser.mockResolvedValue(expectedUser);

    // Act
    const result = await service.getUser('123');

    // Assert
    expect(result).toEqual(expectedUser);
    expect(mockRepository.getUser).toHaveBeenCalledWith('123');
  });
});
```

**Principles demonstrated**:
- **Test Organization**: Nested describes for logical grouping
- **Setup/Teardown**: beforeEach ensures clean state
- **Async Testing**: Proper handling of promises in tests

**Alternative patterns**:
- **Testing Library**: For React components with user-centric tests
- **Playwright/Cypress**: For E2E testing with real browsers
- **Vitest**: Faster alternative to Jest with ESM support
- **Contract Testing**: Use `Pact` for consumer-driven contracts

## Common Design Patterns

**Factory Pattern**
```python
# Python
class DatabaseFactory:
    @staticmethod
    def create(db_type: str) -> Database:
        if db_type == "postgres":
            return PostgresDatabase()
        elif db_type == "mysql":
            return MySQLDatabase()
        else:
            raise ValueError(f"Unknown database type: {db_type}")
```

**Principles demonstrated**:
- **Open/Closed**: Easy to add new database types without modifying existing code
- **Dependency Inversion**: Client code depends on Database interface, not concrete classes
- **Single Responsibility**: Object creation logic is centralized

**Alternative patterns**:
- **Registry Pattern**: Self-registering classes for more flexibility
- **Builder Pattern**: For complex object construction with many parameters
- **Dependency Injection Container**: IoC containers like `dependency-injector`
- **Abstract Factory**: When you need families of related objects

**Strategy Pattern**
```typescript
// TypeScript
interface PaymentStrategy {
  processPayment(amount: number): Promise<PaymentResult>;
}

class PaymentProcessor {
  constructor(private strategy: PaymentStrategy) {}

  async process(amount: number): Promise<PaymentResult> {
    return this.strategy.processPayment(amount);
  }
}
```

**Principles demonstrated**:
- **Open/Closed**: Add new payment methods without changing processor
- **Single Responsibility**: Each strategy handles one payment type
- **Composition over Inheritance**: Strategies are composed, not inherited

**Alternative patterns**:
- **Chain of Responsibility**: For sequential processing with fallbacks
- **Command Pattern**: When you need to queue/undo operations
- **Template Method**: For algorithms with common structure but varying steps
- **Policy Pattern**: Similar to strategy but for business rules

**Repository Pattern**
```python
# Python
from abc import ABC, abstractmethod

class Repository(ABC):
    @abstractmethod
    async def find_by_id(self, id: str) -> Optional[Entity]:
        pass
    
    @abstractmethod
    async def save(self, entity: Entity) -> Entity:
        pass

class UserRepository(Repository):
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def find_by_id(self, id: str) -> Optional[User]:
        # Implementation
        pass
```

**Principles demonstrated**:
- **Separation of Concerns**: Business logic separated from data access
- **Testability**: Easy to mock repositories for unit tests
- **Flexibility**: Switch databases without changing business logic

**Alternative patterns**:
- **Active Record**: Entities that know how to persist themselves
- **Data Mapper**: More complex separation between domain and persistence
- **CQRS**: Separate read and write repositories
- **Unit of Work**: Coordinate multiple repositories in transactions

## Configuration Management

**Environment-based Configuration**
```python
# Python
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str
    api_key: str
    debug: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()
```

```typescript
// TypeScript
interface Config {
  databaseUrl: string;
  apiKey: string;
  debug: boolean;
}

const config: Config = {
  databaseUrl: process.env.DATABASE_URL || 'default',
  apiKey: process.env.API_KEY || '',
  debug: process.env.NODE_ENV !== 'production'
};
```

**Principles demonstrated**:
- **Externalized Configuration**: Settings separate from code
- **Type Safety**: Configuration validated at startup
- **12-Factor App**: Environment-based config for cloud deployment

**Alternative patterns**:
- **Config Files**: YAML/JSON files for complex configurations
- **Feature Flags**: Dynamic configuration with tools like LaunchDarkly
- **Vault Integration**: HashiCorp Vault for secrets management
- **Config Server**: Spring Cloud Config or similar for distributed systems

## Documentation and Code Comments

**Self-Documenting Code**
```python
# Python
def calculate_discount(
    price: Decimal,
    discount_percentage: Decimal,
    max_discount: Optional[Decimal] = None
) -> Decimal:
    """
    Calculate the discounted price.
    
    Args:
        price: Original price
        discount_percentage: Discount as a percentage (0-100)
        max_discount: Maximum discount amount allowed
    
    Returns:
        The final price after discount
    
    Raises:
        ValueError: If discount_percentage is not between 0 and 100
    """
    if not 0 <= discount_percentage <= 100:
        raise ValueError("Discount percentage must be between 0 and 100")
    
    discount_amount = price * (discount_percentage / 100)
    if max_discount:
        discount_amount = min(discount_amount, max_discount)
    
    return price - discount_amount
```

**Principles demonstrated**:
- **Clear Intent**: Function name and parameters explain what it does
- **Type Hints**: Parameters and return types are self-documenting
- **Fail Fast**: Input validation with clear error messages

**Alternative documentation approaches**:
- **Examples in Docstrings**: Include usage examples in documentation
- **Type Aliases**: Create semantic types like `Price = Decimal`
- **Sphinx/MkDocs**: Generate documentation from docstrings
- **ADRs**: Architecture Decision Records for design choices

## Functional Programming Patterns

**Immutability and Pure Functions**
```typescript
// TypeScript
// Instead of mutating
function badAddItem(cart: Cart, item: Item): void {
  cart.items.push(item); // Mutation!
}

// Use immutable updates
function addItem(cart: Cart, item: Item): Cart {
  return {
    ...cart,
    items: [...cart.items, item]
  };
}
```

**Principles demonstrated**:
- **Predictability**: Pure functions always return same output for same input
- **Testability**: No side effects make testing straightforward
- **Concurrency Safety**: Immutable data prevents race conditions

**Alternative patterns**:
- **Immer**: Write mutable-style code that produces immutable updates
- **Immutable.js**: Persistent data structures for performance
- **Ramda/Lodash-FP**: Functional utility libraries
- **Lenses**: For nested immutable updates (e.g., `monocle-ts`)

**Composition**
```python
# Python
from functools import reduce
from typing import Callable, TypeVar

T = TypeVar('T')

def compose(*functions: Callable[[T], T]) -> Callable[[T], T]:
    def inner(arg: T) -> T:
        return reduce(lambda acc, fn: fn(acc), reversed(functions), arg)
    return inner

# Usage
process_data = compose(
    validate_data,
    transform_data,
    enrich_data
)
```

**Principles demonstrated**:
- **DRY**: Reusable functions combined in different ways
- **Single Responsibility**: Each function does one thing
- **Modularity**: Easy to add/remove/reorder processing steps

**Alternative patterns**:
- **Pipe Operator**: Libraries that provide pipeline syntax
- **Decorators**: For composing behavior around functions
- **Monadic Composition**: Chain operations that might fail
- **Transducers**: Composable algorithmic transformations

## Async/Await Best Practices

**Python**
```python
import asyncio
from typing import List

async def fetch_user_data(user_ids: List[str]) -> List[User]:
    # Concurrent execution
    tasks = [fetch_user(user_id) for user_id in user_ids]
    return await asyncio.gather(*tasks)

# Proper cleanup
async def process_with_cleanup():
    resource = await acquire_resource()
    try:
        return await process(resource)
    finally:
        await resource.close()
```

**Principles demonstrated**:
- **Concurrency**: Parallel execution for better performance
- **Resource Management**: Guaranteed cleanup with try/finally
- **Error Propagation**: Exceptions properly bubble up

**Alternative patterns**:
- **AsyncContextManager**: Use `async with` for automatic cleanup
- **Semaphores**: Limit concurrent operations to prevent overload
- **Task Groups**: Python 3.11+ for better error handling
- **Trio/AnyIO**: Alternative async frameworks with structured concurrency

**TypeScript**
```typescript
// Proper error handling in async functions
async function fetchUserData(userIds: string[]): Promise<User[]> {
  const promises = userIds.map(id => 
    fetchUser(id).catch(err => {
      console.error(`Failed to fetch user ${id}:`, err);
      return null;
    })
  );
  
  const results = await Promise.all(promises);
  return results.filter((user): user is User => user !== null);
}
```

**Principles demonstrated**:
- **Graceful Degradation**: Individual failures don't crash entire operation
- **Type Narrowing**: Filter with type guard ensures type safety
- **Observability**: Errors logged but not swallowed silently

**Alternative patterns**:
- **Promise.allSettled**: Get all results regardless of failures
- **p-limit**: Control concurrency with promise pools
- **Async Iterators**: Process large datasets without memory issues
- **AbortController**: Cancel long-running async operations

These practices work together to create code that is:
- **Maintainable**: Clear structure, consistent patterns, and good documentation
- **Debuggable**: Explicit error handling, comprehensive logging, and testable units
- **Extensible**: Loose coupling, dependency injection, and adherence to SOLID principles

The key is consistency - pick patterns that work for your team and apply them uniformly across your codebase.