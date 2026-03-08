# ADR-006: Migration of Regression Analysis API to Domain-Driven Design

## Status

- [x] Proposed
- [ ] Accepted
- [ ] Deprecated

## Context

The current rustgression library uses a getter/setter pattern with data classes (`OlsRegressionParams`, `TlsRegressionParams`). This approach has the following problems:

### Current Problems

1. **Violation of the Tell Don't Ask principle**: The codebase heavily relies on querying object state and then performing operations based on the result.
2. **Use of getters**: The existence of `get_xxx()`-style methods risks introducing setters in the future, which would break object encapsulation. To prevent this, the library design should be unified around behaviour-oriented domain objects rather than the getter/setter pattern.

### Current API Example

```python
# Current approach — requires domain knowledge on the client side
regressor = OlsRegressor(x, y)
params = regressor.get_params()
```

## Decision

Redesign the regression analysis API based on Domain-Driven Design (DDD) principles, migrating to a behaviour-centric API.

### 1. Introduce Property Method Pattern

Replace the getter/setter pattern on data classes with property methods as behaviour:

- `slope() -> float`
- `intercept() -> float`
- `r_value() -> float`
- `p_value() -> float`
- `stderr() -> float`
- `intercept_stderr() -> float`

### 2. Design Based on Single Responsibility Principle

Change the Regressor class itself to expose statistical values as property methods:

```python
class OlsRegressor:
    def __init__(self, x, y):
        # Run regression and compute statistics at instantiation time
        self._slope, self._intercept, self._r_value, self._p_value, self._stderr, self._intercept_stderr = self._calculate_regression(x, y)

    def slope(self) -> float:
        """Return the slope of the regression line."""
        return self._slope

    def intercept(self) -> float:
        """Return the y-intercept of the regression line."""
        return self._intercept

    def r_value(self) -> float:
        """Return the correlation coefficient."""
        return self._r_value

    def p_value(self) -> float:
        """Return the p-value for the slope."""
        return self._p_value

    def stderr(self) -> float:
        """Return the standard error of the slope."""
        return self._stderr

    def intercept_stderr(self) -> float:
        """Return the standard error of the intercept."""
        return self._intercept_stderr
```

### 3. Maintain Backward Compatibility

To preserve compatibility with the existing API:

- Existing data classes (`OlsRegressionParams`, `TlsRegressionParams`) are retained
- `get_params()` method is kept as deprecated
- Supports gradual migration

### 4. Implementation Strategy

#### Phase 1: Foundation

- Implement property methods on `OlsRegressor` and `TlsRegressor` classes
- Add statistical value computation at instantiation time

#### Phase 2: Compatibility

- Deprecate the existing `get_params()` method
- Update documentation to support gradual migration

#### Phase 3: Tests and Documentation

- Add tests for the new API
- Update usage examples and documentation
- Create migration guide

## Consequences

### Expected Benefits

1. **Application of Single Responsibility Principle**: The Regressor class solely owns the computation and provision of statistical values.
2. **Behaviour-centric design**: Property methods provide data access as behaviour.
3. **Improved API consistency**: Unified access pattern for all statistical values.
4. **Performance improvement**: Computation is performed only once at instantiation.
5. **Preservation of existing functionality**: All currently provided statistical values (slope, intercept, r_value, p_value, stderr, intercept_stderr) continue to be available.

### New API Example

```python
# New approach — access property methods directly from the Regressor class
regressor = OlsRegressor(x, y)  # statistics computed at instantiation

# Statistical values accessed as behaviour
print(f"Slope: {regressor.slope():.3f}")
print(f"Intercept: {regressor.intercept():.3f}")
print(f"Correlation coefficient: {regressor.r_value():.3f}")
print(f"p-value: {regressor.p_value():.6f}")
print(f"Standard error of slope: {regressor.stderr():.3f}")
print(f"Standard error of intercept: {regressor.intercept_stderr():.3f}")
```

### Breaking Changes

- Introduction of new property method API
- Direct property method addition to Regressor classes
- Deprecation of `get_params()` method (gradual migration)

### Impact Scope

- Existing client code: minimal impact due to backward compatibility
- Test code: new API tests need to be added
- Documentation: new API description and migration guide needed

## References

- [Domain-Driven Design: Tackling Complexity in the Heart of Software](https://www.amazon.com/Domain-Driven-Design-Tackling-Complexity-Software/dp/0321125215)
- [Tell Don't Ask Principle](https://martinfowler.com/bliki/TellDontAsk.html)
- [Anemic Domain Model Anti-pattern](https://martinfowler.com/bliki/AnemicDomainModel.html)
- [Python Enum Documentation](https://docs.python.org/3/library/enum.html)

## Related File Paths

### Initial implementation (2025-07-30)

To be updated

- `rustgression/regression/base_regressor.py`
- `rustgression/regression/ols_regressor.py`
- `rustgression/regression/tls_regressor.py`
- `rustgression/__init__.py` (export new classes)
- `tests/test_regressor.py` (add tests for new API)
- `examples/scientific_example.py` (usage examples for new API)
- `examples/simple_example.py` (usage examples for new API)
- `docs/ja/development.md` (update API usage examples)
- `docs/en/development.md` (update API usage examples)
