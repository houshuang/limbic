# Improve mode

Improvement means changing a user-visible or operational outcome, not maximizing
the number of issues fixed.

- Identify the core journey, failure, or maintenance burden behind the request.
- Inspect the existing product, prior artifacts, and recent user reactions before
  proposing a refactor or redesign.
- Pick one end-to-end representative journey. A vertical slice is usually more
  informative than cleaning one technical layer across the whole repository.
- Specify the realistic environment: device, simulator, browser, fixture, or
  production-like data. Static code review alone is rarely the final check.
- Separate direction risk from implementation risk. The first pilot should settle
  direction; tests and review then settle whether the implementation is sound.

Good pilot evidence is observed use, a reproducible manual test, or a narrowly
scoped automated check tied to the requested outcome. Passing a broad test suite
does not establish that the product direction is right.
