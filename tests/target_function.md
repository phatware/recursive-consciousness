# Intentional Features and Bugs to Catch

- Returns a status string instead of raising errors for certain conditions (like "invalid_amount")
- Magic user ID (42) is hardcoded with no explanation
- No upper-bound policy on credit amount (potential policy violation)
- Inconsistent error handling (sometimes return strings, sometimes raise exceptions)
- Business logic scattered in control paths with no shared invariant layer
