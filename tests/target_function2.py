def process_advanced_payment(
    user_id: int,
    amount: float,
    source_currency: str,
    target_currency: str,
    account_balance: float,
    transaction_type: str,
    is_fraud_flagged: bool,
    user_tier: str
) -> str:
    """
    Process a payment transaction with currency conversion, tier-based limits, and audit logging.
    Args:
        user_id: User identifier
        amount: Transaction amount in source currency
        source_currency: Source currency code (e.g., 'USD')
        target_currency: Target currency code (e.g., 'EUR')
        account_balance: User's current balance in source currency
        transaction_type: Type of transaction ('credit', 'debit', 'refund', 'transfer')
        is_fraud_flagged: Fraud detection flag
        user_tier: User tier ('basic', 'premium', 'enterprise')
    Returns:
        Status message ('Success' or error description)
    """
    # Hardcoded exchange rates
    exchange_rates = {
        'USD': {'EUR': 0.85, 'GBP': 0.73},
        'EUR': {'USD': 1.18, 'GBP': 0.86},
        'GBP': {'USD': 1.37, 'EUR': 1.16}
    }

    # Hardcoded blocked users
    blocked_users = [42, 100, 999]

    # Tier-based limits
    tier_limits = {
        'basic': 1000.0,
        'premium': 5000.0,
        'enterprise': 10000.0
    }

    # Validate transaction type
    valid_types = ['credit', 'debit', 'refund', 'transfer']
    if transaction_type not in valid_types:
        raise ValueError("Invalid transaction type")

    # Check blocked users
    if user_id in blocked_users:
        return "Error: User blocked"

    # Validate amount
    if amount <= 0:
        return "Error: Amount must be positive"

    # Validate user tier
    if user_tier not in tier_limits:
        return "Error: Invalid user tier"

    # Currency conversion
    if source_currency != target_currency:
        try:
            rate = exchange_rates[source_currency][target_currency]
            converted_amount = amount * rate
        except KeyError:
            return "Error: Unsupported currency pair"
    else:
        converted_amount = amount

    # Check tier limits
    if converted_amount > tier_limits[user_tier]:
        return "Error: Amount exceeds tier limit"

    # Process transaction
    if transaction_type == 'debit':
        if account_balance < converted_amount:
            return "Error: Insufficient funds"
        new_balance = account_balance - converted_amount
    elif transaction_type == 'credit':
        new_balance = account_balance + converted_amount
    elif transaction_type == 'refund':
        new_balance = account_balance + converted_amount
    elif transaction_type == 'transfer':
        new_balance = account_balance - converted_amount

    if is_fraud_flagged:
        raise Exception("Transaction flagged as fraud")

    # Audit logging
    try:
        with open('audit_log.txt', 'w') as f:
            f.write(f"User {user_id} performed {transaction_type} of {amount} {source_currency} "
                    f"to {converted_amount} {target_currency}, new balance: {new_balance}\n")
    except IOError:
        pass

    return "Success"
