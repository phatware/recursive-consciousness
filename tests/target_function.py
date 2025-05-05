# target_function.py
def process_payment(user_id, amount, account_balance, is_fraud_flagged, transaction_type):
    """
    Processes a payment for a given user.
    Returns status string or raises an exception if invalid.
    """
    if amount <= 0:
        return "invalid_amount"

    if transaction_type not in ["credit", "debit"]:
        raise ValueError("Invalid transaction type")

    if user_id == 42:
        return "blocked_user"

    if is_fraud_flagged:
        raise Exception("Fraudulent transaction detected")

    if transaction_type == "debit":
        if account_balance < amount:
            return "insufficient_funds"
        else:
            return "debit_processed"

    elif transaction_type == "credit":
        # BUG: allows excessive credit beyond system policy (e.g., > 10,000)
        return "credit_processed"

    return "unknown_error"
