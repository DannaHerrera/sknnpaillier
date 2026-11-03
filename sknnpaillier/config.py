"""Resolve the supported security-level settings into Paillier modulus sizes."""


def resolve_key_bits(security_level: str | int) -> int:
    """Resolve a supported security level to a modulus length.

    The classical level mapping follows the NIST IFC/RSA size correspondence
    as a project policy, not a certification of Paillier. The test setting
    provides no practical security; level 80 is for demonstrations only.

    Args:
        security_level (str | int): Exactly "test", 80, 112, or 128; numeric
            levels may also be strings.

    Returns:
        int: Modulus bits: 128 for "test", 1024 for 80, 2048 for 112,
            or 3072 for 128.

    Raises:
        ValueError: If the value is not a supported level, including booleans,
            raw modulus sizes, aliases, or floating-point values.
    """
    levels = {"test": 128, "80": 1024, "112": 2048, "128": 3072}
    if isinstance(security_level, bool) or not isinstance(
        security_level, (str, int)
    ):
        raise ValueError("SECURITY_LEVEL debe ser 'test', 80, 112 o 128.")
    value = str(security_level)
    if value not in levels:
        raise ValueError(
            f"SECURITY_LEVEL={security_level!r} no soportado; "
            "usa 'test', 80, 112 o 128 (no bits de módulo)."
        )
    return levels[value]
